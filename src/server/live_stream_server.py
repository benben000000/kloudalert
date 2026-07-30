#!/usr/bin/env python3
"""
Real-Time Live Bataan Weather Telemetry & Liquid Neural Network Server
- Ingests live Open-Meteo telemetry for all 12 Bataan AWS stations every 60 seconds
- Spawns the LFM-230M Self-Improving Agentic Feedback Loop (Predict -> Experiment -> Verify -> Retrain -> Hot-Swap ONNX)
- Computes Haversine / IDW spatial fusion for continuous nowcasting
"""

import os
import sys
import json
import math
import time
import torch
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

# Add src directories to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT / "src" / "models"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "engine"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "data"))

from lfm_foundation_model import LiquidFoundationModel230M
from self_improving_agentic_loop import LFMSelfImprovingAgent
from db_manager import DatabaseManager

STATIONS_PATH = WORKSPACE_ROOT / "src" / "data" / "bataan_stations.json"
PORT = 8085
EARTH_RADIUS_KM = 6371.0088

live_cache = {
    "last_updated": 0,
    "stations_data": {},
    "status": "initializing"
}

# Load Bataan AWS Station Registry
with open(STATIONS_PATH, "r", encoding="utf-8") as f:
    BATAAN_STATIONS = json.load(f)

# Initialize Database Manager & Self-Improving LFM Agent
db_manager = DatabaseManager()
lfm_agent = LFMSelfImprovingAgent()

def haversine_distance(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c

def fetch_live_station_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,surface_pressure,precipitation,rain,wind_speed_10m&timezone=Asia%2FManila"
    req = urllib.request.Request(url, headers={"User-Agent": "LiquidWeatherServer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                return data.get("current", {})
    except Exception as e:
        print(f"[SERVER] Failed to fetch station ({lat}, {lon}): {e}")
    return None

def update_live_telemetry():
    global live_cache
    print("[SERVER] Updating live Open-Meteo telemetry for 12 Bataan AWS stations...")
    results = {}
    for st in BATAAN_STATIONS:
        st_id = st["id"]
        current = fetch_live_station_weather(st["lat"], st["lon"])
        if current:
            results[st_id] = {
                "name": st["name"],
                "lat": st["lat"],
                "lon": st["lon"],
                "temp": current.get("temperature_2m", 30.0),
                "humidity": current.get("relative_humidity_2m", 75.0),
                "pressure": current.get("surface_pressure", 1010.0),
                "precip": current.get("precipitation", 0.0),
                "wind": current.get("wind_speed_10m", 5.0),
                "timestamp": current.get("time", "")
            }
        time.sleep(0.1)

    live_cache = {
        "last_updated": time.time(),
        "stations_count": len(results),
        "stations_data": results,
        "status": "online"
    }
    print(f"[SERVER] Telemetry update complete! {len(results)} stations online.")

    # Trigger Self-Improving Ground-Truth Verification & Retraining Pass
    primary = results.get("AWS-01", next(iter(results.values()), {}))
    if primary:
        retrain_result = lfm_agent.verify_ground_truth_and_improve(primary)
        if retrain_result.get("retrained"):
            print(f"[SERVER] LFM Agent completed online fine-tuning and hot-swapped ONNX binary! Loss: {retrain_result.get('loss'):.4f}")

def background_poller():
    while True:
        try:
            update_live_telemetry()
        except Exception as e:
            print(f"[SERVER] Background polling error: {e}")
        time.sleep(60)

def compute_idw_feature_vector(lat, lon):
    stations_data = live_cache.get("stations_data", {})
    if not stations_data:
        return [30.0, 75.0, 1010.0, 0.0, 5.0, 0.0, 0.0, 36.0]

    weighted_temp = 0.0
    weighted_hum = 0.0
    weighted_press = 0.0
    weighted_precip = 0.0
    weighted_wind = 0.0
    total_weight = 0.0

    for st_id, st in stations_data.items():
        dist = max(haversine_distance(lat, lon, st["lat"], st["lon"]), 0.1)
        w = 1.0 / (dist ** 2)
        weighted_temp += st["temp"] * w
        weighted_hum += st["humidity"] * w
        weighted_press += st["pressure"] * w
        weighted_precip += st["precip"] * w
        weighted_wind += st["wind"] * w
        total_weight += w

    if total_weight > 0:
        t = weighted_temp / total_weight
        h = weighted_hum / total_weight
        p = weighted_press / total_weight
        pr = weighted_precip / total_weight
        wi = weighted_wind / total_weight
        hi = t + 5.5
        return [t, h, p, pr, wi, 0.0, 0.0, hi]

    return [30.0, 75.0, 1010.0, 0.0, 5.0, 0.0, 0.0, 36.0]

class TelemetryAPIHandler(BaseHTTPRequestHandler):
    # In-memory IP rate limiting dictionary {ip: [timestamp1, timestamp2, ...]}
    ip_request_history = {}
    MAX_REQUESTS_PER_MINUTE = 60
    MAX_PAYLOAD_BYTES = 50 * 1024  # 50 KB strict limit for JSON telemetry

    def _check_rate_limit(self):
        client_ip = self.client_address[0]
        now = time.time()
        timestamps = self.ip_request_history.get(client_ip, [])
        # Retain only requests in the last 60 seconds
        timestamps = [ts for ts in timestamps if now - ts < 60]
        
        if len(timestamps) >= self.MAX_REQUESTS_PER_MINUTE:
            return False
            
        timestamps.append(now)
        self.ip_request_history[client_ip] = timestamps
        return True

    def _set_cors(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_cors()

    def do_POST(self):
        if not self._check_rate_limit():
            self._set_cors(429)
            self.wfile.write(json.dumps({"status": "error", "message": "Rate limit exceeded. Maximum 60 requests per minute."}).encode("utf-8"))
            return

        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/v1/telemetry/submit":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > self.MAX_PAYLOAD_BYTES:
                self._set_cors(413)
                self.wfile.write(json.dumps({"status": "error", "message": "Payload Too Large. Maximum 50KB."}).encode("utf-8"))
                return

            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                
                # Input Sanitization & Guardrails
                lat = float(payload.get("latitude", 0.0))
                lon = float(payload.get("longitude", 0.0))
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    raise ValueError("Latitude/Longitude out of valid physical bounds")

                row_id = db_manager.insert_telemetry(payload)
                resp = {
                    "status": "success",
                    "message": "Probe telemetry ingested successfully into Central DB",
                    "telemetry_id": row_id,
                    "timestamp": time.time()
                }
                self._set_cors(200)
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            except Exception as e:
                self._set_cors(400)
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

    def do_GET(self):
        if not self._check_rate_limit():
            self._set_cors(429)
            self.wfile.write(json.dumps({"status": "error", "message": "Rate limit exceeded. Maximum 60 requests per minute."}).encode("utf-8"))
            return

        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/v1/live-bataan-weather":
            self._set_cors()
            self.wfile.write(json.dumps(live_cache).encode('utf-8'))
        elif parsed.path == "/api/v1/nowcast":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                lat = float(query.get("lat", [14.6775])[0])
                lon = float(query.get("lon", [120.5431])[0])
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    raise ValueError("Latitude/Longitude bounds error")
            except Exception:
                lat, lon = 14.6775, 120.5431

            # Construct REAL IDW fused feature sequence matrix (1, 24, 8)
            real_vector = compute_idw_feature_vector(lat, lon)
            real_sequence = [real_vector] * 24

            prob_curve = [0.05] * 18
            try:
                seq_tensor = torch.tensor([real_sequence], dtype=torch.float32)
                with torch.no_grad():
                    out = lfm_agent.model(seq_tensor)
                    prob_curve = out.squeeze(0).tolist()
            except Exception as e:
                print(f"[SERVER] LFM Nowcast error: {e}")

            exp_record = lfm_agent.log_prediction_experiment(lat, lon, real_vector, prob_curve)
            db_manager.insert_experiment(exp_record)

            res = {
                "user_location": {"lat": lat, "lon": lon},
                "experiment_id": exp_record.get("id"),
                "idw_fused_vector": real_vector,
                "prob_curve": prob_curve,
                "max_prob": max(prob_curve),
                "timestamp": time.time()
            }
            self._set_cors()
            self.wfile.write(json.dumps(res).encode('utf-8'))
        elif parsed.path == "/api/v1/telemetry/probes":
            telemetry_list = db_manager.get_recent_telemetry(limit=100)
            self._set_cors()
            self.wfile.write(json.dumps({"count": len(telemetry_list), "probes": telemetry_list}).encode("utf-8"))
        elif parsed.path == "/api/v1/model/latest":
            latest_model = db_manager.get_latest_model_version()
            if not latest_model:
                latest_model = {
                    "version_tag": "v1.0.0-initial",
                    "onnx_path": "web_app/lnn_weather_model.onnx",
                    "avg_loss": 0.05,
                    "verified_sample_count": 100
                }
            self._set_cors()
            self.wfile.write(json.dumps(latest_model).encode("utf-8"))
        elif parsed.path == "/api/v1/lfm-experiments":
            experiments_data = lfm_agent.load_experiments()
            self._set_cors()
            self.wfile.write(json.dumps(experiments_data).encode('utf-8'))
        else:
            # Static File Serving for web_app
            rel_path = parsed.path.lstrip('/')
            if rel_path == '' or rel_path == 'index.html':
                target_file = WORKSPACE_ROOT / "web_app" / "index.html"
                content_type = "text/html; charset=utf-8"
            else:
                target_file = WORKSPACE_ROOT / "web_app" / rel_path
                ext = target_file.suffix.lower()
                content_types = {
                    '.css': 'text/css; charset=utf-8',
                    '.js': 'application/javascript; charset=utf-8',
                    '.json': 'application/json; charset=utf-8',
                    '.onnx': 'application/octet-stream',
                    '.png': 'image/png',
                    '.ico': 'image/x-icon',
                    '.svg': 'image/svg+xml'
                }
                content_type = content_types.get(ext, 'application/octet-stream')

            if target_file.exists() and target_file.is_file():
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(target_file, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Endpoint not found")

def start_server():
    server = HTTPServer(("0.0.0.0", PORT), TelemetryAPIHandler)
    print(f"[SERVER] Live Bataan Weather & LFM-230M Agentic Telemetry API running on port {PORT}...")
    server.serve_forever()

if __name__ == "__main__":
    t = Thread(target=background_poller, daemon=True)
    t.start()
    start_server()

