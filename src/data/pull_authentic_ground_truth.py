#!/usr/bin/env python3
"""
Clean Authentic Telemetry Extractor (`src/data/pull_authentic_ground_truth.py`)
Queries https://api.kloudtechsea.com/api/v1 via Windows curl.exe Schannel TLS
and saves ONLY real ground-truth telemetry records into data/raw/kloudtrack_authentic_2024_2026.json.
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FILE = WORKSPACE_ROOT / "data" / "raw" / "kloudtrack_authentic_2024_2026.json"
OLD_FILE = WORKSPACE_ROOT / "data" / "raw" / "kloudtrack_17stations_2024_2026.json"

BASE_URL = "https://api.kloudtechsea.com/api/v1"
API_KEY = "kloud_live_d2c3dece36db0668228537f7846be15a3b0e9303aeeb704d"

def fetch_url_via_curl(url):
    """Executes live HTTP request using Windows curl.exe Schannel engine."""
    cmd = [
        "curl.exe",
        "-s",
        "-H", f"x-kloudtrack-key: {API_KEY}",
        "-H", "Accept: application/json",
        "-H", "User-Agent: Mozilla/5.0",
        url
    ]
    try:
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True)
        lat = round((time.time() - t0) * 1000, 1)
        data = json.loads(res.stdout)
        return data, lat
    except Exception as e:
        print(f"   [cURL ERROR] {e}")
        return None, 0

def run_extraction():
    print("=================================================================")
    print("KLOUDTRACK AUTHENTIC GROUND-TRUTH TELEMETRY INGESTION")
    print("=================================================================")
    
    # Force delete any old placeholder files
    if OLD_FILE.exists():
        OLD_FILE.unlink()
        print(f"[CLEANUP] Deleted old dataset file: {OLD_FILE}")
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print(f"[CLEANUP] Reset target output file: {OUTPUT_FILE}")

    # Discover Stations
    dash_url = f"{BASE_URL}/telemetry/dashboard"
    dash_data, lat = fetch_url_via_curl(dash_url)
    
    stations = []
    if dash_data and dash_data.get("success"):
        raw_list = dash_data.get("data", [])
        print(f"[OK] [200 OK SUCCESS] Discovered {len(raw_list)} live weather stations ({lat}ms)")
        for item in raw_list:
            st = item.get("station", item) if isinstance(item, dict) else item
            s_id = st.get("id") or st.get("stationId")
            s_name = st.get("stationName") or st.get("name")
            if s_id:
                stations.append({"id": s_id, "name": s_name})

    print(f"\nIngesting Authentic Telemetry for {len(stations)} Weather Stations:")

    dataset = {
        "metadata": {
            "source": "KloudTrack Official Live Production API",
            "base_url": BASE_URL,
            "station_count": len(stations),
            "ingestion_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "stations": {}
    }

    for idx, st in enumerate(stations, 1):
        s_id = st["id"]
        s_name = st["name"]
        print(f"\n[{idx:02d}/{len(stations)}] Querying Station [{s_id}] ({s_name})...")
        
        hist_url = f"{BASE_URL}/telemetry/station/{s_id}/history?skip=0&take=5000&interval=60"
        hist_data, lat = fetch_url_via_curl(hist_url)
        
        records = []
        if hist_data and hist_data.get("success"):
            payload = hist_data.get("data", {})
            if isinstance(payload, dict):
                records = payload.get("telemetry", payload.get("data", []))
            elif isinstance(payload, list):
                records = payload

        print(f"   [OK] Ingested {len(records)} authentic ground-truth records! Latency: {lat}ms")
        
        if len(records) > 0:
            sample = records[0]
            print(f"   Sample Record [0]: recordedAt='{sample.get('recordedAt')}', temp={sample.get('temperature')}°C, humidity={sample.get('humidity')}%, pressure={sample.get('pressure')}hPa")

        dataset["stations"][s_id] = {
            "station_info": st,
            "telemetry": records
        }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print("\n=================================================================")
    print("AUTHENTIC DATASET SAVED SUCCESSFULLY")
    print("=================================================================")
    print(f"File Path: {OUTPUT_FILE}")
    print(f"Total Stations Saved: {len(dataset['stations'])}")
    total_recs = sum(len(v['telemetry']) for v in dataset['stations'].values())
    print(f"Total Authentic Telemetry Records: {total_recs}")
    print("=================================================================")

if __name__ == "__main__":
    run_extraction()
