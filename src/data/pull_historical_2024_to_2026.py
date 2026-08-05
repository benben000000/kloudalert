#!/usr/bin/env python3
"""
Rate-Limit Safe Historical Telemetry Extractor (`src/data/pull_historical_2024_to_2026.py`)

Uses 3 yearly date windows (2024, 2025, 2026) across all 17 weather stations.
Total requests: 51 (fits well under the 300 request / 15 min rate limit quota).
Includes automatic HTTP 429 backoff & RateLimit header parsing.
"""

import sys
import os
import json
import time
import subprocess
import urllib.parse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FILE = WORKSPACE_ROOT / "data" / "raw" / "kloudtrack_full_2024_2026.json"

BASE_URL = "https://api.kloudtechsea.com/api/v1"
API_KEY = "kloud_live_d2c3dece36db0668228537f7846be15a3b0e9303aeeb704d"

STATIONS = [
    {"id": "Rjz2dbXW", "name": "Popolon AWS - Palayan City"},
    {"id": "4VAl2p9k", "name": "Sapang Buho AWS - Palayan City"},
    {"id": "3nzr8bGo", "name": "Alasas AWS - San Fernando City"},
    {"id": "O3z05pGV", "name": "Wawa Limay AWS - Bataan"},
    {"id": "nDbyYbR1", "name": "Sabang Morong AWS - Bataan"},
    {"id": "Bkpj1zRO", "name": "Old Cabalan AWS - Olongapo City"},
    {"id": "rqAkmpKG", "name": "Barretto AWS - Olongapo City"},
    {"id": "wkAWLzlm", "name": "Lazatin AWS - San Fernando City"},
    {"id": "VEpdDpBK", "name": "San Luis AWS - Aurora"},
    {"id": "1Zb102pg", "name": "San Jose City AWS"},
    {"id": "xMbRYxp0", "name": "Avida Asten AWS - Makati City"},
    {"id": "lMAZe9b3", "name": "Abucay AWS - Bataan"},
    {"id": "WYAejdzg", "name": "Poblacion Mariveles AWS - Bataan"},
    {"id": "QgbGldAY", "name": "Pag-asa Bagac AWS - Bataan"},
    {"id": "03pqkGAj", "name": "Bongabon Water District AWS - Nueva Ecija"},
    {"id": "3nzr48bG", "name": "Calumpit AWS - Bulacan"},
    {"id": "nDby4YpR", "name": "General Natividad AWS - Nueva Ecija"}
]

YEAR_WINDOWS = [
    ("2024-01-01T00:00:00.000Z", "2024-12-31T23:59:59.000Z"),
    ("2025-01-01T00:00:00.000Z", "2025-12-31T23:59:59.000Z"),
    ("2026-01-01T00:00:00.000Z", "2026-08-05T23:59:59.000Z")
]

def fetch_url_with_backoff(url, max_retries=4):
    """Executes cURL GET request with automatic 429 rate limit backoff."""
    cmd = [
        "curl.exe",
        "-i",
        "-s",
        "-H", f"x-kloudtrack-key: {API_KEY}",
        "-H", "Accept: application/json",
        "-H", "User-Agent: KloudAlert-RateLimitOptimizer/1.0",
        url
    ]
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
            lat = round((time.time() - t0) * 1000, 1)
            
            output = res.stdout
            parts = output.split("\r\n\r\n", 1) if "\r\n\r\n" in output else output.split("\n\n", 1)
            header_text = parts[0]
            body_text = parts[1] if len(parts) > 1 else ""

            status_code = 200
            first_line = header_text.splitlines()[0] if header_text else ""
            if "HTTP/" in first_line:
                try: status_code = int(first_line.split()[1])
                except Exception: pass

            remaining = None
            reset_sec = 10
            for line in header_text.splitlines():
                l_lower = line.lower()
                if "ratelimit-remaining:" in l_lower:
                    try: remaining = int(line.split(":")[1].strip())
                    except Exception: pass
                elif "ratelimit-reset:" in l_lower:
                    try: reset_sec = int(line.split(":")[1].strip())
                    except Exception: pass

            if status_code == 429:
                sleep_time = max(reset_sec, 2 ** attempt * 4)
                print(f"   ⚠️ [429 RATE LIMIT] Sleeping {sleep_time}s before retrying (Attempt {attempt}/{max_retries})...")
                time.sleep(sleep_time)
                continue

            if status_code in (200, 201):
                data = json.loads(body_text)
                return data, lat, remaining

        except Exception as e:
            print(f"   [CONNECTION NOTE] {e}")
            time.sleep(2)

    return None, 0, None

def run_rate_limit_safe_extraction():
    print("=================================================================")
    print("KLOUDTRACK RATE-LIMIT SAFE HISTORICAL EXTRACTION (2024-2026)")
    print("=================================================================")
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {API_KEY}")
    print(f"Total Requests Planned: {len(STATIONS) * len(YEAR_WINDOWS)} (Rate Limit Quota: 300 / 15m)")

    full_dataset = {
        "metadata": {
            "source": "KloudTrack Production API (Rate-Limit Safe 2024-2026)",
            "base_url": BASE_URL,
            "station_count": len(STATIONS),
            "start_date": "2024-01-01T00:00:00.000Z",
            "end_date": "2026-08-05T23:59:59.000Z",
            "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "stations": {}
    }

    grand_total_records = 0

    for idx, st in enumerate(STATIONS, 1):
        s_id = st["id"]
        s_name = st["name"]
        print(f"\n[{idx:02d}/{len(STATIONS)}] INGESTING HISTORICAL DATA for Station [{s_id}] ({s_name})...")
        
        station_records_map = {}

        for w_idx, (w_start, w_end) in enumerate(YEAR_WINDOWS, 1):
            q_start = urllib.parse.quote(w_start)
            q_end = urllib.parse.quote(w_end)
            hist_url = f"{BASE_URL}/telemetry/station/{s_id}/history?skip=0&take=5000&interval=60&startDate={q_start}&endDate={q_end}"
            
            resp_data, lat, remaining = fetch_url_with_backoff(hist_url)
            
            recs = []
            if resp_data and resp_data.get("success"):
                payload = resp_data.get("data", {})
                if isinstance(payload, dict):
                    recs = payload.get("telemetry", payload.get("data", []))
                elif isinstance(payload, list):
                    recs = payload

            if len(recs) > 0:
                for r in recs:
                    rec_time = r.get("recordedAt")
                    if rec_time and rec_time not in station_records_map:
                        station_records_map[rec_time] = r

                print(f"   Window {w_idx}/{len(YEAR_WINDOWS)} [{w_start[:4]}]: Ingested {len(recs)} records! ({lat}ms) | Quota Remaining: {remaining}")

            # Adaptive Throttle Delay to prevent quota exhaustion
            time.sleep(1.2)

        all_st_recs = list(station_records_map.values())
        all_st_recs.sort(key=lambda x: x.get("recordedAt", ""))

        earliest = all_st_recs[0].get("recordedAt") if all_st_recs else "N/A"
        latest = all_st_recs[-1].get("recordedAt") if all_st_recs else "N/A"

        print(f"   [OK] [STATION COMPLETE] Ingested {len(all_st_recs):,} authentic records!")
        print(f"      • Earliest Deployment Record: {earliest}")
        print(f"      • Latest Record:              {latest}")

        full_dataset["stations"][s_id] = {
            "station_info": st,
            "record_count": len(all_st_recs),
            "earliest_recorded_at": earliest,
            "latest_recorded_at": latest,
            "telemetry": all_st_recs
        }
        grand_total_records += len(all_st_recs)

        # Save incremental dataset
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(full_dataset, f, indent=2)

    print("\n=================================================================")
    print("RATE-LIMIT SAFE EXTRACTION COMPLETE SUMMARY")
    print("=================================================================")
    print(f"Dataset Output Location: {OUTPUT_FILE}")
    print(f"Total Stations Harvested: {len(full_dataset['stations'])}")
    print(f"GRAND TOTAL AUTHENTIC 1-HOUR RECORDS: {grand_total_records:,}")
    print("=================================================================")

if __name__ == "__main__":
    run_rate_limit_safe_extraction()
