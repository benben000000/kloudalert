#!/usr/bin/env python3
"""
Paginated Deep Historical Telemetry Extractor (`src/data/pull_all_historical_telemetry_paginated.py`)

Loops through skip=0, 5000, 10000, 15000... with interval=60 for each station to fetch
EVERY SINGLE 1-hour telemetry reading from the station's earliest recorded date up to present!
Saves authentic records into data/raw/kloudtrack_deep_history_2024_2026.json.
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FILE = WORKSPACE_ROOT / "data" / "raw" / "kloudtrack_deep_history_2024_2026.json"

BASE_URL = "https://api.kloudtechsea.com/api/v1"
API_KEY = "kloud_live_d2c3dece36db0668228537f7846be15a3b0e9303aeeb704d"

def fetch_url_via_curl(url):
    """Executes HTTP request using Windows cURL Schannel engine."""
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
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=25, check=True)
        lat = round((time.time() - t0) * 1000, 1)
        data = json.loads(res.stdout)
        return data, lat
    except Exception as e:
        print(f"   [cURL ERROR] {e}")
        return None, 0

def run_deep_historical_extraction():
    print("=================================================================")
    print("KLOUDTRACK PAGINATED DEEP HISTORICAL TELEMETRY EXTRACTION")
    print("=================================================================")
    print(f"Target Base URL: {BASE_URL}")
    print(f"API Key: {API_KEY}")

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

    dataset = {
        "metadata": {
            "source": "KloudTrack Production API (Deep Paginated History)",
            "base_url": BASE_URL,
            "station_count": len(stations),
            "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "stations": {}
    }

    total_all_records = 0

    for idx, st in enumerate(stations, 1):
        s_id = st["id"]
        s_name = st["name"]
        print(f"\n[{idx:02d}/{len(stations)}] PAGINATING ALL HISTORICAL DATA FOR Station [{s_id}] ({s_name})...")
        
        station_all_telemetry = []
        skip = 0
        take = 5000

        while True:
            hist_url = f"{BASE_URL}/telemetry/station/{s_id}/history?skip={skip}&take={take}&interval=60"
            print(f"   Querying Page (skip={skip}, take={take})...")
            hist_data, lat = fetch_url_via_curl(hist_url)
            
            page_records = []
            if hist_data and hist_data.get("success"):
                payload = hist_data.get("data", {})
                if isinstance(payload, dict):
                    page_records = payload.get("telemetry", payload.get("data", []))
                elif isinstance(payload, list):
                    page_records = payload

            if not page_records or len(page_records) == 0:
                print(f"   [PAGE END] No more records at skip={skip}.")
                break

            print(f"   [OK] Fetched {len(page_records)} hourly records (Latency: {lat}ms)")
            station_all_telemetry.extend(page_records)
            
            # If page returned fewer than `take` records, we reached the earliest record
            if len(page_records) < take:
                break
                
            skip += take
            time.sleep(0.3)

        # Sort telemetry chronologically (earliest to latest)
        station_all_telemetry.sort(key=lambda x: x.get("recordedAt", ""))
        
        earliest_time = station_all_telemetry[0].get("recordedAt") if station_all_telemetry else "N/A"
        latest_time = station_all_telemetry[-1].get("recordedAt") if station_all_telemetry else "N/A"

        print(f"   [OK] [STATION COMPLETE] Ingested {len(station_all_telemetry):,} total 1-hour records!")
        print(f"      • Earliest Record: {earliest_time}")
        print(f"      • Latest Record:   {latest_time}")

        dataset["stations"][s_id] = {
            "station_info": st,
            "record_count": len(station_all_telemetry),
            "earliest_recorded_at": earliest_time,
            "latest_recorded_at": latest_time,
            "telemetry": station_all_telemetry
        }
        total_all_records += len(station_all_telemetry)

        # Save incremental progress
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

    print("\n=================================================================")
    print("DEEP HISTORICAL EXTRACTION COMPLETE SUMMARY")
    print("=================================================================")
    print(f"Target Output File: {OUTPUT_FILE}")
    print(f"Total Weather Stations Ingested: {len(dataset['stations'])}")
    print(f"GRAND TOTAL AUTHENTIC 1-HOUR RECORDS: {total_all_records:,}")
    print("=================================================================")

if __name__ == "__main__":
    run_deep_historical_extraction()
