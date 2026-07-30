#!/usr/bin/env python3
"""
Sequential KloudTech SEA Live Telemetry Extractor
(`src/data/pull_sequential_live_stations.py`)

1. Queries /telemetry/dashboard to discover all registered station HashIDs.
2. Ingests telemetry for 1 station at a time from Jan 1, 2024 to present.
3. If no data exists for 2024-01-01, queries earliest available record to present.
4. Stores separate, distinct authentic telemetry arrays per station in data/raw/kloudtrack_17stations_2024_2026.json.
"""

import sys
import ssl
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = WORKSPACE_ROOT / "data" / "kloudtech_config.json"
DATASET_FILE = WORKSPACE_ROOT / "data" / "raw" / "kloudtrack_17stations_2024_2026.json"

BASE_URL = "https://api.kloudtechsea.com/api/v1"
API_KEY = "kloud_live_d2c3dece36db0668228537f7846be15a3b0e9303aeeb704d"

KNOWN_STATION_HASHIDS = [
    "VEpdDpBK", "nDby4YpR", "3nzr48bG", "03pqkGAj", "QgbGldAY", "WYAejdzg",
    "MA7_SLOB", "LZT_SANF", "SNL_AURR", "SNJ_NUEV", "AVD_MAKT", "ABC_BATN",
    "PBL_MRVL", "PGA_BAGC", "BNG_WTRD", "CLM_BULC", "GNR_NATV"
]

def run_sequential_extraction():
    print("=================================================================")
    print("KLOUDTECH SEA SEQUENTIAL STATION TELEMETRY EXTRACTOR")
    print("=================================================================")
    print(f"Base URL: {BASE_URL}")
    print(f"API Key Header: x-kloudtrack-key: {API_KEY}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "x-kloudtrack-key": API_KEY,
        "Accept": "application/json"
    }

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # Step 1: Discover station IDs via /telemetry/dashboard
    print("\n--- STEP 1: Discovering Registered Station HashIDs via /telemetry/dashboard ---")
    discovered_stations = []
    try:
        url = f"{BASE_URL}/telemetry/dashboard"
        req = urllib.request.Request(url, headers=headers, method="GET")
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            lat = round((time.time() - t0) * 1000, 1)
            if resp.status == 200:
                body = json.loads(resp.read().decode('utf-8'))
                raw_list = body.get("data", body)
                if isinstance(raw_list, list):
                    discovered_stations = raw_list
                    print(f"✅ Dashboard returned {len(discovered_stations)} stations! ({lat}ms)")
    except Exception as e:
        print(f"❌ Dashboard Query Note: {e}")

    # Extract station HashIDs
    station_ids = []
    if discovered_stations:
        for item in discovered_stations:
            s_id = item.get("station", {}).get("id") or item.get("id") or item.get("stationId")
            if s_id and s_id not in station_ids:
                station_ids.append(s_id)

    if not station_ids:
        print("Using Known KloudTrack Station HashIDs directory...")
        station_ids = KNOWN_STATION_HASHIDS

    print(f"\nFinal Registered Station IDs to Ingest ({len(station_ids)} stations):")
    for idx, sid in enumerate(station_ids, 1):
        print(f"  {idx:02d}. [{sid}]")

    # Step 2: Sequential Ingestion (1 Station at a time)
    print("\n--- STEP 2: Sequential Station Telemetry Ingestion (2024 to Present) ---")
    
    start_date = "2024-01-01T00:00:00.000Z"
    end_date = "2026-07-30T23:59:59.000Z"
    
    dataset = {
        "metadata": {
            "source": "KloudTech SEA Authentic Live Dashboard",
            "base_url": BASE_URL,
            "station_count": len(station_ids),
            "ingestion_date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "stations": {}
    }

    for idx, s_id in enumerate(station_ids, 1):
        print(f"\n[{idx:02d}/{len(station_ids)}] INGESTING STATION [{s_id}] (1 Station at a time)...")
        readings = []

        # Attempt 1: Query with 2024-01-01 to 2026-07-30 window
        query1 = f"skip=0&take=5000&interval=60&startDate={urllib.parse.quote(start_date)}&endDate={urllib.parse.quote(end_date)}"
        url1 = f"{BASE_URL}/telemetry/station/{s_id}/history?{query1}"
        
        try:
            print(f"   Querying Primary Window: GET {url1}")
            req = urllib.request.Request(url1, headers=headers, method="GET")
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=12, context=ctx) as resp:
                lat = round((time.time() - t0) * 1000, 1)
                if resp.status == 200:
                    body = json.loads(resp.read().decode('utf-8'))
                    records = body.get("data", body)
                    if isinstance(records, list) and len(records) > 0:
                        readings = records
                        print(f"   ✅ [200 OK] Fetched {len(readings)} authentic records for [{s_id}] ({lat}ms)!")
        except Exception as e:
            print(f"   Note for Primary Window [{s_id}]: {e}")

        # Attempt 2: If no data returned for 2024-01-01, query without startDate (earliest available to present)
        if not readings:
            query2 = f"skip=0&take=5000&interval=60"
            url2 = f"{BASE_URL}/telemetry/station/{s_id}/history?{query2}"
            try:
                print(f"   Querying Earliest-Available Window: GET {url2}")
                req = urllib.request.Request(url2, headers=headers, method="GET")
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=12, context=ctx) as resp:
                    lat = round((time.time() - t0) * 1000, 1)
                    if resp.status == 200:
                        body = json.loads(resp.read().decode('utf-8'))
                        records = body.get("data", body)
                        if isinstance(records, list) and len(records) > 0:
                            readings = records
                            print(f"   ✅ [200 OK Earliest Data] Fetched {len(readings)} records for [{s_id}] ({lat}ms)!")
            except Exception as e:
                print(f"   Note for Earliest Window [{s_id}]: {e}")

        dataset["stations"][s_id] = readings
        
        # Save dataset incrementally so each station has its own separate array
        DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_FILE, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        print(f"   [SAVED] Updated dataset file with Station [{s_id}]")

        time.sleep(0.5)

    print("\n=================================================================")
    print("SEQUENTIAL EXTRACTION COMPLETE SUMMARY")
    print("=================================================================")
    for sid, recs in dataset["stations"].items():
        print(f"  - Station [{sid}]: {len(recs)} separate authentic records")
    print(f"Saved dataset file: {DATASET_FILE}")
    print("=================================================================")

if __name__ == "__main__":
    run_sequential_extraction()
