#!/usr/bin/env python3
"""
Kloudtech Ground Station Fetcher (`src/data/fetch_kloudtech_stations.py`)
Retrieves all registered Kloudtech ground weather stations using the API key.
"""

import sys
import json
import urllib.request
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = WORKSPACE_ROOT / "data" / "kloudtech_config.json"
STATIONS_OUTPUT = WORKSPACE_ROOT / "src" / "data" / "bataan_stations.json"

def fetch_kloudtech_stations():
    if not CONFIG_PATH.exists():
        print(f"[KLOUDTECH] Config file not found at {CONFIG_PATH}")
        return None

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    api_key = cfg.get("kloudtech_api_key")
    print(f"[KLOUDTECH] Querying Ground Station Registry using API Key: {api_key[:12]}...")

    # Attempt to fetch live station list from Kloudtech API endpoint
    endpoints_to_try = [
        f"https://api.kloudtech.ph/v1/ground-station/list?key={api_key}",
        f"https://api.kloudtech.ph/v1/stations?key={api_key}"
    ]

    live_stations = None
    for url in endpoints_to_try:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "KloudAlertStationFetcher/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode('utf-8'))
                    live_stations = data.get("stations", data)
                    print(f"[KLOUDTECH] Successfully fetched live stations from {url}!")
                    break
        except Exception as e:
            print(f"[KLOUDTECH] Query endpoint {url} note: {e}")

    # Load local 12 Bataan AWS station registry
    with open(STATIONS_OUTPUT, "r", encoding="utf-8") as f:
        local_registry = json.load(f)

    if live_stations and isinstance(live_stations, list):
        print(f"[KLOUDTECH] Received {len(live_stations)} live stations from Kloudtech API!")
        local_registry["kloudtech_live_stations"] = live_stations
        with open(STATIONS_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(local_registry, f, indent=2)
    else:
        print("[KLOUDTECH] Configured 12 Bataan AWS Ground Stations in registry:")
        for idx, st in enumerate(local_registry.get("stations", []), 1):
            print(f"  {idx:02d}. [{st['id']}] {st['name']} - Lat: {st['lat']}, Lon: {st['lon']} (Elevation: {st['elevation_m']}m)")

    return local_registry

if __name__ == "__main__":
    fetch_kloudtech_stations()
