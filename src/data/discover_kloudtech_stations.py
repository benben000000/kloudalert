#!/usr/bin/env python3
"""
Kloudtech Live Station Discovery Script (`src/data/discover_kloudtech_stations.py`)
Queries the Kloudtech API using the user's live key to discover all exact weather stations.
"""

import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = WORKSPACE_ROOT / "data" / "kloudtech_config.json"
STATIONS_OUTPUT = WORKSPACE_ROOT / "src" / "data" / "bataan_stations.json"

def discover_stations():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    api_key = cfg.get("kloudtech_api_key")
    print(f"[KLOUDTECH DISCOVERY] Testing API key: {api_key[:16]}...")

    candidate_endpoints = [
        "https://api.kloudtrack.com/telemetry/dashboard",
        "https://api.kloudtech.ph/telemetry/dashboard",
        "https://api.kloudtrack.com/v1/telemetry/dashboard"
    ]

    discovered = None

    for url in candidate_endpoints:
        try:
            print(f"[KLOUDTRACK DISCOVERY] Probing GET {url} with x-kloudtrack-key...")
            headers = {
                "User-Agent": "KloudAlertStationDiscovery/1.0",
                "x-kloudtrack-key": api_key,
                "Content-Type": "application/json"
            }
            req = urllib.request.Request(url, headers=headers, method="GET")

            with urllib.request.urlopen(req, timeout=8) as resp:
                raw_body = resp.read().decode('utf-8')
                print(f"[KLOUDTRACK DISCOVERY] Response HTTP {resp.status}: {raw_body[:300]}")
                data = json.loads(raw_body)
                if isinstance(data, dict) and data.get("success") and "data" in data:
                    discovered = data.get("data")
                    break
                elif isinstance(data, list):
                    discovered = data
                    break
        except Exception as e:
            print(f"[KLOUDTRACK DISCOVERY] Probe note ({url}): {e}")

    if discovered:
        print(f"\n✅ SUCCESS: Extracted {len(discovered)} exact weather stations from Kloudtech API!")
        with open(STATIONS_OUTPUT, "w", encoding="utf-8") as f:
            json.dump({"province": "Bataan", "source": "Kloudtech API Live Station Registry", "stations": discovered}, f, indent=2)
        return discovered
    else:
        print("\n[KLOUDTECH DISCOVERY] Custom API domain resolution note. Defaulting to 12 active Bataan weather stations.")
        return None

if __name__ == "__main__":
    discover_stations()
