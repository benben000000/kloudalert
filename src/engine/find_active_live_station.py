#!/usr/bin/env python3
"""
Inspect all 17 weather stations to find active live sensors with real-time changing readings.
(`src/engine/find_active_live_station.py`)
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path

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

def check_stations():
    print("=================================================================")
    print("INSPECTING LIVE TELEMETRY FOR ALL 17 WEATHER STATIONS")
    print("=================================================================")
    
    for st in STATIONS:
        s_id = st["id"]
        s_name = st["name"]
        url = f"{BASE_URL}/telemetry/station/{s_id}/history?skip=0&take=2"
        cmd = [
            "curl.exe", "-s",
            "-H", f"x-kloudtrack-key: {API_KEY}",
            "-H", "Accept: application/json",
            url
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                if data.get("success"):
                    payload = data.get("data", {})
                    recs = payload.get("telemetry", payload.get("data", [])) if isinstance(payload, dict) else payload
                    if isinstance(recs, list) and len(recs) > 0:
                        r0 = recs[0]
                        print(f"  • Station [{s_id}] ({s_name}):")
                        print(f"    - RecordedAt: {r0.get('recordedAt')}")
                        print(f"    - Temp: {r0.get('temperature') or r0.get('temp')} | Hum: {r0.get('humidity') or r0.get('hum')} | Press: {r0.get('pressure')} | Precip: {r0.get('precipitation')}")
                        print(f"    - Keys: {list(r0.keys())}")
                        print("  ---------------------------------------------------------------")
        except Exception as e:
            print(f"  ❌ Error probing [{s_id}]: {e}")

if __name__ == "__main__":
    check_stations()
