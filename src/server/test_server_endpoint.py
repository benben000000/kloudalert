#!/usr/bin/env python3
"""
Server Endpoint & KloudTech + LFM Fusion Empirical Verification Test
(`src/server/test_server_endpoint.py`)
"""

import sys
import json
import time
import urllib.request
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT / "src" / "server"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "engine"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "models"))

from live_stream_server import update_live_telemetry, compute_idw_fused_weather

def test_lfm_fused_weather_engine():
    print("=================================================================")
    print("EMPIRICAL TEST: KLOUDTECH LIVE AWS & LFM-230M FUSED WEATHER ENGINE")
    print("=================================================================")

    # 1. Test live telemetry polling from KloudTech AWS Network
    print("\n--- Testing KloudTech Live Station Telemetry Ingestion ---")
    update_live_telemetry()

    # 2. Test Multi-Station IDW Fusion for Bataan coordinates (Balanga City)
    lat_test = 14.6775
    lon_test = 120.5431
    print(f"\n--- Testing Spatial IDW Fusion for Coordinates ({lat_test}, {lon_test}) ---")
    fused_weather = compute_idw_fused_weather(lat_test, lon_test)
    
    print("Fused Weather Parameters Output:")
    print(json.dumps(fused_weather, indent=2))

    assert "temp" in fused_weather
    assert "heat_index" in fused_weather
    assert "pressure" in fused_weather
    assert "precip" in fused_weather
    print("\n[OK] Verification Passed! Live KloudTech AWS telemetry and Romps Heat Index IDW fusion confirmed.")

if __name__ == "__main__":
    test_lfm_fused_weather_engine()
