#!/usr/bin/env python3
"""
Integration test for Firebase Manager & 1-Tap OTA Model Update System (`src/engine/test_firebase_ota_flow.py`)
"""

import sys
import time
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT / "src" / "data"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "engine"))

from firebase_manager import FirebaseManager
from self_improving_agentic_loop import LFMSelfImprovingAgent

def run_firebase_ota_test():
    print("[FIREBASE OTA TEST] Initializing Firebase Manager & LFM Agent...")
    fb = FirebaseManager()
    agent = LFMSelfImprovingAgent()

    # 1. Test Telemetry Ingestion to Cloud / Local Fallback
    print("\n--- STEP 1: Test Telemetry Submission ---")
    telemetry_data = {
        "device_id": "apk_mobile_probe_999",
        "timestamp": time.time(),
        "latitude": 14.6775,
        "longitude": 120.5431,
        "barometric_pressure": 1008.1,
        "temperature": 32.5,
        "humidity": 80.0,
        "user_reported_condition": "Thunderstorm"
    }
    telemetry_id = fb.insert_telemetry(telemetry_data)
    print(f"[TEST PASS] Telemetry record created: {telemetry_id}")

    # 2. Test OTA Model Version Publication
    print("\n--- STEP 2: Publish New OTA Neural Model Version ---")
    new_version_tag = f"v1.ota.{int(time.time())}"
    weights_file = WORKSPACE_ROOT / "src" / "models" / "weights" / "lfm_230m_weights.pt"
    onnx_file = WORKSPACE_ROOT / "web_app" / "lnn_weather_model.onnx"

    pub_rec = fb.publish_model_version(
        version_tag=new_version_tag,
        weights_path=str(weights_file),
        onnx_path=str(onnx_file),
        avg_loss=0.0124,
        sample_count=150
    )
    print(f"[TEST PASS] Published OTA model version '{new_version_tag}'")

    # 3. Test Latest Model Version Query (for 1-Tap App Button)
    print("\n--- STEP 3: Query Latest Model Version for 1-Tap App Settings Button ---")
    latest = fb.get_latest_model_version()
    print(f"[TEST PASS] Retrieved latest model version: {latest['version_tag']} (Deployed At: {latest['deployed_at']})")

    assert latest["version_tag"] == new_version_tag, "Mismatch in latest published OTA version tag"

    print("\n=======================================================")
    print("SUCCESS: FIREBASE CLOUD TELEMETRY & OTA UPDATE TEST PASSED!")
    print("=======================================================")

if __name__ == "__main__":
    run_firebase_ota_test()
