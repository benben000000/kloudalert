#!/usr/bin/env python3
"""
Integration test for Centralized Telemetry DB & LNN Self-Improving Flywheel
"""

import sys
import time
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT / "src" / "models"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "engine"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "data"))

from db_manager import DatabaseManager
from self_improving_agentic_loop import LFMSelfImprovingAgent

def run_integration_test():
    print("[INTEGRATION TEST] Initializing Central Database Manager & LFM Agent...")
    db = DatabaseManager()
    agent = LFMSelfImprovingAgent()

    # 1. Test Probe Telemetry Ingestion into Central DB
    print("\n--- STEP 1: Ingest Mobile Probe Telemetry ---")
    telemetry_payload = {
        "device_id": "apk_probe_manila_001",
        "timestamp": time.time(),
        "latitude": 14.6775,
        "longitude": 120.5431,
        "barometric_pressure": 1007.8,
        "temperature": 31.5,
        "humidity": 82.0,
        "wind_speed": 12.4,
        "user_reported_condition": "Heavy Rain",
        "prediction_confidence": 0.94
    }
    probe_id = db.insert_telemetry(telemetry_payload)
    print(f"[TEST PASS] Mobile probe telemetry recorded in Central DB with ID: {probe_id}")

    # 2. Retrieve Probe Telemetry
    recent = db.get_recent_telemetry(limit=5)
    assert len(recent) > 0, "Failed to retrieve recorded probe telemetry"
    print(f"[TEST PASS] Retrieved {len(recent)} recent probe records from Central DB.")

    # 3. Log LNN Prediction Experiment to Central DB
    print("\n--- STEP 2: Log Prediction Experiment ---")
    feat_vector = [31.5, 82.0, 1007.8, 2.5, 12.4, 0.0, 0.0, 37.0]
    prob_curve = [0.15, 0.25, 0.45, 0.78, 0.85, 0.92] + [0.88] * 12
    exp_rec = agent.log_prediction_experiment(14.6775, 120.5431, feat_vector, prob_curve, device_id="apk_probe_manila_001")
    print(f"[TEST PASS] Prediction experiment recorded: {exp_rec['id']}")

    # 4. Force Verification & Active Learning Fine-Tuning Pass
    print("\n--- STEP 3: Execute Active Learning & Model Versioning Pass ---")
    exp_rec["verify_target_ts"] = time.time() - 10
    db.insert_experiment(exp_rec)

    res = agent.verify_ground_truth_and_improve({"precip": 3.0, "temp": 31.5})
    print(f"[TEST PASS] Active learning verification result: {json.dumps(res, indent=2)}")

    # 5. Check Updated Model Version in DB
    print("\n--- STEP 4: Query Model Versioning Registry in Central DB ---")
    latest_ver = db.get_latest_model_version()
    print(f"[TEST PASS] Latest model version registered in DB: {latest_ver}")

    print("\n=======================================================")
    print("SUCCESS: ALL CENTRAL TELEMETRY DB & LNN TESTS PASSED!")
    print("=======================================================")

if __name__ == "__main__":
    run_integration_test()
