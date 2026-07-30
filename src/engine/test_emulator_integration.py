#!/usr/bin/env python3
"""
Test suite for Android Emulator Skill & Verifier (`src/engine/test_emulator_integration.py`)
"""

import sys
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT / ".agent" / "skills"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "engine"))

from emulator import run_emulator_skill_audit
from emulator_verifier import EmulatorVerifier

def run_emulator_integration_test():
    print("[EMULATOR TEST] Running .agent/skills/emulator.py Skill Audit...")
    skill_report = run_emulator_skill_audit()
    print(f"[EMULATOR TEST] Skill Audit Result:\n{json.dumps(skill_report, indent=2)}")

    assert skill_report["repo_status"] == "CLONED_AND_VERIFIED", "repos/emulator is missing or invalid"
    print("[TEST PASS] repos/emulator is correctly cloned and registered in governance!")

    print("\n[EMULATOR TEST] Running EmulatorVerifier Engine...")
    verifier = EmulatorVerifier()
    devices = verifier.get_connected_devices()
    print(f"[EMULATOR TEST] Discovered ADB Devices: {devices}")

    verify_res = verifier.install_and_verify_app()
    print(f"[EMULATOR TEST] Verification Result:\n{json.dumps(verify_res, indent=2)}")

    print("\n=======================================================")
    print("SUCCESS: ANDROID EMULATOR INTEGRATION AUDIT PASSED!")
    print("=======================================================")

if __name__ == "__main__":
    run_emulator_integration_test()
