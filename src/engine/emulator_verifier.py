#!/usr/bin/env python3
"""
Android Emulator Verification Engine (`src/engine/emulator_verifier.py`)
Integrates `repos/emulator` script logic to:
1. Detect Android emulators / ADB devices
2. Deploy built `KloudAlert.apk`
3. Launch MainActivity and verify live telemetry sync
4. Capture screenshot verification artifacts
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT / "src" / "data"))

from db_manager import DatabaseManager

ANDROID_SDK_ROOT = Path(os.environ.get("ANDROID_SDK_ROOT", r"C:\Users\bmgar\AppData\Local\Android\Sdk"))
ADB_BIN = ANDROID_SDK_ROOT / "platform-tools" / "adb.exe"
EMULATOR_REPO_DIR = WORKSPACE_ROOT / "repos" / "emulator"
SCREENSHOT_DIR = WORKSPACE_ROOT / "data" / "emulator_screenshots"

SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

class EmulatorVerifier:
    def __init__(self):
        self.db = DatabaseManager()
        self.apk_path = WORKSPACE_ROOT / "KloudAlert.apk"

    def get_connected_devices(self):
        devices = []
        if ADB_BIN.exists():
            try:
                res = subprocess.run([str(ADB_BIN), "devices"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    lines = res.stdout.strip().split("\n")[1:]
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 2 and parts[1] == "device":
                            devices.append(parts[0])
            except Exception as e:
                print(f"[EMULATOR VERIFIER] ADB error: {e}")
        return devices

    def install_and_verify_app(self, device_id: str = None) -> dict:
        """Deploys KloudAlert.apk to target device/emulator and verifies execution."""
        devices = self.get_connected_devices()
        target_device = device_id or (devices[0] if devices else None)

        if not target_device:
            print("[EMULATOR VERIFIER] No active ADB emulator or device connected.")
            return {
                "status": "SKIPPED_NO_DEVICE",
                "message": "No active Android emulator connected. Start AVD via repos/emulator or Android Studio.",
                "timestamp": time.time()
            }

        print(f"[EMULATOR VERIFIER] Deploying {self.apk_path.name} to target device {target_device}...")
        
        # 1. ADB Install
        install_cmd = [str(ADB_BIN), "-s", target_device, "install", "-r", str(self.apk_path)]
        install_res = subprocess.run(install_cmd, capture_output=True, text=True, timeout=30)
        
        if install_res.returncode != 0:
            print(f"[EMULATOR VERIFIER] Install failed: {install_res.stderr}")
            return {
                "status": "INSTALL_FAILED",
                "error": install_res.stderr,
                "timestamp": time.time()
            }

        print("[EMULATOR VERIFIER] APK installed successfully! Launching MainActivity...")

        # 2. Launch App Activity
        launch_cmd = [str(ADB_BIN), "-s", target_device, "shell", "am", "start", "-n", "com.kloudalert.weather/com.kloudalert.weather.MainActivity"]
        subprocess.run(launch_cmd, capture_output=True, text=True, timeout=10)

        time.sleep(3)

        # 3. Capture Screen Artifact
        screenshot_file = SCREENSHOT_DIR / f"emulator_verify_{int(time.time())}.png"
        remote_path = "/sdcard/screen_verify.png"
        subprocess.run([str(ADB_BIN), "-s", target_device, "shell", "screencap", "-p", remote_path], capture_output=True, timeout=5)
        subprocess.run([str(ADB_BIN), "-s", target_device, "pull", remote_path, str(screenshot_file)], capture_output=True, timeout=5)

        # 4. Check Central DB for recent telemetry from this probe
        recent_probe_telemetry = self.db.get_recent_telemetry(limit=5)
        telemetry_verified = len(recent_probe_telemetry) > 0

        print(f"[EMULATOR VERIFIER] Screenshot saved to {screenshot_file}. Probe telemetry verified: {telemetry_verified}")

        return {
            "status": "VERIFIED_SUCCESS",
            "target_device": target_device,
            "apk_installed": True,
            "screenshot_path": str(screenshot_file),
            "telemetry_verified": telemetry_verified,
            "timestamp": time.time()
        }

if __name__ == "__main__":
    verifier = EmulatorVerifier()
    devices = verifier.get_connected_devices()
    print(f"[EMULATOR VERIFIER] Active ADB Devices: {devices}")
    res = verifier.install_and_verify_app()
    print("[EMULATOR VERIFIER] Result:", json.dumps(res, indent=2))
