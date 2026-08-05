#!/usr/bin/env python3
"""
Firebase Realtime Presence & Telemetry Synchronization Pipeline
(`src/engine/firebase_telemetry_sync.py`)

Architectural Pipeline:
1. Presence Tracking: Registers active app user coordinates in Firebase Realtime Database.
2. Ground-Truth Telemetry Ingestion: Writes user rain observations to Firestore `rain_telemetry_events`.
3. Firebase Remote Config Model Distribution: OTA push of tiny ONNX model (671 KB) to mobile devices.
"""

import sys
import os
import json
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

class FirebaseTelemetrySyncManager:
    """Simulates Firebase Firestore & Realtime Database Sync Pipeline for PIMCAN-v4."""
    def __init__(self):
        self.active_presence_table = {}
        self.firestore_events_collection = []

    def register_user_presence(self, user_id, lat, lon, device_os="Android"):
        """Registers active user presence (Firebase Realtime Database / Firestore)."""
        record = {
          "user_id": user_id,
          "gps": {"lat": lat, "lon": lon},
          "device_os": device_os,
          "status": "ONLINE_ACTIVE",
          "last_ping": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.active_presence_table[user_id] = record
        print(f"[Firebase Presence] Registered Active User [{user_id}] at ({lat:.4f}, {lon:.4f}) | Device: {device_os}")
        return record

    def post_ground_truth_telemetry(self, user_id, lat, lon, is_raining, intensity="MODERATE"):
        """Posts user rain observation to Firebase Firestore collection 'rain_telemetry_events'."""
        doc = {
            "doc_id": f"doc_{int(time.time())}_{user_id}",
            "user_id": user_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "geo_point": {"lat": lat, "lon": lon},
            "user_observation": "RAINING" if is_raining else "DRY",
            "intensity": intensity,
            "synced_to_cloud": True
        }
        self.firestore_events_collection.append(doc)
        print(f"[Firebase Firestore] Added Ground-Truth Event Document [{doc['doc_id']}] -> Rain: {is_raining}")
        return doc

    def check_remote_config_model_update(self, current_client_version="v4.1.0"):
        """Firebase Remote Config OTA Model Version Check."""
        latest_cloud_version = "v4.2.0"
        model_download_url = "https://storage.googleapis.com/kloudalert-models/lnn_weather_model_v4.2.0.onnx"
        
        needs_update = current_client_version != latest_cloud_version
        print(f"[Firebase Remote Config] Client Version: {current_client_version} | Latest Cloud Model: {latest_cloud_version}")
        if needs_update:
            print(f"  -> OTA Model Update Available! Downloading lightweight ONNX ({model_download_url})...")
        return {
            "update_available": needs_update,
            "latest_version": latest_cloud_version,
            "onnx_url": model_download_url
        }

if __name__ == "__main__":
    fb = FirebaseTelemetrySyncManager()
    fb.register_user_presence("usr_wawa_01", 14.5621, 120.5934, device_os="Android PWA")
    fb.post_ground_truth_telemetry("usr_wawa_01", 14.5621, 120.5934, is_raining=True, intensity="HEAVY")
    fb.check_remote_config_model_update(current_client_version="v4.1.0")
