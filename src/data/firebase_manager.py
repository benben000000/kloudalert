#!/usr/bin/env python3
"""
Firebase Cloud Firestore Manager for KloudAlert
Provides cloud synchronization for:
1. `probe_telemetry`: Crowd-sourced mobile telemetry collected across all user devices
2. `model_versions`: Published OTA ONNX model weights and version tags
3. Automatic fallback to local `DatabaseManager` if Firebase Admin credentials are not initialized
"""

import os
import sys
import json
import time
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT / "src" / "data"))

from db_manager import DatabaseManager

SERVICE_ACCOUNT_FILE = WORKSPACE_ROOT / "data" / "firebase_service_account.json"

class FirebaseManager:
    def __init__(self, service_account_path: Optional[Path] = None):
        self.local_db = DatabaseManager()
        self.firestore_db = None
        self.use_firebase = False

        sa_path = service_account_path or SERVICE_ACCOUNT_FILE
        if sa_path.exists():
            try:
                import firebase_admin
                from firebase_admin import credentials, firestore
                if not firebase_admin._apps:
                    cred = credentials.Certificate(str(sa_path))
                    firebase_admin.initialize_app(cred)
                self.firestore_db = firestore.client()
                self.use_firebase = True
                print(f"[FIREBASE MANAGER] Successfully initialized Cloud Firestore from {sa_path.name}")
            except Exception as e:
                print(f"[FIREBASE MANAGER] Firebase Admin init note (using local DB fallback): {e}")

    def insert_telemetry(self, telemetry_data: Dict[str, Any]) -> str:
        """Inserts mobile probe telemetry to Firestore (or local DB fallback)."""
        now_ts = time.time()
        telemetry_data["timestamp"] = telemetry_data.get("timestamp", now_ts)
        telemetry_data["created_at"] = now_ts

        if self.use_firebase and self.firestore_db:
            try:
                doc_ref = self.firestore_db.collection("probe_telemetry").document()
                doc_ref.set(telemetry_data)
                print(f"[FIREBASE MANAGER] Inserted telemetry to Firestore collection 'probe_telemetry' ID: {doc_ref.id}")
                return doc_ref.id
            except Exception as e:
                print(f"[FIREBASE MANAGER] Firestore insert error (falling back to local DB): {e}")

        # Fallback to local SQLite database
        local_id = self.local_db.insert_telemetry(telemetry_data)
        return f"local_{local_id}"

    def publish_model_version(self, version_tag: str, weights_path: str, onnx_path: str, avg_loss: float, sample_count: int) -> Dict[str, Any]:
        """Publishes a new OTA model version and base64 encoded ONNX binary to Firestore & local DB."""
        now_ts = time.time()
        onnx_file = Path(onnx_path)
        onnx_base64 = ""
        if onnx_file.exists():
            try:
                with open(onnx_file, "rb") as f:
                    onnx_base64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                print(f"[FIREBASE MANAGER] Failed to encode ONNX binary: {e}")

        version_data = {
            "version_tag": version_tag,
            "weights_path": str(weights_path),
            "onnx_path": str(onnx_path),
            "onnx_base64": onnx_base64,
            "avg_loss": round(avg_loss, 4),
            "verified_sample_count": sample_count,
            "deployed_at": now_ts
        }

        if self.use_firebase and self.firestore_db:
            try:
                self.firestore_db.collection("model_versions").document(version_tag).set(version_data)
                print(f"[FIREBASE MANAGER] Published OTA model version '{version_tag}' to Firestore!")
            except Exception as e:
                print(f"[FIREBASE MANAGER] Firestore publish error: {e}")

        self.local_db.record_model_version(version_tag, weights_path, onnx_path, avg_loss, sample_count)
        return version_data

    def get_latest_model_version(self) -> Optional[Dict[str, Any]]:
        """Queries the latest published model version from Firestore or local DB."""
        if self.use_firebase and self.firestore_db:
            try:
                query = self.firestore_db.collection("model_versions").order_by("deployed_at", direction="DESCENDING").limit(1)
                docs = list(query.stream())
                if docs:
                    return docs[0].to_dict()
            except Exception as e:
                print(f"[FIREBASE MANAGER] Firestore query note: {e}")

        return self.local_db.get_latest_model_version()

    def get_recent_telemetry(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent crowdsourced probe telemetry points."""
        if self.use_firebase and self.firestore_db:
            try:
                query = self.firestore_db.collection("probe_telemetry").order_by("timestamp", direction="DESCENDING").limit(limit)
                docs = list(query.stream())
                return [d.to_dict() for d in docs]
            except Exception as e:
                print(f"[FIREBASE MANAGER] Firestore query note: {e}")

        return self.local_db.get_recent_telemetry(limit)

if __name__ == "__main__":
    print("[FIREBASE MANAGER] Testing Firebase Manager Module...")
    fb = FirebaseManager()
    rec_id = fb.insert_telemetry({
        "device_id": "test_device_001",
        "latitude": 14.6775,
        "longitude": 120.5431,
        "barometric_pressure": 1009.2,
        "temperature": 30.5,
        "humidity": 76.0
    })
    print(f"[FIREBASE MANAGER] Telemetry inserted ID: {rec_id}")
    latest = fb.get_latest_model_version()
    print(f"[FIREBASE MANAGER] Latest model version: {latest}")
