#!/usr/bin/env python3
"""
Central Database Manager for KloudAlert / LNN Weather System
Provides thread-safe persistence for:
1. `probe_telemetry`: Mobile APK probe telemetry & user observations
2. `model_experiments`: LNN prediction experiments and ground-truth verification status
3. `model_versions`: Model weights versioning and OTA ONNX metadata
"""

import os
import sys
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = WORKSPACE_ROOT / "data" / "db"
DB_PATH = DB_DIR / "kloudalert_central.db"

class DatabaseManager:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Mobile Probe Telemetry Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS probe_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                barometric_pressure REAL,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                user_reported_condition TEXT,
                prediction_confidence REAL,
                created_at REAL NOT NULL
            )
            """)

            # Index for fast spatio-temporal lookup
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_telemetry_geo_time 
            ON probe_telemetry (timestamp DESC, latitude, longitude)
            """)

            # 2. Model Experiments & Ground-Truth Verification Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_experiments (
                experiment_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                feature_vector TEXT NOT NULL, -- JSON string
                prob_curve TEXT NOT NULL,     -- JSON string
                max_prob REAL NOT NULL,
                status TEXT DEFAULT 'PENDING_VERIFICATION',
                verify_target_ts REAL NOT NULL,
                actual_precip REAL,
                actual_heat_index REAL,
                actual_anomaly INTEGER,
                residual_loss REAL,
                verification_ts REAL
            )
            """)

            # Index for pending verification queue
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_status 
            ON model_experiments (status, verify_target_ts)
            """)

            # 3. Model Versions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_tag TEXT UNIQUE NOT NULL,
                weights_path TEXT NOT NULL,
                onnx_path TEXT NOT NULL,
                avg_loss REAL,
                verified_sample_count INTEGER,
                deployed_at REAL NOT NULL
            )
            """)
            conn.commit()

    # --- Telemetry Ingestion API ---
    def insert_telemetry(self, telemetry_data: Dict[str, Any]) -> int:
        now_ts = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO probe_telemetry (
                device_id, timestamp, latitude, longitude,
                barometric_pressure, temperature, humidity, wind_speed,
                user_reported_condition, prediction_confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                telemetry_data.get("device_id", "anonymous_apk"),
                telemetry_data.get("timestamp", now_ts),
                float(telemetry_data.get("latitude", 0.0)),
                float(telemetry_data.get("longitude", 0.0)),
                telemetry_data.get("barometric_pressure"),
                telemetry_data.get("temperature"),
                telemetry_data.get("humidity"),
                telemetry_data.get("wind_speed"),
                telemetry_data.get("user_reported_condition"),
                telemetry_data.get("prediction_confidence"),
                now_ts
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_telemetry(self, limit: int = 50, lat: Optional[float] = None, lon: Optional[float] = None, radius_km: float = 25.0) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM probe_telemetry ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                results.append(item)
            return results

    # --- Experiments & Active Learning API ---
    def insert_experiment(self, exp_record: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO model_experiments (
                experiment_id, device_id, timestamp, latitude, longitude,
                feature_vector, prob_curve, max_prob, status, verify_target_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_record["id"],
                exp_record.get("device_id", "apk_node"),
                exp_record.get("timestamp", time.time()),
                exp_record["location"]["lat"],
                exp_record["location"]["lon"],
                json.dumps(exp_record.get("feature_vector", [])),
                json.dumps(exp_record.get("prob_curve", [])),
                exp_record.get("max_prob", 0.0),
                exp_record.get("status", "PENDING_VERIFICATION"),
                exp_record.get("verify_target_ts", time.time() + 900)
            ))
            conn.commit()

    def get_pending_experiments(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_experiments WHERE status = 'PENDING_VERIFICATION'")
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["feature_vector"] = json.loads(item["feature_vector"])
                item["prob_curve"] = json.loads(item["prob_curve"])
                results.append(item)
            return results

    def mark_experiment_verified(self, experiment_id: str, actual_precip: float, actual_heat_index: float, actual_anomaly: bool, residual_loss: float):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE model_experiments SET
                status = 'VERIFIED',
                actual_precip = ?,
                actual_heat_index = ?,
                actual_anomaly = ?,
                residual_loss = ?,
                verification_ts = ?
            WHERE experiment_id = ?
            """, (
                actual_precip,
                actual_heat_index,
                1 if actual_anomaly else 0,
                residual_loss,
                time.time(),
                experiment_id
            ))
            conn.commit()

    # --- Model Versioning API ---
    def record_model_version(self, version_tag: str, weights_path: str, onnx_path: str, avg_loss: float, sample_count: int):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO model_versions (
                version_tag, weights_path, onnx_path, avg_loss, verified_sample_count, deployed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                version_tag, str(weights_path), str(onnx_path), avg_loss, sample_count, time.time()
            ))
            conn.commit()

    def get_latest_model_version(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_versions ORDER BY version_id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

if __name__ == "__main__":
    print("[DB MANAGER] Testing Centralized Database Manager...")
    db = DatabaseManager()
    row_id = db.insert_telemetry({
        "device_id": "test_apk_001",
        "latitude": 14.6775,
        "longitude": 120.5431,
        "barometric_pressure": 1008.5,
        "temperature": 32.1,
        "humidity": 78.0,
        "user_reported_condition": "rain_shower"
    })
    print(f"[DB MANAGER] Inserted probe telemetry row ID: {row_id}")
    latest = db.get_recent_telemetry(1)
    print(f"[DB MANAGER] Retrieved telemetry: {latest}")
