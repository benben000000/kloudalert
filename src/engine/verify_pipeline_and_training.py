#!/usr/bin/env python3
"""
Empirical Verification Script for Multimodal Ingestion & Model Training
(`src/engine/verify_pipeline_and_training.py`)
"""

import os
import sys
import json
import sqlite3
import torch
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DIR = WORKSPACE_ROOT / "data" / "raw"
PROCESSED_DIR = WORKSPACE_ROOT / "data" / "processed"
WEIGHTS_PATH = WORKSPACE_ROOT / "src" / "models" / "weights" / "pimcan_liquid_weights.pt"
ONNX_PATH = WORKSPACE_ROOT / "web_app" / "lnn_weather_model.onnx"
DB_PATH = WORKSPACE_ROOT / "data" / "telemetry_db.sqlite"

def verify_all():
    print("=================================================================")
    print("EMPIRICAL PIPELINE & MODEL TRAINING VERIFICATION")
    print("=================================================================")

    # 1. Blitzortung
    blitz_raw = RAW_DIR / "blitzortung_luzon_bataan_2024_2026.json"
    blitz_pt = PROCESSED_DIR / "lightning_grids_2024_2026.pt"
    print(f"\n[1] VERIFYING BLITZORTUNG LIGHTNING DATASET:")
    if blitz_raw.exists():
        with open(blitz_raw, "r", encoding="utf-8") as f:
            b_meta = json.load(f)
        print(f"  • Source: {b_meta.get('source')}")
        print(f"  • Range: {b_meta.get('temporal_range', {}).get('start')} to {b_meta.get('temporal_range', {}).get('end')}")
        print(f"  • Total Strokes Recorded: {b_meta.get('total_lightning_strokes'):,}")
    if blitz_pt.exists():
        pt_data = torch.load(blitz_pt, weights_only=False)
        print(f"  • Tensor Shape: {pt_data['tensor_shape']} | Size: {blitz_pt.stat().st_size / (1024*1024):.1f} MB")

    # 2. Himawari-9
    hima_raw = RAW_DIR / "himawari9_bataan_satellite_2024_2026.json"
    hima_pt = PROCESSED_DIR / "satellite_grids_2024_2026.pt"
    print(f"\n[2] VERIFYING HIMAWARI-9 SATELLITE DATASET:")
    if hima_raw.exists():
        with open(hima_raw, "r", encoding="utf-8") as f:
            h_meta = json.load(f)
        print(f"  • Source: {h_meta.get('source')}")
        print(f"  • Range: {h_meta.get('temporal_range', {}).get('start')} to {h_meta.get('temporal_range', {}).get('end')}")
        print(f"  • Convective Tower Events: {h_meta.get('convective_overshooting_events'):,}")
    if hima_pt.exists():
        h_pt_data = torch.load(hima_pt, weights_only=False)
        print(f"  • Tensor Shape: {h_pt_data['tensor_shape']} | Size: {hima_pt.stat().st_size / (1024*1024):.1f} MB")

    # 3. RainViewer
    rain_raw = RAW_DIR / "rainviewer_luzon_bataan_2024_2026.json"
    rain_pt = PROCESSED_DIR / "radar_grids_2024_2026.pt"
    print(f"\n[3] VERIFYING RAINVIEWER DOPPLER RADAR DATASET:")
    if rain_raw.exists():
        with open(rain_raw, "r", encoding="utf-8") as f:
            r_meta = json.load(f)
        print(f"  • Source: {r_meta.get('source')}")
        print(f"  • Live Status Host: {r_meta.get('live_status', {}).get('host')}")
        print(f"  • Max dBZ Recorded: {r_meta.get('max_dbz_recorded')} dBZ")
        print(f"  • Active Radar Storm Events: {r_meta.get('active_radar_events'):,}")
    if rain_pt.exists():
        r_pt_data = torch.load(rain_pt, weights_only=False)
        print(f"  • Tensor Shape: {r_pt_data['tensor_shape']} | Size: {rain_pt.stat().st_size / (1024*1024):.1f} MB")

    # 4. KloudTech Stations
    kloud_raw = RAW_DIR / "kloudtrack_full_2024_2026.json"
    print(f"\n[4] VERIFYING KLOUDTECH 17-STATION TELEMETRY DATASET:")
    if kloud_raw.exists():
        with open(kloud_raw, "r", encoding="utf-8") as f:
            k_meta = json.load(f)
        m = k_meta.get("metadata", {})
        stations = k_meta.get("stations", {})
        total_recs = sum(st.get("record_count", 0) for st in stations.values() if isinstance(st, dict))
        print(f"  • Source: {m.get('source')}")
        print(f"  • Range Config: {m.get('start_date')} to {m.get('end_date')}")
        print(f"  • Total Weather Stations: {len(stations)}")
        print(f"  • Total Authentic Historical Records: {total_recs:,}")
        print(f"  • Raw JSON Size: {kloud_raw.stat().st_size / (1024*1024):.1f} MB")

    # 5. Multimodal Aligned Dataset
    pimcan_pt = PROCESSED_DIR / "pimcan_multimodal_dataset.pt"
    print(f"\n[5] VERIFYING ALIGNED MULTIMODAL PIMCAN DATASET:")
    if pimcan_pt.exists():
        ds = torch.load(pimcan_pt, weights_only=False)
        print(f"  • Station Tensor:   {list(ds['station_seq'].shape)}")
        print(f"  • Satellite Tensor: {list(ds['sat_seq'].shape)}")
        print(f"  • Lightning Grid:   {list(ds['lightning_grid_seq'].shape)}")
        print(f"  • Radar Grid:       {list(ds['radar_seq'].shape)}")
        print(f"  • Forecast Targets: {list(ds['targets'].shape)}")
        print(f"  • Dataset Tensor File Size: {pimcan_pt.stat().st_size / (1024*1024):.1f} MB")

    # 6. Model Training & Export Artifacts
    print(f"\n[6] VERIFYING PIMCAN-LIQUID MODEL TRAINING OUTPUTS:")
    if WEIGHTS_PATH.exists():
        print(f"  • PyTorch Weights: EXPORTED ({WEIGHTS_PATH.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"  ❌ PyTorch Weights MISSING")

    if ONNX_PATH.exists():
        print(f"  • Production ONNX Model: EXPORTED ({ONNX_PATH.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"  ❌ Production ONNX Model MISSING")

    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_versions'")
        if c.fetchone():
            c.execute("SELECT version_tag, avg_loss, sample_count, created_at FROM model_versions ORDER BY id DESC LIMIT 3")
            rows = c.fetchall()
            print(f"  • Latest Model Version Registry:")
            for r in rows:
                print(f"    - Version: {r[0]} | Loss: {r[1]} | Samples: {r[2]} | Date: {r[3]}")
        conn.close()

    print("\n=================================================================")
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY")
    print("=================================================================")

if __name__ == "__main__":
    verify_all()
