#!/usr/bin/env python3
"""
Himawari-9 Satellite Data Preprocessor & Feature Extraction Engine
(`src/engine/himawari_preprocessor.py`)

Processes Himawari-9 Band 13 (Infrared 10.4µm) and Band 8 (Water Vapor 6.2µm)
satellite telemetry into normalized feature vectors for Multimodal LFM-230M.
"""

import sys
import json
import math
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
SATELLITE_FILE = WORKSPACE_ROOT / "data" / "raw" / "himawari9_bataan_satellite.json"

def extract_himawari9_feature_vector(satellite_data=None):
    """
    Extracts 4-dimensional satellite feature vector for LFM-230M:
    [
        0: Cloud Top Min Temperature (°C),
        1: Deep Convective Coverage Ratio (0.0 to 1.0),
        2: 30-min Cloud Cooling Velocity (°C/min),
        3: Upper-Level Water Vapor Index (0.0 to 100.0)
    ]
    """
    if satellite_data is None and SATELLITE_FILE.exists():
        try:
            with open(SATELLITE_FILE, "r", encoding="utf-8") as f:
                satellite_data = json.load(f)
        except Exception:
            satellite_data = None

    if satellite_data and "bands" in satellite_data:
        b13 = satellite_data["bands"].get("band_13_clean_ir", {})
        b8 = satellite_data["bands"].get("band_8_water_vapor", {})

        tb_min = float(b13.get("cloud_top_temp_min_celsius", -45.0))
        convective_pct = float(b13.get("deep_convective_coverage_pct", 10.0)) / 100.0
        cooling_vel = float(b13.get("cooling_rate_30min_cpm", -0.5))
        wv_index = float(b8.get("moisture_index", 50.0))

        return [
            round(tb_min, 2),
            round(convective_pct, 4),
            round(cooling_vel, 4),
            round(wv_index, 2)
        ]

    # Baseline Default Satellite Features
    return [-45.0, 0.10, -0.50, 50.0]

def compute_multimodal_feature_matrix(ground_features_24x8, satellite_vec_4d=None):
    """
    Fuses 8-dimensional ground station IDW features with 4-dimensional Himawari-9
    satellite features to form a 12-dimensional multimodal sequence matrix (24 x 12).
    """
    if satellite_vec_4d is None:
        satellite_vec_4d = extract_himawari9_feature_vector()

    multimodal_matrix = []
    for step_vec in ground_features_24x8:
        multimodal_step = list(step_vec) + list(satellite_vec_4d)
        multimodal_matrix.append(multimodal_step)

    return multimodal_matrix

if __name__ == "__main__":
    sat_vec = extract_himawari9_feature_vector()
    print("=================================================================")
    print("HIMAWARI-9 SATELLITE FEATURE PREPROCESSOR VERIFICATION")
    print("=================================================================")
    print(f"Extracted 4D Satellite Feature Vector: {sat_vec}")
    print("  0: Cloud Top Min Temp:       ", sat_vec[0], "°C")
    print("  1: Convective Coverage Ratio:", sat_vec[1])
    print("  2: Cloud Cooling Velocity:   ", sat_vec[2], "°C/min")
    print("  3: Water Vapor Index:        ", sat_vec[3])
    print("=================================================================")
