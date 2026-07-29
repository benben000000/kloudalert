#!/usr/bin/env python3
"""
Data Preprocessing & Feature Engineering Module
Transforms raw Open-Meteo historical weather data into normalized time-series sequences
and target anomaly probability curves for training the Liquid Neural Network (LTC).
"""

import os
import sys
import json
import math
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_PATH = WORKSPACE_ROOT / "data" / "raw" / "bataan_weather_historical.json"
DATA_PROCESSED_DIR = WORKSPACE_ROOT / "data" / "processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def compute_heat_index(temp_c, humidity):
    """
    Computes Heat Index (°C) using Rothfusz regression equation approximation.
    """
    temp_f = (temp_c * 9.0 / 5.0) + 32.0
    if temp_f < 80.0:
        hi_f = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094))
    else:
        hi_f = -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity \
               - 0.22475541 * temp_f * humidity - 0.00683783 * temp_f**2 \
               - 0.05481717 * humidity**2 + 0.00122874 * temp_f**2 * humidity \
               + 0.00085282 * temp_f * humidity**2 - 0.00000199 * temp_f**2 * humidity**2
    return (hi_f - 32.0) * 5.0 / 9.0

class WeatherPreprocessor:
    def __init__(self, raw_data_path=DATA_RAW_PATH):
        self.raw_data_path = Path(raw_data_path)

    def load_and_preprocess(self, input_seq_len=24, forecast_horizon_steps=18):
        """
        Loads raw Open-Meteo JSON, extracts features, calculates deltas, scales values,
        and constructs sliding window dataset.
        """
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data file not found at {self.raw_data_path}")

        with open(self.raw_data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        hourly = raw_data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        humidity = hourly.get("relative_humidity_2m", [])
        pressures = hourly.get("surface_pressure", [])
        precip = hourly.get("precipitation", [])
        wind_speeds = hourly.get("wind_speed_10m", [])

        num_records = len(times)
        print(f"Loaded {num_records} hourly records for preprocessing.")

        # Compute derived features & normalizations
        processed_records = []
        for i in range(num_records):
            t = temps[i] if temps[i] is not None else 25.0
            h = humidity[i] if humidity[i] is not None else 70.0
            p = pressures[i] if pressures[i] is not None else 1013.0
            pr = precip[i] if precip[i] is not None else 0.0
            w = wind_speeds[i] if wind_speeds[i] is not None else 10.0

            # Delta features (rate of change over previous hour)
            p_delta = (p - pressures[i-1]) if i > 0 and pressures[i-1] is not None else 0.0
            t_delta = (t - temps[i-1]) if i > 0 and temps[i-1] is not None else 0.0

            hi = compute_heat_index(t, h)

            processed_records.append({
                "temp": t,
                "humidity": h,
                "pressure": p,
                "precip": pr,
                "wind": w,
                "p_delta": p_delta,
                "t_delta": t_delta,
                "heat_index": hi
            })

        # Calculate min/max stats for normalization
        feature_keys = ["temp", "humidity", "pressure", "precip", "wind", "p_delta", "t_delta", "heat_index"]
        stats = {}
        for key in feature_keys:
            vals = [r[key] for r in processed_records]
            min_v = min(vals)
            max_v = max(vals)
            range_v = (max_v - min_v) if (max_v - min_v) > 1e-6 else 1.0
            stats[key] = {"min": min_v, "max": max_v, "range": range_v}

        # Normalize features (0.0 to 1.0)
        normalized_matrix = []
        for r in processed_records:
            vec = []
            for key in feature_keys:
                norm_val = (r[key] - stats[key]["min"]) / stats[key]["range"]
                vec.append(round(norm_val, 5))
            normalized_matrix.append(vec)

        # Construct sliding window sequences
        X_samples = []
        Y_samples = [] # 18-step probability targets for next 45 minutes

        for i in range(input_seq_len, num_records - 3):
            # Input X: past 24 hours (seq_len x num_features)
            x_seq = normalized_matrix[i - input_seq_len : i]

            # Future target Y: next 3 hours (representing 18 sub-steps of 2.5 min each)
            future_precip = [processed_records[i + k]["precip"] for k in range(3)]
            
            # Map future precipitation to 18-step probability curve
            y_curve = []
            for step in range(forecast_horizon_steps):
                hour_idx = min(step // 6, 2)
                p_val = future_precip[hour_idx]
                # Anomaly probability sigmoid curve (rain > 0.5mm -> probability > 0.7)
                prob = 1.0 / (1.0 + math.exp(-3.0 * (p_val - 0.2)))
                y_curve.append(round(prob, 4))

            X_samples.append(x_seq)
            Y_samples.append(y_curve)

        print(f"Generated {len(X_samples)} sliding window sequences.")

        dataset = {
            "num_samples": len(X_samples),
            "input_seq_len": input_seq_len,
            "num_features": len(feature_keys),
            "feature_names": feature_keys,
            "stats": stats,
            "X": X_samples,
            "Y": Y_samples
        }

        out_path = DATA_PROCESSED_DIR / "preprocessed_dataset.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f)

        print(f"Saved dataset to {out_path}")
        return {
            "success": True,
            "num_samples": len(X_samples),
            "dataset_path": str(out_path),
            "feature_names": feature_keys
        }

if __name__ == "__main__":
    processor = WeatherPreprocessor()
    res = processor.load_and_preprocess()
    print(json.dumps({k: v for k, v in res.items() if k != "X"}, indent=2))
