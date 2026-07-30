#!/usr/bin/env python3
"""
Data Quality & Outlier Guardrail Module (`src/engine/data_quality_guard.py`)
Ensures robust telemetry data ingestion for the LFM self-improving agentic loop.
Filters noisy sensor anomalies (e.g., pressure spikes, unphysical temperatures) before PyTorch fine-tuning.
"""

import math
import time

class DataQualityGuard:
    # Physical sensor boundaries for Philippines / Tropical Bataan region
    MIN_TEMP_C = 10.0
    MAX_TEMP_C = 55.0
    MIN_HUMIDITY = 5.0
    MAX_HUMIDITY = 100.0
    MIN_PRESSURE_HPA = 850.0
    MAX_PRESSURE_HPA = 1100.0
    MAX_WIND_SPEED_MS = 90.0

    @classmethod
    def validate_telemetry_reading(cls, reading: dict) -> tuple[bool, str]:
        """
        Validates a single telemetry reading dictionary.
        Returns (is_valid, reason_if_invalid).
        """
        if not isinstance(reading, dict):
            return False, "Payload is not a valid JSON object"

        # Check required physical fields
        temp = reading.get("temperature", reading.get("temp"))
        humidity = reading.get("humidity")
        pressure = reading.get("barometric_pressure", reading.get("pressure"))
        lat = reading.get("latitude", reading.get("lat"))
        lon = reading.get("longitude", reading.get("lon"))

        if any(v is None for v in [temp, humidity, pressure, lat, lon]):
            return False, "Missing required sensor telemetry fields (temp, humidity, pressure, lat, lon)"

        try:
            temp = float(temp)
            humidity = float(humidity)
            pressure = float(pressure)
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return False, "Sensor telemetry contains non-numeric values"

        # Validate Physical Boundaries
        if not (cls.MIN_TEMP_C <= temp <= cls.MAX_TEMP_C):
            return False, f"Temperature {temp}°C out of physical bounds [{cls.MIN_TEMP_C}, {cls.MAX_TEMP_C}]"

        if not (cls.MIN_HUMIDITY <= humidity <= cls.MAX_HUMIDITY):
            return False, f"Humidity {humidity}% out of physical bounds [{cls.MIN_HUMIDITY}, {cls.MAX_HUMIDITY}]"

        if not (cls.MIN_PRESSURE_HPA <= pressure <= cls.MAX_PRESSURE_HPA):
            return False, f"Pressure {pressure} hPa out of physical bounds [{cls.MIN_PRESSURE_HPA}, {cls.MAX_PRESSURE_HPA}]"

        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return False, f"Coordinates ({lat}, {lon}) out of global geographic bounds"

        return True, "Valid"

    @classmethod
    def filter_telemetry_batch(cls, batch: list[dict]) -> tuple[list[dict], int]:
        """
        Filters a batch of raw telemetry readings before PyTorch training pass.
        Returns (clean_batch, rejected_count).
        """
        clean = []
        rejected = 0
        for item in batch:
            valid, reason = cls.validate_telemetry_reading(item)
            if valid:
                clean.append(item)
            else:
                rejected += 1
                print(f"[DATA GUARD] Rejected anomaly reading: {reason}")
        return clean, rejected

if __name__ == "__main__":
    test_reading = {
        "temperature": 32.5,
        "humidity": 78.0,
        "barometric_pressure": 1008.2,
        "latitude": 14.6775,
        "longitude": 120.5431
    }
    is_valid, msg = DataQualityGuard.validate_telemetry_reading(test_reading)
    print(f"[TEST] Reading valid: {is_valid} ({msg})")
