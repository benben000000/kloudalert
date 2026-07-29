#!/usr/bin/env python3
"""
Open-Meteo Historical Data Ingestion Pipeline
Fetches high-resolution weather data (Temperature, Humidity, Pressure, Precipitation, Wind)
for training the Liquid Neural Network (LTC) weather anomaly predictor.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = WORKSPACE_ROOT / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

# Default location: Balanga City, Bataan, Philippines (Lat: 14.6775, Lon: 120.5431)
DEFAULT_LAT = 14.6775
DEFAULT_LON = 120.5431

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

class OpenMeteoFetcher:
    def __init__(self, lat=DEFAULT_LAT, lon=DEFAULT_LON):
        self.lat = lat
        self.lon = lon

    def fetch_historical_data(self, start_date="2023-01-01", end_date="2026-07-25", output_file="bataan_weather_historical.json"):
        """
        Fetches multi-year hourly weather data for Bataan, Philippines from Open-Meteo Archive API (2023-2026).
        """
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,precipitation,rain,wind_speed_10m,wind_gusts_10m",
            "timezone": "Asia/Manila"
        }

        query_string = urllib.parse.urlencode(params)
        full_url = f"{OPEN_METEO_ARCHIVE_URL}?{query_string}"
        
        print(f"Fetching Open-Meteo data from: {full_url}")
        
        req = urllib.request.Request(full_url, headers={"User-Agent": "WeatherAnomalyLNN/1.0"})
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    raw_body = response.read().decode("utf-8")
                    data = json.loads(raw_body)
                    
                    hourly = data.get("hourly", {})
                    timestamps = hourly.get("time", [])
                    print(f"Successfully fetched {len(timestamps)} hourly data records!")

                    output_path = DATA_RAW_DIR / output_file
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)

                    return {
                        "success": True,
                        "records_count": len(timestamps),
                        "output_path": str(output_path),
                        "sample_features": list(hourly.keys())
                    }
                else:
                    return {"success": False, "error": f"HTTP Status {response.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    fetcher = OpenMeteoFetcher()
    # Fetch multi-year hourly weather data (2023-01-01 to 2026-07-25)
    result = fetcher.fetch_historical_data(start_date="2023-01-01", end_date="2026-07-25")
    print(json.dumps(result, indent=2))
