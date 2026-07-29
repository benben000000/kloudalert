#!/usr/bin/env python3
"""
Location-Aware Multi-Station IDW Geospatial Router
Calculates Haversine great-circle distances between user GPS position and multiple Automated Weather Stations (AWS),
applying Inverse Distance Weighting (IDW) to fuse real-time spatial weather vectors.
"""

import math
import json
from pathlib import Path

# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0088

class MultiStationIDWRouter:
    def __init__(self, idw_power=2.0, max_stations=5):
        """
        :param idw_power: Distance decay parameter p for Inverse Distance Weighting (w_i = 1 / d_i^p)
        :param max_stations: Maximum number of nearest stations to weight
        """
        self.idw_power = idw_power
        self.max_stations = max_stations

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculates great-circle distance in kilometers between two GPS coordinates using Haversine formula.
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

        return EARTH_RADIUS_KM * c

    def compute_weights_and_interpolate(self, user_lat, user_lon, stations_data):
        """
        Computes IDW spatial weights and returns fused sensor vector for user position.

        :param user_lat: Current User Latitude
        :param user_lon: Current User Longitude
        :param stations_data: List of dicts, e.g. [
            {"station_id": "AWS-01", "lat": 25.77, "lon": -80.19, "readings": {"temperature": 28.5, "pressure": 1012.3, "rain": 0.0}}, ...
        ]
        """
        if not stations_data:
            raise ValueError("stations_data cannot be empty")

        station_distances = []
        for station in stations_data:
            dist = self.haversine_distance(user_lat, user_lon, station["lat"], station["lon"])
            station_distances.append((dist, station))

        # Sort by distance ascending
        station_distances.sort(key=lambda x: x[0])
        nearest = station_distances[:self.max_stations]

        # Exact match check (distance < 0.001 km)
        if nearest[0][0] < 0.001:
            return {
                "fused_vector": nearest[0][1]["readings"],
                "nearest_station_id": nearest[0][1]["station_id"],
                "nearest_distance_km": round(nearest[0][0], 4),
                "station_weights": {nearest[0][1]["station_id"]: 1.0}
            }

        # Calculate IDW weights
        total_weight = 0.0
        weights = []
        for dist, station in nearest:
            w = 1.0 / (dist ** self.idw_power)
            weights.append((w, station, dist))
            total_weight += w

        # Interpolate features
        sensor_keys = nearest[0][1]["readings"].keys()
        fused_vector = {key: 0.0 for key in sensor_keys}
        weight_distribution = {}

        for w, station, dist in weights:
            normalized_weight = w / total_weight
            weight_distribution[station["station_id"]] = round(normalized_weight, 4)
            for key in sensor_keys:
                fused_vector[key] += station["readings"][key] * normalized_weight

        return {
            "user_coords": {"lat": user_lat, "lon": user_lon},
            "fused_vector": {k: round(v, 4) for k, v in fused_vector.items()},
            "nearest_station_id": nearest[0][1]["station_id"],
            "nearest_distance_km": round(nearest[0][0], 4),
            "station_weights": weight_distribution
        }

if __name__ == "__main__":
    # Unit demonstration with 3 simulated Automated Weather Stations (AWS)
    simulated_stations = [
        {"station_id": "AWS-Miami-North", "lat": 25.80, "lon": -80.19, "readings": {"temperature": 29.1, "pressure": 1008.2, "rain_rate": 12.5}},
        {"station_id": "AWS-Miami-Beach", "lat": 25.78, "lon": -80.13, "readings": {"temperature": 27.8, "pressure": 1010.5, "rain_rate": 2.1}},
        {"station_id": "AWS-Miami-South", "lat": 25.72, "lon": -80.24, "readings": {"temperature": 30.2, "pressure": 1006.8, "rain_rate": 25.0}}
    ]

    # User commuting at lat: 25.76, lon: -80.20
    router = MultiStationIDWRouter()
    res = router.compute_weights_and_interpolate(25.76, -80.20, simulated_stations)
    print(json.dumps(res, indent=2))
