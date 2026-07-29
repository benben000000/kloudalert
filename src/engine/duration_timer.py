#!/usr/bin/env python3
"""
Anomaly Duration Countdown Engine ("When Will It End?")
Parses 18-step probability curves output by the Liquid Neural Network (LTC) model,
computing anomaly onset, duration countdown, severity classification, and user alerts.
"""

import json
from pathlib import Path

STEP_INTERVAL_MINUTES = 2.5  # Each of the 18 steps represents 2.5 minutes (Total 45 minutes)

class AnomalyDurationEngine:
    def __init__(self, onset_threshold=0.50, dissipation_threshold=0.35):
        self.onset_threshold = onset_threshold
        self.dissipation_threshold = dissipation_threshold

    def calculate_duration_and_alert(self, probability_curve):
        """
        Parses 18-step probability curve vector and outputs duration countdown metrics.

        :param probability_curve: List of 18 float probabilities [0.0 - 1.0] for t+2.5min .. t+45min
        """
        if len(probability_curve) != 18:
            raise ValueError(f"Expected probability curve with 18 steps, got {len(probability_curve)}")

        max_prob = max(probability_curve)
        peak_step = probability_curve.index(max_prob)
        peak_time_mins = round((peak_step + 1) * STEP_INTERVAL_MINUTES, 1)

        # Check if an anomaly is predicted
        if max_prob < self.onset_threshold:
            return {
                "anomaly_detected": False,
                "status": "NORMAL",
                "severity": "LOW",
                "max_probability": round(max_prob, 4),
                "alert_text": "Clear weather conditions. No anomaly expected in the next 45 minutes.",
                "summary": "No weather anomaly expected in the next 45 minutes.",
                "countdown_minutes": 0,
                "duration_minutes": 0
            }

        # Find onset step (first step >= onset_threshold)
        onset_step = None
        for idx, prob in enumerate(probability_curve):
            if prob >= self.onset_threshold:
                onset_step = idx
                break

        # Find dissipation step (first step after onset < dissipation_threshold)
        dissipation_step = 17
        for idx in range(onset_step, len(probability_curve)):
            if probability_curve[idx] < self.dissipation_threshold:
                dissipation_step = idx - 1
                break

        onset_time_mins = round((onset_step + 1) * STEP_INTERVAL_MINUTES, 1)
        end_time_mins = round((dissipation_step + 1) * STEP_INTERVAL_MINUTES, 1)
        duration_mins = max(round(end_time_mins - onset_time_mins + STEP_INTERVAL_MINUTES, 1), STEP_INTERVAL_MINUTES)

        # Severity Classification
        if max_prob >= 0.85:
            severity = "SEVERE"
            anomaly_type = "Heavy Rain & Squall Barrage"
        elif max_prob >= 0.70:
            severity = "HIGH"
            anomaly_type = "Heavy Rainfall"
        else:
            severity = "MODERATE"
            anomaly_type = "Light Rain & Passing Shower"

        # Generate User Alert String
        if onset_step == 0:
            status = "ACTIVE"
            alert_text = f"{anomaly_type} currently active. Expected to end in {int(end_time_mins)} minutes."
        else:
            status = "INCOMING"
            alert_text = f"{anomaly_type} expected to begin in {int(onset_time_mins)} mins and end in {int(end_time_mins)} mins (Duration: {int(duration_mins)} mins)."

        return {
            "anomaly_detected": True,
            "anomaly_type": anomaly_type,
            "status": status,
            "severity": severity,
            "max_probability": round(max_prob, 4),
            "onset_time_mins": onset_time_mins,
            "end_time_mins": end_time_mins,
            "duration_minutes": duration_mins,
            "alert_text": alert_text,
            "timeline_curve": [
                {"time_min": round((i + 1) * STEP_INTERVAL_MINUTES, 1), "probability": round(prob, 4)}
                for i, prob in enumerate(probability_curve)
            ]
        }

if __name__ == "__main__":
    # Test demonstration with simulated incoming storm curve
    simulated_curve = [
        0.12, 0.25, 0.48, 0.72, 0.89, 0.94, 0.91, 0.83, 0.67,
        0.52, 0.38, 0.24, 0.15, 0.10, 0.08, 0.05, 0.04, 0.02
    ]
    engine = AnomalyDurationEngine()
    result = engine.calculate_duration_and_alert(simulated_curve)
    print(json.dumps(result, indent=2))
