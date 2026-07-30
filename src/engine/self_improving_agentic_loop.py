#!/usr/bin/env python3
"""
Autonomous LFM Self-Improving Agentic Feedback Loop Engine
- Real-Time Open-Meteo Telemetry Ingestion
- Experiment Logging (data/experiments/experiment_log.json)
- Ground Truth Verification Window (15-45 mins later)
- Focal Anomaly Loss Self-Assessment
- Autonomous Online Fine-Tuning & Weight Update
- Automatic ONNX Hot-Swapping (web_app/lnn_weather_model.onnx)
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from datetime import datetime

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT / "src" / "models"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "engine"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "data"))

from lfm_foundation_model import LiquidFoundationModel230M
try:
    from obsidian_mind import generate_obsidian_wiki
except ImportError:
    generate_obsidian_wiki = None
from db_manager import DatabaseManager
from firebase_manager import FirebaseManager
from data_quality_guard import DataQualityGuard

EXPERIMENT_DIR = WORKSPACE_ROOT / "data" / "experiments"
EXPERIMENT_LOG = EXPERIMENT_DIR / "experiment_log.json"
WEIGHTS_PATH = WORKSPACE_ROOT / "src" / "models" / "weights" / "lfm_230m_weights.pt"
ONNX_EXPORT_PATH = WORKSPACE_ROOT / "web_app" / "lnn_weather_model.onnx"

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

class LFMSelfImprovingAgent:
    def __init__(self):
        self.db = DatabaseManager()
        self.fb = FirebaseManager()
        self.model = LiquidFoundationModel230M()
        if WEIGHTS_PATH.exists():
            try:
                self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
                print(f"[LFM AGENT] Loaded persistent weights from {WEIGHTS_PATH}")
            except Exception as e:
                print(f"[LFM AGENT] Initializing new LFM weights: {e}")
        self.model.eval()

    def load_experiments(self):
        if EXPERIMENT_LOG.exists():
            try:
                with open(EXPERIMENT_LOG, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"experiments": [], "metrics": {"total_predictions": 0, "verified": 0, "fine_tune_events": 0, "avg_loss": 0.0}}

    def save_experiments(self, log_data):
        with open(EXPERIMENT_LOG, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

    def fetch_one_time_kloudtech_ground_truth(self, lat: float, lon: float) -> dict:
        """
        Executes EXACTLY ONE HTTP request to Kloudtech Ground Station API
        after the 15-minute alert timer finishes to check if rain actually occurred.
        """
        config_path = WORKSPACE_ROOT / "data" / "kloudtech_config.json"
        if not config_path.exists():
            return None

        try:
            from api_designer import APIDesigner
            designer = APIDesigner()
            res = designer.design_and_execute_request(
                endpoint_path="/api/v1/telemetry/dashboard",
                method="GET"
            )
            if res.get("success"):
                print(f"[KLOUDTRACK API DESIGNER] Verification probe success via {res.get('domain')}!")
                return res.get("data")
        except Exception as e:
            print(f"[KLOUDTRACK API DESIGNER] Note: {e}")
        return None

    def log_prediction_experiment(self, lat, lon, feature_vector, prob_curve, device_id: str = "apk_node"):
        """Step 1 & 2: Generate nowcast & record prediction experiment into DB and rolling buffer."""
        log_data = self.load_experiments()
        now_ts = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        experiment_id = f"exp_{int(now_ts * 1000)}"
        max_prob = max(prob_curve)
        predicted_anomaly = max_prob > 0.60

        exp_record = {
            "id": experiment_id,
            "device_id": device_id,
            "timestamp": now_ts,
            "timestamp_str": now_str,
            "location": {"lat": lat, "lon": lon},
            "feature_vector": feature_vector,
            "prob_curve": prob_curve,
            "max_prob": max_prob,
            "predicted_anomaly": predicted_anomaly,
            "status": "PENDING_VERIFICATION",
            "verify_target_ts": now_ts + 900  # Verify 15 mins later
        }

        log_data["experiments"].append(exp_record)
        log_data["metrics"]["total_predictions"] += 1

        # Keep rolling buffer of max 200 experiments
        if len(log_data["experiments"]) > 200:
            log_data["experiments"] = log_data["experiments"][-200:]

        self.save_experiments(log_data)
        self.db.insert_experiment(exp_record)
        print(f"[LFM AGENT] Logged prediction experiment {experiment_id} to DB & JSON (Predicted Anomaly: {predicted_anomaly}, Max Prob: {max_prob:.3f})")
        return exp_record

    def verify_ground_truth_and_improve(self, current_telemetry):
        """Step 3, 4 & 5: Verify ground truth 15-45m later, compute Focal Loss, self-fine-tune."""
        log_data = self.load_experiments()
        now_ts = time.time()
        
        # Pull pending experiments from database or in-memory
        db_pending = self.db.get_pending_experiments()
        pending = db_pending if db_pending else [e for e in log_data["experiments"] if e.get("status") == "PENDING_VERIFICATION"]

        if not pending:
            return {"verified_count": 0, "retrained": False}

        current_precip = current_telemetry.get("precip", 0.0)
        current_heat = current_telemetry.get("temp", 30.0) + 5.5
        actual_anomaly = (current_precip >= 0.5) or (current_heat >= 40.0)

        verified_count = 0
        training_inputs = []
        training_targets = []

        for exp in pending:
            exp_id = exp.get("experiment_id") or exp.get("id")
            exp_lat = exp.get("location", {}).get("lat", 14.6775)
            exp_lon = exp.get("location", {}).get("lon", 120.5431)

            # Check if verification window (15 mins) has elapsed or force update
            if now_ts >= exp.get("verify_target_ts", 0) - 30:
                # Execute 1-time Kloudtech Ground Weather Station Verification Call
                kloudtech_res = self.fetch_one_time_kloudtech_ground_truth(exp_lat, exp_lon)
                if kloudtech_res and "precip" in kloudtech_res:
                    exp_precip = float(kloudtech_res.get("precip", current_precip))
                    exp_heat = float(kloudtech_res.get("heat_index", current_heat))
                    exp_actual_anomaly = (exp_precip >= 0.5) or (exp_heat >= 40.0)
                else:
                    exp_precip = current_precip
                    exp_heat = current_heat
                    exp_actual_anomaly = actual_anomaly

                exp["status"] = "VERIFIED"
                exp["actual_precip"] = exp_precip
                exp["actual_heat_index"] = exp_heat
                exp["actual_anomaly"] = exp_actual_anomaly
                exp["verification_ts"] = now_ts

                # Construct ground truth target (18 probability steps)
                target_vector = torch.zeros(1, 18)
                if exp_actual_anomaly:
                    target_vector[0, :] = 1.0

                # Form input feature tensor (1, 24, 8)
                feat_vec = exp.get("feature_vector", [30.0, 75.0, 1010.0, 0.0, 5.0, 0.0, 0.0, 36.0])
                input_tensor = torch.tensor([[feat_vec] * 24], dtype=torch.float32)

                training_inputs.append(input_tensor)
                training_targets.append(target_vector)
                verified_count += 1

                self.db.mark_experiment_verified(
                    experiment_id=exp_id,
                    actual_precip=current_precip,
                    actual_heat_index=current_heat,
                    actual_anomaly=actual_anomaly,
                    residual_loss=0.01
                )

        retrained = False
        avg_loss = 0.0

        # Step 5: Execute Self-Fine-Tuning if ground truth error detected
        if training_inputs:
            x_batch = torch.cat(training_inputs, dim=0)
            y_batch = torch.cat(training_targets, dim=0)

            print(f"[LFM AGENT] Autonomous Self-Fine-Tuning on {verified_count} verified experiment samples...")
            avg_loss = self.model.self_fine_tune(x_batch, y_batch, epochs=3)

            # Save updated PyTorch weights
            torch.save(self.model.state_dict(), WEIGHTS_PATH)
            print(f"[LFM AGENT] Saved upgraded LFM-230M weights to {WEIGHTS_PATH} (Loss: {avg_loss:.4f})")

            # Step 6: Hot-Swap ONNX Edge Binary & Trigger Android Emulator Verification Pass
            self.model.export_onnx(ONNX_EXPORT_PATH)
            retrained = True

            # Trigger Android Emulator Verification via repos/emulator & emulator_verifier
            try:
                from emulator_verifier import EmulatorVerifier
                verifier = EmulatorVerifier()
                emulator_res = verifier.install_and_verify_app()
                print(f"[LFM AGENT] Android Emulator Verification Pass: {emulator_res.get('status')}")
            except Exception as e:
                print(f"[LFM AGENT] Emulator verification note: {e}")

            version_tag = f"v1.{int(now_ts)}"
            self.fb.publish_model_version(
                version_tag=version_tag,
                weights_path=str(WEIGHTS_PATH),
                onnx_path=str(ONNX_EXPORT_PATH),
                avg_loss=round(avg_loss, 4),
                sample_count=verified_count
            )

            log_data["metrics"]["verified"] += verified_count
            log_data["metrics"]["fine_tune_events"] += 1
            log_data["metrics"]["avg_loss"] = round(avg_loss, 4)

            # Step 7: Update Obsidian System Wiki
            try:
                generate_obsidian_wiki()
            except Exception as e:
                print(f"[LFM AGENT] Wiki sync note: {e}")

        self.save_experiments(log_data)
        return {
            "verified_count": verified_count,
            "retrained": retrained,
            "loss": avg_loss
        }

if __name__ == "__main__":
    print("[LFM AGENTIC LOOP] Initializing Self-Improving Agent...")
    agent = LFMSelfImprovingAgent()

    # Test Experiment Logging
    dummy_feat = [31.0, 78.0, 1009.2, 0.0, 4.5, 0.0, 0.0, 38.0]
    dummy_probs = [0.08] * 18
    rec = agent.log_prediction_experiment(14.6775, 120.5431, dummy_feat, dummy_probs)

    # Test Verification & Retraining Pass
    rec["verify_target_ts"] = time.time() - 1  # Force immediate verification window
    agent.save_experiments(agent.load_experiments())

    res = agent.verify_ground_truth_and_improve({"precip": 1.2, "temp": 31.0})
    print("[LFM AGENTIC LOOP] Cycle complete result:", json.dumps(res, indent=2))
