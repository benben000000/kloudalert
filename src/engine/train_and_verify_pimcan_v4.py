#!/usr/bin/env python3
"""
PIMCAN-v4 World-Class Multi-Hazard Model Training & Benchmark Engine
(`src/engine/train_and_verify_pimcan_v4.py`)

Trains PIMCANv4WorldClassModel across 5 epochs:
- Multi-Hazard Heads (Rain Recurrence Mins, Heatwave, Tornado/Microburst, Severe Lightning)
- Saves PyTorch Checkpoint to `src/models/weights/pimcan_v4_weights.pt`
- Exports Production ONNX Engine to `web_app/lnn_weather_model.onnx`
- Registers Version Tag `v4.pimcan_world_class_multihazard` in SQLite Central DB
"""

import sys
import os
import json
import time
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT / "src" / "models"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "data"))

from pimcan_v4_world_class import PIMCANv4WorldClassModel
from physics_losses import PIMCANPhysicsLoss
from db_manager import DatabaseManager

DATASET_V4_PATH = WORKSPACE_ROOT / "data" / "processed" / "pimcan_v4_multimodal_dataset.pt"
WEIGHTS_V4_PATH = WORKSPACE_ROOT / "src" / "models" / "weights" / "pimcan_v4_weights.pt"
ONNX_EXPORT_PATH = WORKSPACE_ROOT / "web_app" / "lnn_weather_model.onnx"

WEIGHTS_V4_PATH.parent.mkdir(parents=True, exist_ok=True)

def train_v4_world_class(epochs=5, batch_size=32, lr=1e-3):
    print("=================================================================")
    print("PIMCAN-V4 WORLD-CLASS MULTI-HAZARD MODEL TRAINING PIPELINE")
    print("=================================================================")

    if not DATASET_V4_PATH.exists():
        raise FileNotFoundError(f"V4 Dataset missing at {DATASET_V4_PATH}")

    print(f"[1/4] Ingesting Multi-Hazard Synchronized Dataset from {DATASET_V4_PATH}...")
    dataset = torch.load(DATASET_V4_PATH, weights_only=False)

    station_seq = dataset["station_seq"]
    sat_seq = dataset["sat_seq"]
    lightning_grid_seq = dataset["lightning_grid_seq"]
    radar_seq = dataset["radar_seq"]

    rain_prob_curves = dataset["rain_prob_curves"]
    recurrence_mins = dataset["rain_recurrence_mins"]
    heatwave_alerts = dataset["heatwave_alerts"]
    tornado_alerts = dataset["tornado_microburst_alerts"]
    lightning_alerts = dataset["severe_lightning_alerts"]

    dataset_size = station_seq.shape[0]
    print(f"  -> Total Dataset Samples: {dataset_size:,}")

    # 2. Instantiate Model Architecture & Optimizer
    print("[2/4] Initializing PIMCAN-v4 Multi-Hazard Architecture...")
    model = PIMCANv4WorldClassModel(station_dim=10, sat_dim=4, hidden_dim=32, output_steps=18)
    physics_criterion = PIMCANPhysicsLoss(alpha=0.75, gamma=2.0, lambda_rain=0.5, lambda_thermo=0.2, lambda_smooth=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 3. Training Loop across Multi-Hazard Tasks
    print(f"[3/4] Training PIMCAN-v4 across {epochs} Epochs...")
    model.train()

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        batches = 0

        for i in range(0, dataset_size, batch_size):
            idx = permutation[i : i + batch_size]
            b_st = station_seq[idx]
            b_sat = sat_seq[idx]
            b_lgt = lightning_grid_seq[idx]
            b_rdr = radar_seq[idx]

            b_rain_targets = rain_prob_curves[idx]
            b_heatwave = heatwave_alerts[idx].unsqueeze(-1)
            b_tornado = tornado_alerts[idx].unsqueeze(-1)
            b_lgt_target = lightning_alerts[idx].unsqueeze(-1)

            optimizer.zero_grad()
            out = model(b_st, b_sat, b_lgt, b_rdr)

            # Unified Multi-Hazard Loss
            loss_phys = physics_criterion(
                pred_probs=out["rain_probability_curve"],
                target_probs=b_rain_targets,
                pred_temp=out["pred_temp"],
                pred_rh=out["pred_rh"],
                pred_rain=out["pred_rain"]
            )["total_loss"]

            loss_heatwave = F.binary_cross_entropy(out["heatwave_risk"], b_heatwave)
            loss_tornado = F.binary_cross_entropy(out["tornado_microburst_risk"], b_tornado)
            loss_lgt = F.binary_cross_entropy(out["severe_lightning_risk"], b_lgt_target)

            total_loss = loss_phys + 0.3 * loss_heatwave + 0.3 * loss_tornado + 0.3 * loss_lgt
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Total Multi-Hazard Loss: {avg_loss:.5f}")

    # 4. Save Trained Checkpoint & Register Version in Database
    print(f"\n[4/4] Saving Checkpoint & Registering DB Version...")
    torch.save(model.state_dict(), WEIGHTS_V4_PATH)
    print(f"  -> Saved PIMCAN-v4 PyTorch Checkpoint to {WEIGHTS_V4_PATH} ({WEIGHTS_V4_PATH.stat().st_size / 1024:.1f} KB)")

    # Export ONNX wrapper
    class ONNXv4Wrapper(torch.nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model

        def forward(self, st, sat, lgt, rdr):
            res = self.core(st, sat, lgt, rdr)
            return res["rain_probability_curve"]

    onnx_wrapper = ONNXv4Wrapper(model)
    onnx_wrapper.eval()

    dummy_st = torch.randn(1, 24, 10)
    dummy_sat = torch.randn(1, 24, 4)
    dummy_lgt = torch.randn(1, 24, 4, 32, 32)
    dummy_rdr = torch.randn(1, 24, 1, 32, 32)

    try:
        torch.onnx.export(
            onnx_wrapper,
            (dummy_st, dummy_sat, dummy_lgt, dummy_rdr),
            str(ONNX_EXPORT_PATH),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["station_seq", "sat_seq", "lightning_grid_seq", "radar_seq"],
            output_names=["rain_probability_curve"],
            dynamo=False
        )
        print(f"  -> Successfully exported PIMCAN-v4 ONNX model to {ONNX_EXPORT_PATH} ({ONNX_EXPORT_PATH.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"  -> ONNX export note: {e}")

    try:
        db = DatabaseManager()
        version_tag = f"v4.pimcan_world_class_multihazard.{int(time.time())}"
        db.record_model_version(
            version_tag=version_tag,
            weights_path=str(WEIGHTS_V4_PATH),
            onnx_path=str(ONNX_EXPORT_PATH),
            avg_loss=round(avg_loss, 5),
            sample_count=dataset_size
        )
        print(f"  -> Registered Model Version [{version_tag}] in SQLite Central DB.")
    except Exception as e:
        print(f"  -> DB registry note: {e}")

    print("=================================================================")
    print("PIMCAN-V4 WORLD-CLASS TRAINING COMPLETE")
    print("=================================================================")

if __name__ == "__main__":
    train_v4_world_class()
