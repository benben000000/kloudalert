#!/usr/bin/env python3
"""
PIMCAN-Liquid Multimodal Model Training Pipeline
(`src/models/train_multimodal_lfm.py`)

Trains PIMCAN-Liquid Core Model across 4 synchronized modalities:
- Ground Stations (LTC Encoder)
- Himawari-9 Satellite (Conv-CfC Encoder)
- Blitzortung Lightning 2D Grids (CfC Grid Encoder)
- Radar Precipitation Structure (Conv-CfC Encoder)

Optimizes multi-objective Physics Loss (Focal + Non-negative Rain + Thermo Bounds + Temporal Smoothness).
Exports production ONNX model (`web_app/lnn_weather_model.onnx`).
"""

import sys
import os
import json
import time
import torch
import torch.optim as optim
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT / "src" / "models"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "engine"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "data"))

from pimcan_liquid_model import PIMCANLiquidModel
from physics_losses import PIMCANPhysicsLoss
from db_manager import DatabaseManager

DATASET_PATH = WORKSPACE_ROOT / "data" / "processed" / "pimcan_multimodal_dataset.pt"
WEIGHTS_PATH = WORKSPACE_ROOT / "src" / "models" / "weights" / "pimcan_liquid_weights.pt"
ONNX_EXPORT_PATH = WORKSPACE_ROOT / "web_app" / "lnn_weather_model.onnx"

WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

def train_pimcan_liquid(epochs=5, batch_size=32, lr=1e-3):
    print("=================================================================")
    print("PIMCAN-LIQUID MULTIMODAL MODEL TRAINING PIPELINE")
    print("=================================================================")

    if not DATASET_PATH.exists():
        print("[NOTE] Dataset missing. Invoking build_pimcan_multimodal_dataset...")
        from build_pimcan_multimodal_dataset import build_pimcan_multimodal_dataset
        build_pimcan_multimodal_dataset()

    print(f"[1/4] Ingesting Synchronized PIMCAN Multimodal Dataset from {DATASET_PATH}...")
    dataset = torch.load(DATASET_PATH, weights_only=False)

    station_seq = dataset["station_seq"]
    sat_seq = dataset["sat_seq"]
    lightning_grid_seq = dataset["lightning_grid_seq"]
    radar_seq = dataset["radar_seq"]
    targets = dataset["targets"]

    dataset_size = station_seq.shape[0]
    print(f"  -> Total Training Samples: {dataset_size:,}")

    # 2. Instantiate Model & Physics Loss Engine
    print("[2/4] Initializing PIMCAN-Liquid Network & Physics Loss Engine...")
    model = PIMCANLiquidModel(station_dim=8, sat_dim=4, hidden_dim=32, fused_dim=64, output_steps=18)
    physics_criterion = PIMCANPhysicsLoss(alpha=0.75, gamma=2.0, lambda_rain=0.5, lambda_thermo=0.2, lambda_smooth=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 3. Training Loop
    print(f"[3/4] Training PIMCAN-Liquid Core across {epochs} Epochs...")
    model.train()

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(dataset_size)
        epoch_total_loss = 0.0
        epoch_focal_loss = 0.0
        epoch_physics_penalty = 0.0
        batches = 0

        for i in range(0, dataset_size, batch_size):
            idx = permutation[i : i + batch_size]
            b_st = station_seq[idx]
            b_sat = sat_seq[idx]
            b_lgt = lightning_grid_seq[idx]
            b_rdr = radar_seq[idx]
            b_target = targets[idx]

            optimizer.zero_grad()
            outputs = model(b_st, b_sat, b_lgt, b_rdr)

            loss_dict = physics_criterion(
                pred_probs=outputs["anomaly_probability_curve"],
                target_probs=b_target,
                pred_temp=outputs["pred_temp"],
                pred_rh=outputs["pred_rh"],
                pred_rain=outputs["pred_rain"]
            )

            total_loss = loss_dict["total_loss"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_focal_loss += loss_dict["focal_loss"]
            epoch_physics_penalty += loss_dict["physics_penalty"]
            batches += 1

        avg_loss = epoch_total_loss / batches
        avg_focal = epoch_focal_loss / batches
        avg_physics = epoch_physics_penalty / batches

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Total Loss: {avg_loss:.5f} (Focal: {avg_focal:.5f} | Physics Penalty: {avg_physics:.5f})")

    # 4. Save Weights & Export ONNX
    print(f"\n[4/4] Saving Trained PyTorch Weights & Exporting Web/Mobile ONNX Model...")
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"  -> Saved PyTorch Weights to {WEIGHTS_PATH}")

    # Export simplified forward wrapper for ONNX export
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model

        def forward(self, st, sat, lgt, rdr):
            out = self.core(st, sat, lgt, rdr)
            return out["anomaly_probability_curve"]

    onnx_wrapper = ONNXWrapper(model)
    onnx_wrapper.eval()

    dummy_st = torch.randn(1, 24, 8)
    dummy_sat = torch.randn(1, 24, 4)
    dummy_lgt = torch.randn(1, 24, 4, 32, 32)
    dummy_rdr = torch.randn(1, 24, 1)

    try:
        torch.onnx.export(
            onnx_wrapper,
            (dummy_st, dummy_sat, dummy_lgt, dummy_rdr),
            str(ONNX_EXPORT_PATH),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["station_seq", "sat_seq", "lightning_grid_seq", "radar_seq"],
            output_names=["anomaly_probability_curve"],
            dynamo=False
        )
        print(f"  -> Successfully exported PIMCAN-Liquid ONNX model to {ONNX_EXPORT_PATH} ({ONNX_EXPORT_PATH.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"  -> ONNX export note: {e}")

    # Register Model Version in Database
    try:
        db = DatabaseManager()
        version_tag = f"v2.pimcan_liquid.blitzortung.{int(time.time())}"
        db.record_model_version(
            version_tag=version_tag,
            weights_path=str(WEIGHTS_PATH),
            onnx_path=str(ONNX_EXPORT_PATH),
            avg_loss=round(avg_loss, 5),
            sample_count=dataset_size
        )
        print(f"  -> Registered Model Version [{version_tag}] in SQLite Central DB.")
    except Exception as e:
        print(f"  -> DB registry note: {e}")

    print("=================================================================")
    print("PIMCAN-LIQUID MODEL TRAINING COMPLETE")
    print("=================================================================")
    return {"status": "SUCCESS", "final_loss": avg_loss, "weights_path": str(WEIGHTS_PATH)}

if __name__ == "__main__":
    train_pimcan_liquid()
