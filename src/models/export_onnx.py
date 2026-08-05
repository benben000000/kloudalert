#!/usr/bin/env python3
"""
ONNX Model Exporter Module for PIMCAN-Liquid Architecture
(`src/models/export_onnx.py`)

Exports trained PyTorch PIMCAN-Liquid model to ONNX format for low-latency web/mobile edge inference.
"""

import os
import sys
import json
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from pimcan_liquid_model import PIMCANLiquidModel

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = WORKSPACE_ROOT / "src" / "models" / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "pimcan_liquid_weights.pt"
WEB_APP_ONNX_PATH = WORKSPACE_ROOT / "web_app" / "lnn_weather_model.onnx"

class ONNXWrapper(torch.nn.Module):
    def __init__(self, core_model):
        super().__init__()
        self.core = core_model

    def forward(self, st, sat, lgt, rdr):
        out = self.core(st, sat, lgt, rdr)
        return out["anomaly_probability_curve"]

def export_pimcan_to_onnx():
    print(f"[EXPORT ONNX] Loading PIMCAN-Liquid model from {WEIGHTS_PATH}...")
    model = PIMCANLiquidModel(station_dim=8, sat_dim=4, hidden_dim=32, fused_dim=64, output_steps=18)
    if WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
        print("[EXPORT ONNX] PyTorch weights loaded successfully.")
    else:
        print("[EXPORT ONNX] Warning: Weights file not found, exporting baseline architecture.")

    wrapper = ONNXWrapper(model)
    wrapper.eval()

    dummy_st = torch.randn(1, 24, 8)
    dummy_sat = torch.randn(1, 24, 4)
    dummy_lgt = torch.randn(1, 24, 4, 32, 32)
    dummy_rdr = torch.randn(1, 24, 1)

    WEB_APP_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.onnx.export(
            wrapper,
            (dummy_st, dummy_sat, dummy_lgt, dummy_rdr),
            str(WEB_APP_ONNX_PATH),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["station_seq", "sat_seq", "lightning_grid_seq", "radar_seq"],
            output_names=["anomaly_probability_curve"],
            dynamo=False
        )
        size_kb = round(os.path.getsize(WEB_APP_ONNX_PATH) / 1024.0, 2)
        print(f"[EXPORT ONNX] Export Successful! ONNX Binary Size: {size_kb} KB at {WEB_APP_ONNX_PATH}")
        return {"success": True, "onnx_path": str(WEB_APP_ONNX_PATH), "size_kb": size_kb}
    except Exception as e:
        print(f"[EXPORT ONNX] Export Error: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    res = export_pimcan_to_onnx()
    print(json.dumps(res, indent=2))
