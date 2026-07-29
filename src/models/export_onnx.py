#!/usr/bin/env python3
"""
ONNX Model Exporter Module
Exports trained PyTorch Liquid Neural Network (LTC) model to ONNX format
for low-latency mobile edge inference.
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add parent directory for module imports
sys.path.append(str(Path(__file__).resolve().parent))
from lnn_model import LiquidNeuralNetwork

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = WORKSPACE_ROOT / "src" / "models" / "weights"
META_PATH = WEIGHTS_DIR / "model_meta.json"
ONNX_OUT_PATH = WEIGHTS_DIR / "lnn_weather_model.onnx"

def export_lnn_to_onnx():
    if not META_PATH.exists():
        raise FileNotFoundError(f"Model metadata not found at {META_PATH}")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    input_dim = meta["input_dim"]
    hidden_dim = meta["hidden_dim"]
    output_steps = meta["output_steps"]
    weights_path = Path(meta["weights_path"])

    print(f"Loading trained PyTorch model from {weights_path}...")
    model = LiquidNeuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_steps=output_steps)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    # Dummy input sequence: (batch_size=1, seq_len=24, input_dim=8)
    dummy_input = torch.randn(1, 24, input_dim, dtype=torch.float32)

    # Export TorchScript for Native Mobile Edge (PyTorch Mobile / C++ LibTorch)
    TS_OUT_PATH = WEIGHTS_DIR / "lnn_weather_model.ptc"
    print(f"Exporting model to TorchScript Mobile format at {TS_OUT_PATH}...")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(str(TS_OUT_PATH))
    ts_size_kb = round(os.path.getsize(TS_OUT_PATH) / 1024.0, 2)
    print(f"TorchScript Mobile Export Successful! File Size: {ts_size_kb} KB")

    # Export ONNX if onnxscript / ONNX dependencies available
    WEB_APP_ONNX_PATH = WORKSPACE_ROOT / "web_app" / "lnn_weather_model.onnx"
    onnx_exported = False
    onnx_size_kb = 0.0
    try:
        print(f"Exporting model to ONNX at {ONNX_OUT_PATH} and {WEB_APP_ONNX_PATH}...")
        torch.onnx.export(
            model,
            dummy_input,
            str(ONNX_OUT_PATH),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_weather_sequence'],
            output_names=['anomaly_probability_curve'],
            dynamo=False
        )
        torch.onnx.export(
            model,
            dummy_input,
            str(WEB_APP_ONNX_PATH),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_weather_sequence'],
            output_names=['anomaly_probability_curve'],
            dynamo=False
        )
        onnx_size_kb = round(os.path.getsize(WEB_APP_ONNX_PATH) / 1024.0, 2)
        onnx_exported = True
        print(f"ONNX Export Successful! File Size: {onnx_size_kb} KB at {WEB_APP_ONNX_PATH}")
    except Exception as e:
        print(f"ONNX export exception: {type(e).__name__} - {e}. TorchScript Mobile format ({TS_OUT_PATH}) ready for edge deployment.")

    return {
        "success": True,
        "torchscript_path": str(TS_OUT_PATH),
        "torchscript_size_kb": ts_size_kb,
        "onnx_exported": onnx_exported,
        "onnx_size_kb": onnx_size_kb,
        "input_shape": [1, 24, input_dim],
        "output_shape": [1, output_steps]
    }

if __name__ == "__main__":
    res = export_lnn_to_onnx()
    print(json.dumps(res, indent=2))
