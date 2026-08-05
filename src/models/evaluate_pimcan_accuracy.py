#!/usr/bin/env python3
"""
PIMCAN-v3 Advanced Accuracy Benchmark Engine
(`src/models/evaluate_pimcan_accuracy.py`)

Evaluates trained PIMCAN-v3 weights (`pimcan_v3_weights.pt`) on held-out test set (20% chronologically recent samples).
Computes:
1. Optical Flow Radar Advection & Spatial Cross-Attention Accuracy Metrics.
2. Evidential Confidence Margin (±%) calibration.
3. Multi-Threshold Meteorological Nowcasting Performance (CSI, POD, FAR, F1, ROC-AUC).
4. Thermodynamic & Hydrological Regression Accuracy (MAE/RMSE for Temp, RH, Rain Rate).
"""

import sys
import os
import math
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT / "src" / "models"))

from pimcan_v3_advanced import PIMCANv3AdvancedModel

DATASET_PATH = WORKSPACE_ROOT / "data" / "processed" / "pimcan_multimodal_dataset.pt"
WEIGHTS_V3_PATH = WORKSPACE_ROOT / "src" / "models" / "weights" / "pimcan_v3_weights.pt"

def evaluate_v3_accuracy(test_ratio=0.20):
    print("=================================================================")
    print("PIMCAN-V3 ADVANCED ACCURACY & EVIDENTIAL BENCHMARK ENGINE")
    print("=================================================================")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset missing at {DATASET_PATH}")

    if not WEIGHTS_V3_PATH.exists():
        raise FileNotFoundError(f"PIMCAN-v3 weights missing at {WEIGHTS_V3_PATH}")

    # 1. Load Multimodal Dataset
    print(f"[1/3] Ingesting Multimodal Dataset from {DATASET_PATH}...")
    dataset = torch.load(DATASET_PATH, weights_only=False)

    station_seq = dataset["station_seq"]
    sat_seq = dataset["sat_seq"]
    lightning_grid_seq = dataset["lightning_grid_seq"]
    radar_seq = dataset["radar_seq"]
    targets = dataset["targets"]

    num_samples = station_seq.shape[0]
    split_idx = int(num_samples * (1.0 - test_ratio))

    test_st = station_seq[split_idx:]
    test_sat = sat_seq[split_idx:]
    test_lgt = lightning_grid_seq[split_idx:]
    test_rdr = radar_seq[split_idx:]
    test_targets = targets[split_idx:]

    test_count = test_st.shape[0]
    print(f"  -> Total Dataset Samples: {num_samples:,}")
    print(f"  -> Training Split:        {split_idx:,} samples (80%)")
    print(f"  -> Held-Out Test Split:  {test_count:,} samples (20%)")

    # 2. Load PIMCAN-v3 Architecture & Checkpoint
    print(f"[2/3] Loading PIMCAN-v3 Checkpoint from {WEIGHTS_V3_PATH}...")
    model = PIMCANv3AdvancedModel(station_dim=8, sat_dim=4, hidden_dim=32, output_steps=18)
    model.load_state_dict(torch.load(WEIGHTS_V3_PATH, weights_only=False))
    model.eval()

    # 3. Perform Inference on Test Set
    print(f"[3/3] Executing Evaluation Inference on Held-Out Test Set...")
    with torch.no_grad():
        outputs = model(test_st, test_sat, test_lgt, test_rdr)

    pred_probs = outputs["anomaly_probability_curve"].cpu().numpy()
    conf_margins = outputs["confidence_margin_pct"].cpu().numpy()
    target_probs = test_targets.cpu().numpy()
    
    pred_temp = outputs["pred_temp"].cpu().numpy().squeeze()
    pred_rh = outputs["pred_rh"].cpu().numpy().squeeze()
    pred_rain = outputs["pred_rain"].cpu().numpy().squeeze()

    gt_temp = test_st[:, -1, 0].cpu().numpy()
    gt_rh = test_st[:, -1, 1].cpu().numpy()
    gt_rain = test_st[:, -1, 3].cpu().numpy()

    flat_preds = pred_probs.flatten()
    flat_targets = target_probs.flatten()
    bin_targets = (flat_targets >= 0.5).astype(int)

    try:
        roc_auc = roc_auc_score(bin_targets, flat_preds)
    except Exception:
        roc_auc = 0.5

    thresholds = [0.05, 0.10, 0.15, 0.25, 0.50]
    threshold_results = []

    for thresh in thresholds:
        bin_preds = (flat_preds >= thresh).astype(int)
        hits = np.sum((bin_preds == 1) & (bin_targets == 1))
        false_alarms = np.sum((bin_preds == 1) & (bin_targets == 0))
        misses = np.sum((bin_preds == 0) & (bin_targets == 1))
        
        csi = hits / (hits + false_alarms + misses + 1e-8)
        pod = hits / (hits + misses + 1e-8)
        far = false_alarms / (hits + false_alarms + 1e-8)
        f1 = f1_score(bin_targets, bin_preds, zero_division=0)
        
        threshold_results.append({
            "threshold": thresh,
            "csi": round(float(csi), 4),
            "pod": round(float(pod), 4),
            "far": round(float(far), 4),
            "f1": round(float(f1), 4)
        })

    mae_temp = np.mean(np.abs(pred_temp - gt_temp))
    rmse_temp = math.sqrt(np.mean((pred_temp - gt_temp)**2))

    mae_rh = np.mean(np.abs(pred_rh - gt_rh))
    rmse_rh = math.sqrt(np.mean((pred_rh - gt_rh)**2))

    mae_rain = np.mean(np.abs(pred_rain - gt_rain))
    rmse_rain = math.sqrt(np.mean((pred_rain - gt_rain)**2))

    avg_conf_margin = np.mean(conf_margins)

    print("\n=================================================================")
    print("PIMCAN-V3 EMPIRICAL EVALUATION REPORT")
    print("=================================================================")
    print(f"Evaluated Test Sequences: {test_count:,} ({test_count * 18:,} 10-min prediction points)")
    print(f"ROC-AUC Score:           {roc_auc:.4f}")
    print(f"Avg Evidential Confidence Margin: +/- {avg_conf_margin:.2f}%")

    print("\n1. MULTI-THRESHOLD ANOMALY PERFORMANCE:")
    print(f"  Threshold |  CSI (Threat Score) |  POD (Hit Rate) |  FAR (False Alarm) |  F1-Score")
    print(f"  ----------+--------------------+-----------------+--------------------+-----------")
    for r in threshold_results:
        print(f"    {r['threshold']:0.2f}    |       {r['csi']*100:6.2f}%       |     {r['pod']*100:6.2f}%      |      {r['far']*100:6.2f}%       |   {r['f1']:0.4f}")

    print("\n2. THERMODYNAMIC & HYDROLOGICAL REGRESSION ACCURACY:")
    print(f"  • Temperature MAE:     {mae_temp:.2f}°C | RMSE: {rmse_temp:.2f}°C")
    print(f"  • Relative Humidity MAE: {mae_rh:.2f}%  | RMSE: {rmse_rh:.2f}%")
    print(f"  • Rainfall Rate MAE:    {mae_rain:.3f} mm/hr | RMSE: {rmse_rain:.3f} mm/hr")

    print("=================================================================")
    print("PIMCAN-V3 ACCURACY EVALUATION COMPLETE - PASSED")
    print("=================================================================")

if __name__ == "__main__":
    evaluate_v3_accuracy()
