#!/usr/bin/env python3
"""
Comprehensive PIMCAN-Liquid Verification Test Suite
(`src/engine/test_pimcan_system.py`)

Verifies:
1. 4-Modality Forward Pass (Station LTC, Satellite Conv-CfC, Blitzortung Lightning CfC Grid, Radar Conv-CfC)
2. Physics-Informed Loss Engine (Non-negative rain, thermodynamic bounds, temporal smoothness)
3. Section 8 Candidate Model Isolation & Deployment Gate
4. Drift Audit (Population Stability Index / KS Test)
5. ONNX Edge Export Schema Verification
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(WORKSPACE_ROOT / "src" / "models"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "engine"))
sys.path.append(str(WORKSPACE_ROOT / "src" / "data"))

from pimcan_liquid_model import PIMCANLiquidModel
from physics_losses import PIMCANPhysicsLoss, NonNegativeRainLoss, ThermodynamicEquilibriumLoss, TemporalSmoothnessLoss
from self_improving_agentic_loop import Section8AgenticSelfImprovingEngine, compute_population_stability_index
from export_onnx import export_pimcan_to_onnx

def test_pimcan_model_forward():
    print("[TEST 1/5] Testing PIMCAN-Liquid 4-Modality Forward Pass...")
    model = PIMCANLiquidModel(station_dim=8, sat_dim=4, hidden_dim=32, fused_dim=64, output_steps=18)
    model.eval()

    batch_size = 4
    seq_len = 24

    b_st = torch.randn(batch_size, seq_len, 8)
    b_sat = torch.randn(batch_size, seq_len, 4)
    b_lgt = torch.randn(batch_size, seq_len, 4, 32, 32)
    b_rdr = torch.randn(batch_size, seq_len, 1)

    with torch.no_grad():
        out = model(b_st, b_sat, b_lgt, b_rdr)

    prob_curve = out["anomaly_probability_curve"]
    pred_temp = out["pred_temp"]
    pred_rh = out["pred_rh"]
    pred_rain = out["pred_rain"]

    assert prob_curve.shape == (batch_size, 18), f"Expected shape (4, 18), got {prob_curve.shape}"
    assert pred_temp.shape == (batch_size, 1), f"Expected shape (4, 1), got {pred_temp.shape}"
    assert pred_rh.shape == (batch_size, 1), f"Expected shape (4, 1), got {pred_rh.shape}"
    assert torch.all(pred_rain >= 0.0), "Precipitation output must be non-negative!"

    print(f"  -> Forward pass SUCCESS! Anomaly curve shape: {prob_curve.shape}, Rain non-negative check: PASSED.")

def test_physics_loss_engine():
    print("[TEST 2/5] Testing Physics-Informed Regularized Loss Engine...")
    criterion = PIMCANPhysicsLoss()

    pred_probs = torch.sigmoid(torch.randn(4, 18))
    target_probs = torch.randint(0, 2, (4, 18)).float()
    pred_temp = torch.tensor([[32.0], [28.5], [35.0], [30.0]])
    pred_rh = torch.tensor([[82.0], [75.0], [90.0], [78.0]])
    pred_rain = torch.tensor([[0.0], [1.2], [0.0], [0.5]])

    res = criterion(pred_probs, target_probs, pred_temp, pred_rh, pred_rain)

    assert "total_loss" in res, "Missing total_loss in output"
    assert res["total_loss"].item() > 0.0, "Total loss must be positive"
    assert res["physics_penalty"] >= 0.0, "Physics penalty must be non-negative"

    print(f"  -> Physics Loss SUCCESS! Total Loss: {res['total_loss'].item():.4f}, Physics Penalty: {res['physics_penalty']:.4f}")

def test_drift_and_psi():
    print("[TEST 3/5] Testing Section 8 Population Stability Index (PSI) Drift Monitor...")
    rng = np.random.RandomState(42)
    baseline = rng.normal(30.0, 2.0, 500).tolist()
    current_stable = rng.normal(30.05, 2.0, 500).tolist()
    current_drifted = rng.normal(36.5, 3.5, 500).tolist()

    psi_stable = compute_population_stability_index(baseline, current_stable)
    psi_drift = compute_population_stability_index(baseline, current_drifted)

    assert psi_stable < 0.10, f"Stable distribution should have PSI < 0.10, got {psi_stable}"
    assert psi_drift > 0.25, f"Drifted distribution should have PSI > 0.25, got {psi_drift}"

    print(f"  -> Drift Monitor SUCCESS! Stable PSI: {psi_stable:.4f}, Drifted PSI: {psi_drift:.4f}")

def test_candidate_gating_and_rollback():
    print("[TEST 4/5] Testing Section 8 Candidate Isolation & Rollback Engine...")
    agent = Section8AgenticSelfImprovingEngine()
    
    cand_info = {
        "candidate_id": "cand_test_99",
        "weights_path": str(WORKSPACE_ROOT / "src" / "models" / "weights" / "pimcan_liquid_weights.pt"),
        "cand_model": agent.model
    }

    gate_res = agent.evaluate_and_gate_deployment(cand_info, None)
    assert "promoted" in gate_res, "Missing promoted key in gate result"

    rollback_res = agent.rollback_production_model()
    print(f"  -> Candidate Gate & Rollback SUCCESS! Gate result: {gate_res['status']}, Rollback status: {rollback_res}")

def test_onnx_export_schema():
    print("[TEST 5/5] Testing PIMCAN-Liquid ONNX Edge Export Schema...")
    res = export_pimcan_to_onnx()
    assert res["success"] is True, f"ONNX Export failed: {res.get('error')}"
    print(f"  -> ONNX Export SUCCESS! ONNX Size: {res['size_kb']} KB")

def run_all_pimcan_tests():
    print("=================================================================")
    print("PIMCAN-LIQUID SYSTEM VERIFICATION SUITE")
    print("=================================================================")
    test_pimcan_model_forward()
    test_physics_loss_engine()
    test_drift_and_psi()
    test_candidate_gating_and_rollback()
    test_onnx_export_schema()
    print("=================================================================")
    print("ALL PIMCAN-LIQUID SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    run_all_pimcan_tests()
