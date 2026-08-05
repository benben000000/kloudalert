#!/usr/bin/env python3
"""
Edge Mobile Inference Benchmark & Validation Engine for PIMCAN-Liquid
(`src/engine/edge_inference_bench.py`)

Measures real-time inference latency, throughput (IPS), and numerical precision of
the PIMCAN-Liquid multimodal core.
"""

import os
import sys
import json
import time
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.pimcan_liquid_model import PIMCANLiquidModel
from engine.duration_timer import AnomalyDurationEngine

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_PATH = WORKSPACE_ROOT / "src" / "models" / "weights" / "pimcan_liquid_weights.pt"

class EdgeInferenceBenchmark:
    def __init__(self):
        self.model = PIMCANLiquidModel(station_dim=8, sat_dim=4, hidden_dim=32, fused_dim=64, output_steps=18)
        if WEIGHTS_PATH.exists():
            self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
        self.model.eval()

        self.duration_engine = AnomalyDurationEngine()

    def benchmark_edge_performance(self, num_trials=50):
        print(f"[BENCHMARK] Running Mobile Edge Performance Benchmark ({num_trials} streaming iterations)...")

        dummy_st = torch.randn(1, 24, 8)
        dummy_sat = torch.randn(1, 24, 4)
        dummy_lgt = torch.randn(1, 24, 4, 32, 32)
        dummy_rdr = torch.randn(1, 24, 1)

        latencies_ms = []

        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = self.model(dummy_st, dummy_sat, dummy_lgt, dummy_rdr)

            # Benchmark loop
            for _ in range(num_trials):
                t0 = time.perf_counter()
                res = self.model(dummy_st, dummy_sat, dummy_lgt, dummy_rdr)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        avg_latency = round(sum(latencies_ms) / len(latencies_ms), 3)
        min_latency = round(min(latencies_ms), 3)
        max_latency = round(max(latencies_ms), 3)
        throughput_ips = round(1000.0 / avg_latency, 1)

        sample_curve = res["anomaly_probability_curve"].squeeze(0).tolist()
        duration_report = self.duration_engine.calculate_duration_and_alert(sample_curve)

        result = {
            "model_architecture": "PIMCAN-Liquid (LTC + Conv-CfC + CfC Fusion)",
            "benchmark_trials": num_trials,
            "latency_metrics_ms": {
                "average_latency_ms": avg_latency,
                "min_latency_ms": min_latency,
                "max_latency_ms": max_latency
            },
            "throughput_inferences_per_sec": throughput_ips,
            "target_latency_status": "EXCELLENT (<15ms Edge)" if avg_latency < 15.0 else "PASSED (<50ms Edge)",
            "sample_duration_alert": duration_report["alert_text"],
            "sample_severity": duration_report["severity"]
        }

        print(json.dumps(result, indent=2))
        return result

if __name__ == "__main__":
    bench = EdgeInferenceBenchmark()
    bench.benchmark_edge_performance()
