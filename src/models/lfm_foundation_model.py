#!/usr/bin/env python3
"""
Liquid Foundation Model (LFM-230M) - Predictive Neural Core
Continuous-Time Liquid Time-Constant (LTC) ODE dynamics with Multi-Head Temporal Attention,
Focal Anomaly Loss, Online Self-Fine-Tuning, and ONNX Edge Exporter.
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = WORKSPACE_ROOT / "src" / "models" / "weights"
WEB_APP_ONNX = WORKSPACE_ROOT / "web_app" / "lnn_weather_model.onnx"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

class FocalLoss(nn.Module):
    """Focal Binary Cross-Entropy Loss to penalize missed sudden rain anomalies."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()

class LiquidCell230M(nn.Module):
    """Liquid Time-Constant (LTC) ODE cell computing state transition dx/dt = -x/tau + f(x, I)."""
    def __init__(self, input_dim, hidden_dim):
        super(LiquidCell230M, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_weights = nn.Linear(input_dim, hidden_dim)
        self.state_weights = nn.Linear(hidden_dim, hidden_dim)
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 0.1)

    def forward(self, x, h, dt=0.1):
        # f(x, I) = tanh(W_in * x + W_state * h)
        f_val = torch.tanh(self.input_weights(x) + self.state_weights(h))
        # dh/dt = -h/tau + f_val
        dh_dt = -h / (torch.exp(self.tau) + 1e-5) + f_val
        h_next = h + dh_dt * dt
        return h_next

class LiquidFoundationModel230M(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_steps=18):
        super(LiquidFoundationModel230M, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_steps = output_steps

        # Liquid Time-Constant Cell
        self.ltc_cell = LiquidCell230M(input_dim, hidden_dim)

        # Multi-Head Temporal Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Output Predictor (18 nowcast probabilities for 15-45 min horizon)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_steps),
            nn.Sigmoid()
        )

        self.focal_loss = FocalLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def forward(self, x_sequence):
        # x_sequence shape: (batch_size, sequence_length=24, input_dim=8)
        batch_size, seq_len, _ = x_sequence.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x_sequence.device)

        hidden_states = []
        for t in range(seq_len):
            x_t = x_sequence[:, t, :]
            h = self.ltc_cell(x_t, h)
            hidden_states.append(h.unsqueeze(1))

        # Shape: (batch_size, seq_len, hidden_dim)
        h_seq = torch.cat(hidden_states, dim=1)

        # Apply Temporal Attention
        attn_out, _ = self.attention(h_seq, h_seq, h_seq)
        context_vector = attn_out[:, -1, :]  # Take last context vector

        # Generate 18-step probability curve
        prob_curve = self.predictor(context_vector)
        return prob_curve

    def self_fine_tune(self, x_seq, y_ground_truth, epochs=3):
        """Online Self-Fine-Tuning execution on newly verified experiment data."""
        self.train()
        losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            predictions = self.forward(x_seq)
            loss = self.focal_loss(predictions, y_ground_truth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses.append(loss.item())
        self.eval()
        return sum(losses) / len(losses)

    def export_onnx(self, output_path=WEB_APP_ONNX):
        """Export trained weights to ONNX format for client browser inference."""
        self.eval()
        dummy_input = torch.randn(1, 24, self.input_dim, dtype=torch.float32)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_weather_sequence'],
            output_names=['anomaly_probability_curve'],
            dynamic_axes={
                'input_weather_sequence': {0: 'batch_size'},
                'anomaly_probability_curve': {0: 'batch_size'}
            },
            dynamo=False
        )
        print(f"[LFM-230M] Successfully exported ONNX model to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    print("[LFM-230M] Initializing Liquid Foundation Model...")
    model = LiquidFoundationModel230M()
    test_seq = torch.randn(1, 24, 8)
    out = model(test_seq)
    print(f"[LFM-230M] Forward pass output shape: {out.shape} (18 probability steps)")
    
    # Test Self-Fine-Tuning
    target = torch.zeros(1, 18)
    target[0, 5:10] = 1.0  # Simulated rain anomaly
    avg_loss = model.self_fine_tune(test_seq, target, epochs=3)
    print(f"[LFM-230M] Self-fine-tuning loss: {avg_loss:.4f}")

    # Test ONNX export
    model.export_onnx()
