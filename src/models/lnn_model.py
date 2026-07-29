#!/usr/bin/env python3
"""
Liquid Neural Network (LTC) Model & Training Pipeline
Implements Liquid Time-Constant (LTC) continuous-time neural ODE dynamics
for predicting 18-step (45-minute lookahead) weather anomaly probability curves.
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
DATA_PROCESSED_PATH = WORKSPACE_ROOT / "data" / "processed" / "preprocessed_dataset.json"
WEIGHTS_DIR = WORKSPACE_ROOT / "src" / "models" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

class LiquidCell(nn.Module):
    """
    Liquid Time-Constant (LTC) Neural Cell implementation.
    Updates hidden states via continuous-time ODE dynamic step:
    dx/dt = - (1 / tau + f(x, input)) * x + f(x, input) * A
    """
    def __init__(self, input_dim, hidden_dim):
        super(LiquidCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.w_input = nn.Linear(input_dim, hidden_dim)
        self.w_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.A = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x, h):
        # Activation f(x, input)
        gate = torch.tanh(self.w_input(x) + self.w_hidden(h))
        # ODE discretization step dt = 0.1
        dt = 0.1
        dh = - (1.0 / (torch.abs(self.tau) + 1e-5) + torch.abs(gate)) * h + gate * self.A
        new_h = h + dt * dh
        return new_h

class LiquidNeuralNetwork(nn.Module):
    """
    Full Liquid Neural Network architecture for weather anomaly forecasting.
    """
    def __init__(self, input_dim=8, hidden_dim=32, output_steps=18):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_steps = output_steps
        self.cell = LiquidCell(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_steps),
            nn.Sigmoid()
        )

    def forward(self, x_seq):
        # x_seq shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x_seq.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            h = self.cell(x_t, h)

        # Output 18-step probability curve
        prob_curve = self.head(h)
        return prob_curve

class LNNTrainer:
    def __init__(self, dataset_path=DATA_PROCESSED_PATH):
        self.dataset_path = Path(dataset_path)

    def train_model(self, epochs=15, batch_size=64, lr=1e-3):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {self.dataset_path}")

        print(f"Loading preprocessed dataset from {self.dataset_path}...")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        X = torch.tensor(dataset["X"], dtype=torch.float32)
        Y = torch.tensor(dataset["Y"], dtype=torch.float32)

        num_samples, seq_len, num_features = X.shape
        output_steps = Y.shape[1]

        print(f"Training LNN model on {num_samples} samples (seq_len={seq_len}, features={num_features}, output_steps={output_steps})")

        model = LiquidNeuralNetwork(input_dim=num_features, hidden_dim=32, output_steps=output_steps)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        model.train()
        train_losses = []

        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(num_samples)
            epoch_loss = 0.0
            batches = 0

            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X[indices], Y[indices]

                optimizer.zero_grad()
                pred_y = model(batch_x)
                loss = criterion(pred_y, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            avg_loss = epoch_loss / max(batches, 1)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch:02d}/{epochs:02d} - Training Loss (MSE): {avg_loss:.6f}")

        # Save model weights & metadata
        weights_path = WEIGHTS_DIR / "lnn_weather_model.pt"
        torch.save(model.state_dict(), weights_path)

        meta = {
            "input_dim": num_features,
            "hidden_dim": 32,
            "output_steps": output_steps,
            "final_loss": round(train_losses[-1], 6),
            "epochs": epochs,
            "feature_names": dataset["feature_names"],
            "weights_path": str(weights_path)
        }
        meta_path = WEIGHTS_DIR / "model_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"Model successfully saved to {weights_path}")
        return meta

if __name__ == "__main__":
    trainer = LNNTrainer()
    res = trainer.train_model(epochs=15)
    print(json.dumps(res, indent=2))
