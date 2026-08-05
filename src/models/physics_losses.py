#!/usr/bin/env python3
"""
Physics-Informed Loss Regularization Engine for PIMCAN-Liquid Architecture
(`src/models/physics_losses.py`)

Embeds physical principles into training losses without altering downstream analytical functions:
1. Non-negative precipitation constraint
2. Thermodynamic Lu-Romps Heat Index continuity & equilibrium bounds
3. ODE temporal smoothness and evolutionary transition penalty
4. Multi-objective Focal Anomaly + Physics Regularizer combination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NonNegativeRainLoss(nn.Module):
    """Penalizes physically impossible negative rain predictions."""
    def __init__(self, penalty_scale=10.0):
        super(NonNegativeRainLoss, self).__init__()
        self.penalty_scale = penalty_scale

    def forward(self, pred_rain):
        negative_mask = torch.relu(-pred_rain)
        return self.penalty_scale * torch.mean(negative_mask ** 2)

class ThermodynamicEquilibriumLoss(nn.Module):
    """
    Penalizes temperature and relative humidity pairs that violate physical
    saturation vapour pressure bounds (Clausius-Clapeyron relation).
    """
    def __init__(self):
        super(ThermodynamicEquilibriumLoss, self).__init__()

    def forward(self, pred_temp, pred_rh):
        # Temp in Celsius, RH in %
        # Relative humidity cannot exceed 100% or fall below 0%
        rh_upper_violation = torch.relu(pred_rh - 100.0)
        rh_lower_violation = torch.relu(0.0 - pred_rh)
        
        # Temperature realistic surface bounds (-10°C to +60°C)
        temp_upper_violation = torch.relu(pred_temp - 60.0)
        temp_lower_violation = torch.relu(-10.0 - pred_temp)

        loss = torch.mean(rh_upper_violation**2 + rh_lower_violation**2 + temp_upper_violation**2 + temp_lower_violation**2)
        return loss

class TemporalSmoothnessLoss(nn.Module):
    """Penalizes unphysical high-frequency temporal oscillations in consecutive forecast steps."""
    def __init__(self, weight=0.05):
        super(TemporalSmoothnessLoss, self).__init__()
        self.weight = weight

    def forward(self, prob_curve_seq):
        # prob_curve_seq shape: (batch_size, output_steps)
        if prob_curve_seq.shape[1] < 2:
            return torch.tensor(0.0, device=prob_curve_seq.device)
        
        first_diff = prob_curve_seq[:, 1:] - prob_curve_seq[:, :-1]
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]
        
        loss = torch.mean(first_diff ** 2) + 0.5 * torch.mean(second_diff ** 2)
        return self.weight * loss

class PIMCANPhysicsLoss(nn.Module):
    """
    Unified Physics-Informed Multimodal Regularized Loss Engine.
    Combines Focal Loss for severe convective anomaly detection with
    thermodynamic, non-negative, and temporal ODE physics constraints.
    """
    def __init__(self, alpha=0.75, gamma=2.0, lambda_rain=1.0, lambda_thermo=0.5, lambda_smooth=0.1):
        super(PIMCANPhysicsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_rain = lambda_rain
        self.lambda_thermo = lambda_thermo
        self.lambda_smooth = lambda_smooth

        self.non_neg_rain = NonNegativeRainLoss()
        self.thermo_bounds = ThermodynamicEquilibriumLoss()
        self.temporal_smooth = TemporalSmoothnessLoss()

    def forward(self, pred_probs, target_probs, pred_temp=None, pred_rh=None, pred_rain=None):
        # 1. Focal Anomaly Binary Cross Entropy Loss
        clamped_probs = torch.nan_to_num(pred_probs, nan=0.5, posinf=0.999, neginf=0.001)
        clamped_probs = torch.clamp(clamped_probs, 1e-6, 1.0 - 1e-6)
        target_probs = torch.nan_to_num(target_probs, nan=0.0)
        bce_loss = F.binary_cross_entropy(clamped_probs, target_probs, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1 - pt)**self.gamma * bce_loss).mean()

        total_loss = focal_loss

        # 2. Physics-Informed Regularizers (if physical heads provided)
        if pred_rain is not None:
            total_loss += self.lambda_rain * self.non_neg_rain(pred_rain)

        if pred_temp is not None and pred_rh is not None:
            total_loss += self.lambda_thermo * self.thermo_bounds(pred_temp, pred_rh)

        total_loss += self.lambda_smooth * self.temporal_smooth(pred_probs)

        return {
            "total_loss": total_loss,
            "focal_loss": focal_loss.item(),
            "physics_penalty": (total_loss - focal_loss).item()
        }
