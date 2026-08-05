#!/usr/bin/env python3
"""
PIMCAN-Liquid Core Architecture Implementation
(`src/models/pimcan_liquid_model.py`)

Physics-Informed Multimodal Continuous-time Anomaly Nowcasting Network with
Liquid Neural Encoders, Closed-form Continuous-Time (CfC) Fusion Core, Multi-Head Outputs,
and Physics Loss Regularization.

Modality Encoders:
1. Station Encoder: LTC continuous-time neural ODE encoder for station series.
2. Satellite Encoder: Lightweight CNN + CfC temporal block for Himawari-9 AHI tiles.
3. Radar Encoder: Conv-CfC continuous-time encoder for precipitation structure & storm motion.
4. Lightning Encoder: 2D temporal CfC grid encoder for Blitzortung flash-density fields.
5. Multimodal Fusion Core: CfC multimodal fusion block.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LTCCell(nn.Module):
    """Liquid Time-Constant (LTC) continuous-time ODE cell."""
    def __init__(self, input_dim, hidden_dim):
        super(LTCCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.w_in = nn.Linear(input_dim, hidden_dim)
        self.w_h = nn.Linear(hidden_dim, hidden_dim)
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 0.1)
        self.A = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x, h, dt=0.1):
        gate = torch.tanh(self.w_in(x) + self.w_h(h))
        dh = - (1.0 / (torch.exp(self.tau) + 1e-5) + torch.abs(gate)) * h + gate * self.A
        return torch.clamp(h + dt * dh, -50.0, 50.0)

class CfCCell(nn.Module):
    """Closed-form Continuous-Time (CfC) Neural Cell for fast closed-form ODE state update."""
    def __init__(self, input_dim, hidden_dim):
        super(CfCCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.ff1 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.ff2 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.time_a = nn.Linear(hidden_dim, hidden_dim)
        self.time_b = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h, t_step=0.1):
        cat_in = torch.cat([x, h], dim=-1)
        f_gate = torch.sigmoid(self.ff1(cat_in))
        g_val = torch.tanh(self.ff2(cat_in))
        
        # Closed-form continuous-time decay factor in (0, 1]
        decay_rate = - (torch.abs(self.time_a(h)) + 1e-3) * t_step
        t_factor = torch.exp(decay_rate)
        h_next = g_val * (1.0 - f_gate * t_factor) + h * (f_gate * t_factor)
        return torch.clamp(h_next, -10.0, 10.0)

class LTCStationEncoder(nn.Module):
    """Station Telemetry LTC Encoder for irregular ground station observations."""
    def __init__(self, input_dim=8, hidden_dim=32):
        super(LTCStationEncoder, self).__init__()
        self.ltc = LTCCell(input_dim, hidden_dim)

    def forward(self, station_seq):
        # station_seq: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = station_seq.shape
        h = torch.zeros(batch_size, self.ltc.hidden_dim, device=station_seq.device)
        for t in range(seq_len):
            h = self.ltc(station_seq[:, t, :], h)
        return h

class ConvCfCSatelliteEncoder(nn.Module):
    """Himawari-9 Satellite Spatial-Temporal Encoder (CNN + CfC Temporal Block)."""
    def __init__(self, sat_input_dim=4, hidden_dim=32):
        super(ConvCfCSatelliteEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(sat_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.ReLU()
        )
        self.cfc = CfCCell(hidden_dim, hidden_dim)

    def forward(self, sat_seq):
        # sat_seq: (batch_size, seq_len, sat_input_dim)
        batch_size, seq_len, _ = sat_seq.shape
        h = torch.zeros(batch_size, self.cfc.hidden_dim, device=sat_seq.device)
        for t in range(seq_len):
            feat = self.conv(sat_seq[:, t, :])
            h = self.cfc(feat, h)
        return h

class CfCLightningEncoder(nn.Module):
    """Blitzortung Lightning Grid Encoder (2D Spatial Flash Density + CfC Cell)."""
    def __init__(self, in_channels=4, hidden_dim=32):
        super(CfCLightningEncoder, self).__init__()
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, hidden_dim),
            nn.ReLU()
        )
        self.cfc = CfCCell(hidden_dim, hidden_dim)

    def forward(self, lightning_grid_seq):
        # lightning_grid_seq: (batch_size, seq_len, channels, H, W)
        batch_size, seq_len, C, H, W = lightning_grid_seq.shape
        h = torch.zeros(batch_size, self.cfc.hidden_dim, device=lightning_grid_seq.device)
        for t in range(seq_len):
            grid_t = lightning_grid_seq[:, t, :, :, :]
            spatial_feat = self.spatial_cnn(grid_t)
            h = self.cfc(spatial_feat, h)
        return h

class ConvCfCRadarEncoder(nn.Module):
    """Conv-CfC Encoder for 2D Radar Precipitation Reflectivity Fields & Storm Motion."""
    def __init__(self, in_channels=1, hidden_dim=32):
        super(ConvCfCRadarEncoder, self).__init__()
        self.radar_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, hidden_dim),
            nn.ReLU()
        )
        self.radar_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.ReLU()
        )
        self.cfc = CfCCell(hidden_dim, hidden_dim)

    def forward(self, radar_seq):
        # radar_seq can be (batch_size, seq_len, C, H, W) or (batch_size, seq_len, 1)
        if radar_seq.ndim == 5:
            batch_size, seq_len, C, H, W = radar_seq.shape
            h = torch.zeros(batch_size, self.cfc.hidden_dim, device=radar_seq.device)
            for t in range(seq_len):
                r_feat = self.radar_cnn(radar_seq[:, t, :, :, :])
                h = self.cfc(r_feat, h)
            return h
        else:
            batch_size, seq_len, _ = radar_seq.shape
            h = torch.zeros(batch_size, self.cfc.hidden_dim, device=radar_seq.device)
            for t in range(seq_len):
                r_feat = self.radar_fc(radar_seq[:, t, :])
                h = self.cfc(r_feat, h)
            return h

class CfCMultimodalFusion(nn.Module):
    """Closed-form Continuous-Time (CfC) Multimodal Fusion Core."""
    def __init__(self, hidden_dim=32, fused_dim=64):
        super(CfCMultimodalFusion, self).__init__()
        self.fusion_linear = nn.Linear(hidden_dim * 4, fused_dim)
        self.fusion_cfc = CfCCell(fused_dim, fused_dim)

    def forward(self, h_station, h_sat, h_lightning, h_radar):
        cat_latent = torch.cat([h_station, h_sat, h_lightning, h_radar], dim=-1)
        fused_init = torch.relu(self.fusion_linear(cat_latent))
        h_fused = torch.zeros_like(fused_init)
        h_fused = self.fusion_cfc(fused_init, h_fused, t_step=0.1)
        return h_fused

class PIMCANLiquidModel(nn.Module):
    """
    Complete PIMCAN-Liquid Model Architecture
    Fuses Station + Satellite + Blitzortung Lightning + Radar via CfC Continuous-Time Core.
    """
    def __init__(self, station_dim=8, sat_dim=4, hidden_dim=32, fused_dim=64, output_steps=18):
        super(PIMCANLiquidModel, self).__init__()
        self.station_encoder = LTCStationEncoder(input_dim=station_dim, hidden_dim=hidden_dim)
        self.satellite_encoder = ConvCfCSatelliteEncoder(sat_input_dim=sat_dim, hidden_dim=hidden_dim)
        self.lightning_encoder = CfCLightningEncoder(in_channels=4, hidden_dim=hidden_dim)
        self.radar_encoder = ConvCfCRadarEncoder(hidden_dim=hidden_dim)
        
        self.fusion_core = CfCMultimodalFusion(hidden_dim=hidden_dim, fused_dim=fused_dim)

        # Multi-Head Predictors
        self.anomaly_prob_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_steps),
            nn.Sigmoid()
        )
        
        self.temp_head = nn.Linear(fused_dim, 1)
        self.rh_head = nn.Linear(fused_dim, 1)
        self.rain_intensity_head = nn.Sequential(nn.Linear(fused_dim, 1), nn.ReLU()) # Non-negative constraint
        self.convective_score_head = nn.Sequential(nn.Linear(fused_dim, 1), nn.Sigmoid())

    def forward(self, station_seq, sat_seq, lightning_grid_seq, radar_seq):
        station_seq = torch.nan_to_num(station_seq, nan=0.0)
        sat_seq = torch.nan_to_num(sat_seq, nan=0.0)
        lightning_grid_seq = torch.nan_to_num(lightning_grid_seq, nan=0.0)
        radar_seq = torch.nan_to_num(radar_seq, nan=0.0)

        h_st = self.station_encoder(station_seq)
        h_sat = self.satellite_encoder(sat_seq)
        h_lgt = self.lightning_encoder(lightning_grid_seq)
        h_rdr = self.radar_encoder(radar_seq)

        fused_state = self.fusion_core(h_st, h_sat, h_lgt, h_rdr)

        prob_curve = self.anomaly_prob_head(fused_state)
        pred_temp = self.temp_head(fused_state) + 29.5 # Mean temp baseline offset
        pred_rh = torch.clamp(self.rh_head(fused_state) + 75.0, 0.0, 100.0)
        pred_rain = self.rain_intensity_head(fused_state)
        convective_score = self.convective_score_head(fused_state)

        return {
            "anomaly_probability_curve": prob_curve,
            "pred_temp": pred_temp,
            "pred_rh": pred_rh,
            "pred_rain": pred_rain,
            "convective_score": convective_score,
            "fused_state": fused_state
        }
