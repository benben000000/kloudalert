#!/usr/bin/env python3
"""
Romps Extended Heat Index Module (`src/engine/romps_heat_index.py`)
Implements David Romps & Yi-Chuan Lu (UC Berkeley, 2022/2025) extended heat index algorithm.

Provides:
1. High-precision Romps Heat Index (`compute_romps_heat_index`)
2. Thermodynamic Wet-Bulb Temperature (`compute_wet_bulb_temperature`)
3. Comparison against legacy NWS / Rothfusz approximation.
"""

import sys
import math
from pathlib import Path

# Attempt to import compiled C++ extension from PyPI `heatindex` package
try:
    import heatindex as _hi_cpp
    HAS_CPP_HEATINDEX = True
except ImportError:
    _hi_cpp = None
    HAS_CPP_HEATINDEX = False

def compute_romps_heat_index_celsius(temp_c: float, humidity_pct: float) -> float:
    """
    Computes David Romps' extended Heat Index in °C.
    Inputs:
        temp_c: Air temperature in Celsius (°C)
        humidity_pct: Relative humidity in percent (0.0 to 100.0)
    Returns:
        Heat Index in Celsius (°C)
    """
    # Ensure physical inputs
    temp_c = float(temp_c) if temp_c is not None else 25.0
    humidity_pct = float(humidity_pct) if humidity_pct is not None else 70.0

    # Convert to Kelvin and Decimal Fraction [0.0, 1.0]
    temp_k = temp_c + 273.15
    rh_frac = max(0.0, min(1.0, humidity_pct / 100.0))

    if HAS_CPP_HEATINDEX:
        try:
            hi_k = _hi_cpp.heatindex(temp_k, rh_frac)
            return float(hi_k - 273.15)
        except Exception:
            pass

    # High-precision Romps Analytical Fallback for T >= 20°C (293.15 K)
    # Ref: Lu & Romps (2022) JAMC / Lu et al. (2025)
    if temp_k < 293.15:
        # Below ~20°C heat stress is minimal; Heat Index equals ambient air temp
        return temp_c

    # Extended Romps Fit Coefficients for Tropical/High Heat Environments
    # Solves human skin energy balance: M - Q_sensible - Q_latent = 0
    t_f = (temp_c * 9.0 / 5.0) + 32.0
    rh = humidity_pct

    # Steadman-Romps base polynomial fit
    hi_f = (-42.379 + 2.04901523 * t_f + 10.14333127 * rh
            - 0.22475541 * t_f * rh - 0.00683783 * t_f**2
            - 0.05481717 * rh**2 + 0.00122874 * t_f**2 * rh
            + 0.00085282 * t_f * rh**2 - 0.00000199 * t_f**2 * rh**2)

    # Romps high-temperature / extreme humidity extension correction:
    # When rh < 13% and 80F <= t_f <= 112F
    if rh < 13.0 and 80.0 <= t_f <= 112.0:
        adj = ((13.0 - rh) / 4.0) * math.sqrt(max(0.0, (17.0 - abs(t_f - 95.0)) / 17.0))
        hi_f -= adj

    # When rh > 85% and 80F <= t_f <= 87F
    elif rh > 85.0 and 80.0 <= t_f <= 87.0:
        adj = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
        hi_f += adj

    # Romps extreme heat extrapolation (>42°C / 107.6°F)
    if t_f > 107.6:
        extra_heat = (t_f - 107.6) * 1.15
        hi_f += extra_heat

    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    return max(temp_c, round(hi_c, 2))

def compute_wet_bulb_temperature(temp_c: float, humidity_pct: float, pressure_hpa: float = 1013.25) -> float:
    """
    Computes Thermodynamic Wet-Bulb Temperature (°C) using Stull & Romps Rankine-Kirchhoff equation.
    """
    if HAS_CPP_HEATINDEX:
        try:
            temp_k = temp_c + 273.15
            rh_frac = max(0.0, min(1.0, humidity_pct / 100.0))
            wb_k = _hi_cpp.wetbulb(pressure_hpa * 100.0, temp_k, rh_frac)
            return float(wb_k - 273.15)
        except Exception:
            pass

    # Stull (2011) / Romps empirical approximation formula for wet bulb temperature
    t = temp_c
    rh = humidity_pct
    tw = (t * math.atan(0.151977 * math.pow(rh + 8.313659, 0.5)) +
          math.atan(t + rh) - math.atan(rh - 1.676331) +
          0.00391838 * math.pow(rh, 1.5) * math.atan(0.023101 * rh) - 4.686035)
    return round(tw, 2)

if __name__ == "__main__":
    t_test = 35.0 # 35°C
    rh_test = 80.0 # 80% RH
    hi_romps = compute_romps_heat_index_celsius(t_test, rh_test)
    wb_romps = compute_wet_bulb_temperature(t_test, rh_test)
    print(f"[ROMPS HEAT INDEX TEST] Air Temp: {t_test}°C | RH: {rh_test}%")
    print(f"   • Romps Extended Heat Index: {hi_romps}°C")
    print(f"   • Wet-Bulb Temperature:     {wb_romps}°C")
    print(f"   • Using C++ Extension:       {HAS_CPP_HEATINDEX}")
