#!/usr/bin/env python3
"""
DEPRECATED: `lnn_model.py` has been unified into `pimcan_liquid_model.py`.
This module redirects imports to `PIMCANLiquidModel` in accordance with the PIMCAN-Liquid specification.
"""

from pimcan_liquid_model import PIMCANLiquidModel as LiquidNeuralNetwork
from pimcan_liquid_model import PIMCANLiquidModel

__all__ = ["LiquidNeuralNetwork", "PIMCANLiquidModel"]
