#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel Grid Coordinates → Logical Coordinates Converter (Affine Transform Based)

The Excel optimizer produces outfielder coordinates in the "Excel grid" system:
    - RF: (180–200, 80–110)
    - CF: (105–145, 60–90)
    - LF: (40–70, 80–110)

These must be converted into the universal logical coordinate system:
    - Logical coordinate bounds: X[-89.50, 52.66], Y[-43.85, -14.18]
    - Logical label anchor points:
        LF(-60, 20), CF(-20, 25), RF(20, 20)

Core principle:
    Use 3 Excel grid reference points and map them to 3 logical reference points
    to compute a unique affine transformation.
"""

import numpy as np
from typing import Tuple, Dict
from outfield_region import solve_affine_transform

# Excel grid reference centers for LF, CF, RF
EXCEL_GRID_REFERENCE = {
    "LF": (55.0, 95.0),    # Center of grid interval: (40+70)/2, (80+110)/2
    "CF": (125.0, 75.0),   # (105+145)/2, (60+90)/2
    "RF": (190.0, 95.0)    # (180+200)/2, (80+110)/2
}

# Logical coordinate reference points (image/label anchor positions)
LOGICAL_REFERENCE = {
    "LF": (-60.0, 20.0),
    "CF": (-20.0, 25.0),
    "RF": (20.0, 20.0)
}

# Logical coordinate boundaries (kept for documentation)
LOGICAL_X_MIN = -89.50
LOGICAL_X_MAX = 52.66
LOGICAL_Y_MIN = -43.85
LOGICAL_Y_MAX = -14.18


class ExcelGridToLogicalConverter:
    """
    Converts Excel grid coordinates (optimizer output) into logical coordinates
    using an affine transformation computed from 3 reference points.
    """

    def __init__(self):
        """Compute affine transform from Excel grid → logical coordinates."""
        
        # Three Excel grid reference centers
        excel_points = [
            EXCEL_GRID_REFERENCE["LF"],
            EXCEL_GRID_REFERENCE["CF"],
            EXCEL_GRID_REFERENCE["RF"]
        ]

        # Three logical reference coordinates
        logical_points = [
            LOGICAL_REFERENCE["LF"],
            LOGICAL_REFERENCE["CF"],
            LOGICAL_REFERENCE["RF"]
        ]

        # Compute affine transform such that:
        # logical = A @ excel + b
        self.A, self.b = solve_affine_transform(excel_points, logical_points)

        print("[ExcelGridToLogicalConverter] Affine transform initialized")
        print("  Matrix A:")
        print(self.A)
        print(f"  Translation vector b: {self.b}")

    def excel_to_logical(self, excel_x: float, excel_y: float) -> Tuple[float, float]:
        """
        Convert Excel grid coordinates (excel_x, excel_y) → logical coordinate space.

        Args:
            excel_x, excel_y: Coordinates in Excel grid space

        Returns:
            (logical_x, logical_y): Transformed coordinates
        """
        # Affine transform: logical = A @ excel + b
        excel_vec = np.array([excel_x, excel_y])
        logical_vec = self.A @ excel_vec + self.b

        logical_x = float(logical_vec[0])
        logical_y = float(logical_vec[1])

        # IMPORTANT:
        # We do NOT clamp logical coordinates.
        # Affine transforms can legitimately output values outside the logical bounds,
        # especially depending on optimizer positions. Clamping would distort results.
        
        return (logical_x, logical_y)


# -------------------------------------------------------
# Singleton instance
# -------------------------------------------------------

_converter_instance = None

def get_converter() -> ExcelGridToLogicalConverter:
    """Return global singleton converter instance."""
    global _converter_instance
    if _converter_instance is None:
        _converter_instance = ExcelGridToLogicalConverter()
    return _converter_instance


# -------------------------------------------------------
# Helper: Convert all optimizer positions
# -------------------------------------------------------

def convert_optimizer_positions_to_logical(
    positions: Dict[str, Tuple[float, float]]
) -> Dict[str, Tuple[float, float]]:
    """
    Convert optimizer output (Excel grid coordinates) to logical coordinates.

    Args:
        positions: e.g.
            {
                "LF": (excel_x, excel_y),
                "CF": (excel_x, excel_y),
                "RF": (excel_x, excel_y)
            }

    Returns:
        Dictionary with logical coordinate pairs:
            {
                "LF": (logical_x, logical_y),
                "CF": (logical_x, logical_y),
                "RF": (logical_x, logical_y)
            }
    """
    converter = get_converter()
    result = {}

    for name, (excel_x, excel_y) in positions.items():
        logical_x, logical_y = converter.excel_to_logical(excel_x, excel_y)
        result[name] = (logical_x, logical_y)

    return result
