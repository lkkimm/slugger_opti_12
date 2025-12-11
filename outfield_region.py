#!/usr/bin/env python3
"""
Outfield Region Polygon Management Module

Provides:
1. Affine transform calculation between pixel LF/CF/RF label positions and logical coordinates.
2. Conversion of outfield region boundary points between pixel and logical coordinates.
3. Point-in-polygon tests and projection onto polygon boundaries.
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import json

# Fixed logical coordinate locations for LF / CF / RF labels
LOGICAL_LABEL_POSITIONS = {
    "LF": (-60.0, 20.0),
    "CF": (-20.0, 25.0),
    "RF": (20.0, 20.0)
}


# -------------------------------------------------------
# Affine Transform Solver
# -------------------------------------------------------

def solve_affine_transform(
    pixel_points: List[Tuple[float, float]],
    logical_points: List[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute affine transform from 3 matched point pairs.

    Args:
        pixel_points: list of 3 pixel coordinate pairs
        logical_points: list of 3 logical coordinate pairs

    Returns:
        (A, b): A is 2x2 transform matrix, b is 2x1 translation vector
                logical = A @ pixel + b
    """
    if len(pixel_points) != 3 or len(logical_points) != 3:
        raise ValueError("Exactly 3 point pairs are required.")

    # Construct linear system for affine parameters
    A_matrix = np.zeros((6, 6))
    b_vector = np.zeros(6)

    # For each of the 3 point pairs:
    for i, ((px_x, px_y), (log_x, log_y)) in enumerate(zip(pixel_points, logical_points)):
        # x_log = a11*x_px + a12*y_px + b1
        A_matrix[i * 2, 0] = px_x
        A_matrix[i * 2, 1] = px_y
        A_matrix[i * 2, 2] = 1.0
        b_vector[i * 2] = log_x

        # y_log = a21*x_px + a22*y_px + b2
        A_matrix[i * 2 + 1, 3] = px_x
        A_matrix[i * 2 + 1, 4] = px_y
        A_matrix[i * 2 + 1, 5] = 1.0
        b_vector[i * 2 + 1] = log_y

    solution = np.linalg.solve(A_matrix, b_vector)

    A = np.array([[solution[0], solution[1]],
                  [solution[3], solution[4]]])
    b = np.array([solution[2], solution[5]])

    return A, b


# -------------------------------------------------------
# Transform Utilities
# -------------------------------------------------------

def pixel_to_logical(
    pixel_point: Tuple[float, float],
    A: np.ndarray,
    b: np.ndarray
) -> Tuple[float, float]:
    """
    Convert a pixel coordinate to logical coordinate using affine transform.

    Args:
        pixel_point: (x, y) pixel coordinate
        A: 2x2 affine matrix
        b: translation vector

    Returns:
        (logical_x, logical_y)
    """
    pixel_vec = np.array([pixel_point[0], pixel_point[1]])
    logical_vec = A @ pixel_vec + b
    return (float(logical_vec[0]), float(logical_vec[1]))


def logical_to_pixel(
    logical_point: Tuple[float, float],
    A: np.ndarray,
    b: np.ndarray
) -> Tuple[float, float]:
    """
    Convert logical coordinate to pixel coordinate (inverse transform).

    Formula:
        pixel = A_inv @ (logical - b)
    """
    logical_vec = np.array(logical_point)
    A_inv = np.linalg.inv(A)
    pixel_vec = A_inv @ (logical_vec - b)
    return (float(pixel_vec[0]), float(pixel_vec[1]))


# -------------------------------------------------------
# Polygon Algorithms
# -------------------------------------------------------

def point_in_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]]
) -> bool:
    """
    Ray-casting algorithm to determine if a point lies inside a polygon.

    Args:
        point: (x, y)
        polygon: list of vertices

    Returns:
        True if point is inside; False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def project_to_polygon_edge(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]]
) -> Tuple[float, float]:
    """
    Project a point onto the nearest edge of a polygon.

    Args:
        point: (x, y)
        polygon: list of polygon vertex coordinates

    Returns:
        (x, y) projected onto the closest polygon edge
    """
    x, y = point
    min_dist = float("inf")
    closest_point = point

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]

        proj = project_to_line_segment(point, p1, p2)
        dist = np.hypot(proj[0] - x, proj[1] - y)

        if dist < min_dist:
            min_dist = dist
            closest_point = proj

    return closest_point


def project_to_line_segment(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Project a point onto a line segment.

    Args:
        point: (x, y)
        line_start: segment start
        line_end: segment end

    Returns:
        Closest point on the segment.
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq == 0:
        return line_start

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return (proj_x, proj_y)


def clamp_to_outfield_region(
    point: Tuple[float, float],
    outfield_polygon: List[Tuple[float, float]]
) -> Tuple[float, float]:
    """
    Clamp a point into the outfield region. If it lies outside,
    project it onto the nearest polygon edge.
    """
    if point_in_polygon(point, outfield_polygon):
        return point
    return project_to_polygon_edge(point, outfield_polygon)


# -------------------------------------------------------
# Outfield Region Manager
# -------------------------------------------------------

class OutfieldRegionManager:
    """Manages outfield region geometry, polygon transforms, and label mappings."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: optional JSON configuration file
        """
        self.pixel_labels = {}
        self.A = None
        self.b = None

        self.outfield_polygon_pixel = []
        self.outfield_polygon_logical = []

        if config_path and Path(config_path).exists():
            self.load_config(config_path)

    def set_label_pixel_positions(
        self,
        lf_pixel: Tuple[float, float],
        cf_pixel: Tuple[float, float],
        rf_pixel: Tuple[float, float]
    ):
        """Define LF/CF/RF pixel label positions and compute affine transform."""
        self.pixel_labels = {
            "LF": lf_pixel,
            "CF": cf_pixel,
            "RF": rf_pixel
        }

        logical_points = [
            LOGICAL_LABEL_POSITIONS["LF"],
            LOGICAL_LABEL_POSITIONS["CF"],
            LOGICAL_LABEL_POSITIONS["RF"]
        ]

        pixel_points = [lf_pixel, cf_pixel, rf_pixel]

        self.A, self.b = solve_affine_transform(pixel_points, logical_points)

    def set_outfield_polygon_pixel(self, polygon_pixel: List[Tuple[float, float]]):
        """Store pixel coordinates of the outfield boundary polygon and compute logical version."""
        self.outfield_polygon_pixel = polygon_pixel

        if self.A is None or self.b is None:
            raise ValueError("Affine transform not set. Call set_label_pixel_positions() first.")

        self.outfield_polygon_logical = [
            pixel_to_logical(p, self.A, self.b)
            for p in polygon_pixel
        ]

    def logical_to_pixel(self, logical_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert logical → pixel coordinates.

        Note:
            No offset is applied here. The caller handles offsets if needed.

        Used for:
          - Outfielder position markers (caller applies offset)
          - Outfield boundary polygons (pixel coordinates already map cleanly)
        """
        if self.A is None or self.b is None:
            raise ValueError("Affine transform not set.")

        return logical_to_pixel(logical_point, self.A, self.b)

    def clamp_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Clamp a logical point to within the outfield region."""
        if not self.outfield_polygon_logical:
            return point
        return clamp_to_outfield_region(point, self.outfield_polygon_logical)

    def is_point_in_region(self, point: Tuple[float, float]) -> bool:
        """Return True if the logical point lies inside the outfield region."""
        if not self.outfield_polygon_logical:
            return True
        return point_in_polygon(point, self.outfield_polygon_logical)

    def save_config(self, config_path: str):
        """Save affine transform, labels, and polygon data to JSON."""
        config = {
            "pixel_labels": self.pixel_labels,
            "affine_transform": {
                "A": self.A.tolist() if self.A is not None else None,
                "b": self.b.tolist() if self.b is not None else None,
            },
            "outfield_polygon_pixel": self.outfield_polygon_pixel,
            "outfield_polygon_logical": self.outfield_polygon_logical
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load_config(self, config_path: str):
        """Load affine transform, labels, and polygon data from a JSON config file."""
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.pixel_labels = {k: tuple(v) for k, v in config.get("pixel_labels", {}).items()}

        affine = config.get("affine_transform", {})
        if affine.get("A"):
            self.A = np.array(affine["A"])
        if affine.get("b"):
            self.b = np.array(affine["b"])

        self.outfield_polygon_pixel = [
            tuple(p) for p in config.get("outfield_polygon_pixel", [])
        ]
        self.outfield_polygon_logical = [
            tuple(p) for p in config.get("outfield_polygon_logical", [])
        ]
