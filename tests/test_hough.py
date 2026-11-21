"""Unit tests for Hough Transform module (src/core/hough.py).
Tests accumulator generation, peak detection, line suppression, edge cases (empty edge map,
pure vertical/horizontal lines, duplicate suppression ablation).
"""
import numpy as np
import pytest

from src.core.hough import (
    hough_accumulator,
    find_hough_peaks,
    suppress_similar_lines,
    hough_line_to_points,
    detect_hough_lines,
)


def test_empty_edge_map():
    """All-zero edge map must produce an all-zero accumulator and empty detected lines."""
    edge_map = np.zeros((60, 60), dtype=np.uint8)
    res = detect_hough_lines(edge_map, threshold=10)
    
    assert np.all(res["accumulator"] == 0), "Accumulator should be empty for blank edge map."
    assert len(res["raw_peaks"]) == 0, "No peaks should be found in blank image."
    assert len(res["filtered_lines"]) == 0, "No lines should be detected in blank image."


def test_horizontal_line_detection():
    """Horizontal line (y = const) must be detected with high confidence."""
    H, W = 80, 80
    edge_map = np.zeros((H, W), dtype=np.uint8)
    row_y = 35
    edge_map[row_y, 10:70] = 255  # Horizontal segment of length 60
    
    res = detect_hough_lines(
        edge_map,
        rho_res=1.0,
        theta_res=np.pi / 180.0,
        threshold=30,
        suppress_duplicates=True,
    )
    
    assert len(res["filtered_lines"]) >= 1, "Failed to detect horizontal line."
    best_rho, best_theta, votes = res["filtered_lines"][0]
    
    # For horizontal line y = row_y:
    # rho = x * cos(theta) + y * sin(theta)
    # When theta = pi/2: rho = y * 1 = 35
    # When theta = -pi/2: rho = -35
    # Either theta is approx pi/2 or -pi/2, and |rho| is approx 35.
    sin_val = np.sin(best_theta)
    cos_val = np.cos(best_theta)
    assert abs(cos_val) < 0.1, f"Expected theta ~ +/-90 deg, got theta={best_theta}"
    est_y = best_rho / sin_val
    assert abs(est_y - row_y) <= 2.0, f"Detected line y={est_y} deviates from true y={row_y}"


def test_vertical_line_detection():
    """Vertical line (x = const) must be detected with theta ~ 0 deg."""
    H, W = 80, 80
    edge_map = np.zeros((H, W), dtype=np.uint8)
    col_x = 42
    edge_map[10:70, col_x] = 255  # Vertical segment of length 60
    
    res = detect_hough_lines(
        edge_map,
        rho_res=1.0,
        theta_res=np.pi / 180.0,
        threshold=30,
        suppress_duplicates=True,
    )
    
    assert len(res["filtered_lines"]) >= 1, "Failed to detect vertical line."
    best_rho, best_theta, votes = res["filtered_lines"][0]
    
    # For vertical line x = col_x:
    # theta should be ~ 0 (or pi), rho ~ col_x
    sin_val = np.sin(best_theta)
    cos_val = np.cos(best_theta)
    assert abs(sin_val) < 0.1, f"Expected theta ~ 0 deg, got theta={best_theta}"
    est_x = best_rho / cos_val
    assert abs(est_x - col_x) <= 2.0, f"Detected line x={est_x} deviates from true x={col_x}"


def test_suppress_similar_lines():
    """suppress_similar_lines must eliminate nearby duplicate peaks."""
    # Synthetic candidates: two very close lines and one distant line
    lines = [
        (50.0, np.radians(45.0), 100),  # Top line 1
        (52.0, np.radians(46.0), 95),   # Duplicate of line 1
        (51.0, np.radians(44.0), 90),   # Duplicate of line 1
        (120.0, np.radians(85.0), 80),  # Distinct line 2
    ]
    
    filtered = suppress_similar_lines(
        lines,
        min_rho_diff=10.0,
        min_theta_diff=np.radians(5.0),
    )
    
    assert len(filtered) == 2, f"Expected 2 unique lines, got {len(filtered)}"
    assert filtered[0] == lines[0]
    assert filtered[1] == lines[3]


def test_ablation_suppression_toggle():
    """Ablation flag suppress_duplicates=False must preserve all raw peaks up to top_k."""
    H, W = 60, 60
    edge_map = np.zeros((H, W), dtype=np.uint8)
    # Thick line (2 parallel lines close to each other)
    edge_map[30, 10:50] = 255
    edge_map[31, 10:50] = 255
    
    res_suppressed = detect_hough_lines(
        edge_map, threshold=25, suppress_duplicates=True, top_k=10
    )
    res_unsuppressed = detect_hough_lines(
        edge_map, threshold=25, suppress_duplicates=False, top_k=10
    )
    
    assert len(res_unsuppressed["filtered_lines"]) >= len(res_suppressed["filtered_lines"])


def test_hough_line_to_points_clipping():
    """hough_line_to_points must return two distinct points on the image boundary."""
    # Line passing through center of 100x100 image horizontally: rho=50, theta=pi/2
    pts = hough_line_to_points(50.0, np.pi / 2, width=100, height=100)
    assert pts is not None
    (x1, y1), (x2, y2) = pts
    assert y1 == 50 and y2 == 50
    assert {x1, x2} == {0, 99} or (abs(x1 - x2) >= 98)
