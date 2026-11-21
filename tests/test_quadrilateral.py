"""Unit tests for Quadrilateral Selection module (src/core/quadrilateral.py).
Tests line intersections, convexity check, corner ordering, edge support, rejection cases
(insufficient lines, collinear vertices, self-intersecting quads), and ablation toggle.
"""
import numpy as np
import pytest

from src.core.quadrilateral import (
    intersect_lines,
    order_corners_perimeter,
    is_convex_quadrilateral,
    polygon_area,
    compute_edge_support,
    score_quadrilateral,
    find_document_quadrilateral,
)


def test_intersect_orthogonal_lines():
    """Intersecting x = 10 (rho=10, theta=0) and y = 20 (rho=20, theta=pi/2) should give (10, 20)."""
    line_v = (10.0, 0.0)
    line_h = (20.0, np.pi / 2)
    pt = intersect_lines(line_v, line_h)
    assert pt is not None
    x, y = pt
    assert pytest.approx(x, abs=1e-3) == 10.0
    assert pytest.approx(y, abs=1e-3) == 20.0


def test_intersect_parallel_lines():
    """Parallel lines should have no intersection (return None)."""
    line1 = (10.0, 0.0)
    line2 = (20.0, 0.0)
    assert intersect_lines(line1, line2) is None


def test_order_corners_perimeter():
    """Shuffled corners of a rectangle must be ordered in consistent clockwise perimeter order."""
    # Rectangle: TL=(10, 10), TR=(80, 10), BR=(80, 60), BL=(10, 60)
    shuffled = np.array([[80.0, 60.0], [10.0, 10.0], [10.0, 60.0], [80.0, 10.0]])
    ordered = order_corners_perimeter(shuffled)
    
    assert ordered.shape == (4, 2)
    # First corner should be top-left (10, 10)
    np.testing.assert_allclose(ordered[0], [10.0, 10.0])
    # Clockwise order: (10,10) -> (80,10) -> (80,60) -> (10,60)
    np.testing.assert_allclose(ordered[1], [80.0, 10.0])
    np.testing.assert_allclose(ordered[2], [80.0, 60.0])
    np.testing.assert_allclose(ordered[3], [10.0, 60.0])


def test_is_convex_quadrilateral_valid_and_invalid():
    """Test convex vs collinear / non-convex / self-intersecting quads."""
    # Valid convex rectangle
    rect = np.array([[10.0, 10.0], [80.0, 10.0], [80.0, 60.0], [10.0, 60.0]])
    assert is_convex_quadrilateral(rect) is True
    
    # Degenerate: 3 collinear points (10,10), (40,10), (80,10)
    collinear = np.array([[10.0, 10.0], [40.0, 10.0], [80.0, 10.0], [10.0, 60.0]])
    assert is_convex_quadrilateral(collinear) is False
    
    # Non-convex ("arrowhead" / dart shape)
    concave = np.array([[0.0, 0.0], [100.0, 0.0], [50.0, 20.0], [0.0, 100.0]])
    assert is_convex_quadrilateral(concave) is False
    
    # Self-intersecting "hourglass" shape
    bowtie = np.array([[0.0, 0.0], [100.0, 100.0], [100.0, 0.0], [0.0, 100.0]])
    assert is_convex_quadrilateral(bowtie) is False


def test_rejection_insufficient_lines():
    """Fewer than 4 lines must return status rejected with reason 'insufficient_lines'."""
    lines = [(10.0, 0.0, 100), (20.0, np.pi / 2, 90)]
    edge_map = np.zeros((100, 100), dtype=np.uint8)
    res = find_document_quadrilateral(lines, edge_map, image_shape=(100, 100))
    
    assert res["status"] == "rejected"
    assert res["reject_reason"] == "insufficient_lines"
    assert res["corners"] is None


def test_find_document_quadrilateral_synthetic():
    """Full detection on synthetic 4 lines and rectangle edge map."""
    H, W = 100, 100
    edge_map = np.zeros((H, W), dtype=np.uint8)
    # Draw rectangle: x from 20 to 80, y from 15 to 75
    edge_map[15, 20:81] = 255
    edge_map[75, 20:81] = 255
    edge_map[15:76, 20] = 255
    edge_map[15:76, 80] = 255
    
    # 4 perfect lines:
    # Top: y = 15 => rho = 15, theta = pi/2
    # Bottom: y = 75 => rho = 75, theta = pi/2
    # Left: x = 20 => rho = 20, theta = 0
    # Right: x = 80 => rho = 80, theta = 0
    lines = [
        (15.0, np.pi / 2, 60),
        (75.0, np.pi / 2, 60),
        (20.0, 0.0, 60),
        (80.0, 0.0, 60),
    ]
    
    res = find_document_quadrilateral(lines, edge_map, image_shape=(H, W), use_edge_support=True)
    assert res["status"] == "accepted"
    assert res["reject_reason"] is None
    corners = res["corners"]
    assert corners is not None and corners.shape == (4, 2)
    # Corners should be (20, 15), (80, 15), (80, 75), (20, 75)
    np.testing.assert_allclose(corners[0], [20.0, 15.0], atol=1.0)
    np.testing.assert_allclose(corners[1], [80.0, 15.0], atol=1.0)
    np.testing.assert_allclose(corners[2], [80.0, 75.0], atol=1.0)
    np.testing.assert_allclose(corners[3], [20.0, 75.0], atol=1.0)


def test_ablation_edge_support_toggle():
    """Ablation toggle use_edge_support=False should score candidates purely by area."""
    corners = np.array([[20.0, 20.0], [60.0, 20.0], [60.0, 60.0], [20.0, 60.0]])
    edge_map = np.zeros((100, 100), dtype=np.uint8)  # Empty edge map -> 0 support
    
    score_with_support = score_quadrilateral(corners, edge_map, (100, 100), use_edge_support=True, alpha=0.5)
    score_without_support = score_quadrilateral(corners, edge_map, (100, 100), use_edge_support=False)
    
    area_ratio = 1600.0 / 10000.0
    assert pytest.approx(score_without_support, abs=1e-4) == area_ratio
    assert score_with_support < score_without_support
