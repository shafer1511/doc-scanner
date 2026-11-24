import numpy as np
from src.core.quadrilateral import (
    find_document_quad,
    intersect_lines,
    is_convex_quad,
    order_corners_perimeter,
    polygon_area,
)


def test_order_corners_and_area():
    pts = np.array([[200, 300], [50, 40], [50, 300], [200, 40]], dtype=np.float32)
    ordered = order_corners_perimeter(pts)
    expected = np.array([[50, 40], [200, 40], [200, 300], [50, 300]], dtype=np.float32)
    assert np.allclose(ordered, expected)
    assert is_convex_quad(ordered)
    assert abs(polygon_area(ordered) - (150.0 * 260.0)) < 1e-4


def test_degenerate_collinear_quad_rejected():
    collinear = np.array([[10, 10], [50, 10], [90, 10], [50, 80]], dtype=np.float32)
    assert not is_convex_quad(collinear)


def test_intersect_parallel_returns_none():
    l1 = {"rho": 20.0, "theta": 0.0}
    l2 = {"rho": 80.0, "theta": 0.01}
    assert intersect_lines(l1, l2) is None


def test_find_document_quad_accepts_synthetic_rect():
    h, w = 200, 240
    edges = np.zeros((h, w), dtype=np.uint8)
    edges[30:171, 40] = 255
    edges[30:171, 200] = 255
    edges[30, 40:201] = 255
    edges[170, 40:201] = 255

    lines = [
        {"rho": 40.0, "theta": 0.0, "votes": 141},
        {"rho": 200.0, "theta": 0.0, "votes": 141},
        {"rho": 30.0, "theta": np.deg2rad(90.0), "votes": 161},
        {"rho": 170.0, "theta": np.deg2rad(90.0), "votes": 161},
    ]
    res = find_document_quad(lines, edges, (h, w))
    assert res["status"] == "accepted"
    assert res["edge_support"] > 0.9
    expected = np.array([[40, 30], [200, 30], [200, 170], [40, 170]], dtype=np.float32)
    assert np.allclose(res["corners"], expected, atol=1.0)


def test_find_document_quad_rejects_insufficient_lines():
    edges = np.zeros((100, 100), dtype=np.uint8)
    res = find_document_quad([{"rho": 10.0, "theta": 0.0, "votes": 50}], edges, (100, 100))
    assert res["status"] == "rejected"
    assert res["reject_reason"] == "insufficient_hough_lines"
