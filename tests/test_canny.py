"""Unit tests for Canny Edge Detector module (src/core/canny.py).
Tests pure functions, intermediate outputs, and edge cases specified in RULE.md.
"""
import numpy as np
import pytest

from src.core.canny import (
    gaussian_blur,
    compute_gradients,
    non_maximum_suppression,
    double_threshold,
    hysteresis_tracking,
    canny_edge_detector,
)


def test_constant_image():
    """Constant image must produce zero gradient and completely blank edge map."""
    for val in [0, 128, 255]:
        img = np.full((50, 50), val, dtype=np.uint8)
        res = canny_edge_detector(img, low_threshold=30.0, high_threshold=80.0, sigma=1.0)
        assert np.all(res["grad_mag"] == 0.0), f"Constant image {val} had non-zero gradients."
        assert np.all(res["edge_map"] == 0), f"Constant image {val} produced edges."


def test_step_edge_horizontal():
    """Horizontal step edge must produce thin horizontal edges."""
    img = np.zeros((40, 40), dtype=np.float64)
    img[20:, :] = 200.0  # Sharp step edge along row 20
    res = canny_edge_detector(img, low_threshold=40.0, high_threshold=100.0, sigma=1.0)
    edge_map = res["edge_map"]
    
    assert edge_map.shape == (40, 40)
    assert edge_map.dtype == np.uint8
    # Edge should exist around row 19-20 and not anywhere far from row 20
    assert np.any(edge_map[18:22, 5:35] == 255), "Step edge not detected."
    assert np.all(edge_map[:15, :] == 0), "Spurious edges found in top uniform region."
    assert np.all(edge_map[25:, :] == 0), "Spurious edges found in bottom uniform region."


def test_step_edge_vertical():
    """Vertical step edge must produce thin vertical edges."""
    img = np.zeros((40, 40), dtype=np.float64)
    img[:, 20:] = 200.0  # Sharp step edge along col 20
    res = canny_edge_detector(img, low_threshold=40.0, high_threshold=100.0, sigma=1.0)
    edge_map = res["edge_map"]
    
    assert edge_map.shape == (40, 40)
    assert np.any(edge_map[5:35, 18:22] == 255), "Vertical step edge not detected."
    assert np.all(edge_map[:, :15] == 0), "Spurious edges in left uniform region."
    assert np.all(edge_map[:, 25:] == 0), "Spurious edges in right uniform region."


def test_nms_thinning():
    """NMS must suppress adjacent non-maximal gradients to thin edges."""
    mag = np.zeros((7, 7), dtype=np.float64)
    # Horizontal edge profile across rows: row 2=50, row 3=100, row 4=50
    mag[2, :] = 50.0
    mag[3, :] = 100.0
    mag[4, :] = 50.0
    
    # Gradient direction is along Y axis (approx 90 deg or pi/2)
    direction = np.full((7, 7), np.pi / 2, dtype=np.float64)
    
    nms = non_maximum_suppression(mag, direction)
    assert np.all(nms[2, :] == 0.0), "Row 2 was not suppressed."
    assert np.all(nms[4, :] == 0.0), "Row 4 was not suppressed."
    assert np.all(nms[3, :] == 100.0), "Row 3 peak was incorrectly suppressed."


def test_hysteresis_connectivity():
    """Hysteresis must retain weak edges connected to strong edges, and discard isolated weak edges."""
    strong = np.zeros((10, 10), dtype=bool)
    weak = np.zeros((10, 10), dtype=bool)
    
    # Strong edge at (5, 5)
    strong[5, 5] = True
    # Weak edge connected at (5, 6) and (6, 7) (8-connected chain)
    weak[5, 6] = True
    weak[6, 7] = True
    
    # Isolated weak edge at (1, 1)
    weak[1, 1] = True
    
    edge_map = hysteresis_tracking(strong, weak)
    
    assert edge_map[5, 5] == 255
    assert edge_map[5, 6] == 255, "Connected weak pixel at (5, 6) should be preserved."
    assert edge_map[6, 7] == 255, "Chain weak pixel at (6, 7) should be preserved."
    assert edge_map[1, 1] == 0, "Isolated weak pixel at (1, 1) must be discarded."


def test_ablation_gaussian_blur_toggle():
    """apply_blur=False should bypass Gaussian blur for ablation study."""
    img = np.random.RandomState(42).randint(0, 255, size=(30, 30)).astype(np.uint8)
    res_blurred = canny_edge_detector(img, sigma=1.5, apply_blur=True)
    res_no_blur = canny_edge_detector(img, sigma=1.5, apply_blur=False)
    
    assert not np.array_equal(res_blurred["blurred"], res_no_blur["blurred"])
    assert np.array_equal(res_no_blur["blurred"], img.astype(np.float64))


def test_canny_intermediate_outputs_contract():
    """Check that all required intermediate arrays are present with correct shapes and dtypes."""
    img = np.zeros((32, 32), dtype=np.uint8)
    img[10:22, 10:22] = 255
    
    res = canny_edge_detector(img, low_threshold=50.0, high_threshold=150.0, sigma=1.0)
    required_keys = ["blurred", "grad_mag", "grad_dir", "nms", "strong", "weak", "edge_map"]
    for k in required_keys:
        assert k in res, f"Missing intermediate output key: {k}"
        assert res[k].shape == (32, 32), f"Key {k} shape mismatch: {res[k].shape}"
        
    assert res["edge_map"].dtype == np.uint8
    assert set(np.unique(res["edge_map"])).issubset({0, 255})
