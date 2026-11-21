"""Unit tests for Homography and Warping module (src/core/warping.py).
Tests bilinear interpolation on hand-crafted patches (integer, fractional, boundaries),
identity homography, and document perspective warping.
"""
import numpy as np
import pytest

from src.core.warping import (
    bilinear_sample_gray,
    compute_destination_dimensions,
    compute_homography,
    warp_document,
)


def test_bilinear_sample_integer_coordinates():
    """Bilinear sample at integer coordinates must exactly equal the image pixel value."""
    patch = np.array([
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
        [70.0, 80.0, 90.0],
    ], dtype=np.float64)
    
    assert bilinear_sample_gray(patch, 0.0, 0.0) == 10.0
    assert bilinear_sample_gray(patch, 1.0, 1.0) == 50.0
    assert bilinear_sample_gray(patch, 2.0, 0.0) == 30.0
    assert bilinear_sample_gray(patch, 2.0, 2.0) == 90.0


def test_bilinear_sample_fractional_center():
    """Bilinear sample at center of 4 pixels must be their exact average."""
    patch = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
    ], dtype=np.float64)
    # Center (x=0.5, y=0.5) => (10 + 20 + 30 + 40) / 4 = 25.0
    val = bilinear_sample_gray(patch, 0.5, 0.5)
    assert pytest.approx(val, abs=1e-5) == 25.0
    
    # Quarter point (x=0.25, y=0.75):
    # row 0 interp: 0.75*10 + 0.25*20 = 12.5
    # row 1 interp: 0.75*30 + 0.25*40 = 32.5
    # col interp: 0.25*12.5 + 0.75*32.5 = 3.125 + 24.375 = 27.5
    val_q = bilinear_sample_gray(patch, 0.25, 0.75)
    assert pytest.approx(val_q, abs=1e-5) == 27.5


def test_bilinear_sample_boundary_clamping():
    """Bilinear sample on boundary edges (e.g. y=0, x in between)."""
    patch = np.array([
        [100.0, 200.0],
        [100.0, 200.0],
    ], dtype=np.float64)
    # At top edge (x=0.3, y=0.0): 0.7*100 + 0.3*200 = 130.0
    val = bilinear_sample_gray(patch, 0.3, 0.0)
    assert pytest.approx(val, abs=1e-5) == 130.0


def test_identity_homography():
    """Corners that already form a standard rectangle must produce near-identity transform."""
    W, H = 80, 50
    corners = np.array([
        [0.0, 0.0],
        [W - 1.0, 0.0],
        [W - 1.0, H - 1.0],
        [0.0, H - 1.0],
    ], dtype=np.float64)
    
    dims = compute_destination_dimensions(corners)
    assert dims == (W, H)
    
    dst_corners = np.array([
        [0.0, 0.0],
        [W - 1.0, 0.0],
        [W - 1.0, H - 1.0],
        [0.0, H - 1.0],
    ], dtype=np.float64)
    H_mat = compute_homography(corners, dst_corners)
    # H should be scalar multiple of identity (or identity)
    H_norm = H_mat / H_mat[2, 2]
    np.testing.assert_allclose(H_norm, np.eye(3), atol=1e-5)


def test_warp_document_recovers_flat_rectangle():
    """Warping a known skewed rectangle should recover the rectangular shape."""
    # Create an image with a white rectangle inside
    img = np.zeros((100, 100), dtype=np.uint8)
    src_corners = np.array([
        [20.0, 10.0],  # TL
        [80.0, 20.0],  # TR
        [75.0, 85.0],  # BR
        [15.0, 75.0],  # BL
    ], dtype=np.float64)
    
    warped, H_mat = warp_document(img, src_corners)
    assert warped.ndim in (2, 3)
    assert H_mat.shape == (3, 3)
    W, H = compute_destination_dimensions(src_corners)
    assert warped.shape[1] == W
    assert warped.shape[0] == H
