"""Unit tests for Binarization module (src/core/thresholding.py).
Tests Otsu threshold selection, Sauvola local adaptive thresholding, ink evaluation metrics,
and edge cases (uniform image, shadowed synthetic gradient).
"""
import numpy as np
import pytest

from src.core.thresholding import (
    otsu_threshold,
    sauvola_threshold,
    compute_binarization_metrics,
)


def test_otsu_bimodal_distribution():
    """Otsu on a clear bimodal image (peaks at 40 and 200) must find threshold between 40 and 200."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:50, :] = 40   # Foreground / ink cluster
    img[50:, :] = 200  # Background cluster
    
    threshold, binary = otsu_threshold(img)
    
    # Between-class variance maximum should lie comfortably between the two peaks
    assert 40 < threshold < 200
    assert 90 <= threshold <= 150, f"Expected threshold around ~120, got {threshold}"
    # Pixels > threshold should be 255 (background), <= threshold should be 0 (ink)
    assert np.all(binary[50:, :] == 255)
    assert np.all(binary[:50, :] == 0)


def test_otsu_constant_image():
    """Constant image has zero variance; Otsu must return without exception."""
    img = np.full((40, 40), 128, dtype=np.uint8)
    threshold, binary = otsu_threshold(img)
    assert isinstance(threshold, (int, float))
    assert binary.shape == (40, 40)


def test_sauvola_shadow_adaptation():
    """Sauvola must adapt locally to uneven illumination gradient where global threshold fails."""
    H, W = 100, 100
    # Illumination ramp from left (80 = dark shadow) to right (230 = bright light)
    ramp = np.linspace(80, 230, W, dtype=np.float64)
    background = np.tile(ramp, (H, 1))
    
    img = background.copy()
    # Draw ink text lines (contrast drop of 50 relative to local background)
    # Left ink line: background ~100, ink ~50
    img[30:70, 15:20] = 50.0
    # Right ink line: background ~210, ink ~160
    img[30:70, 80:85] = 160.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    thresh_map, binary = sauvola_threshold(img, window_size=25, k=0.2, r=128.0)
    
    assert thresh_map.shape == (H, W)
    # In both left and right regions, ink pixels (values 50 and 160) must be detected as 0 (ink)
    assert np.all(binary[32:68, 16:19] == 0), "Sauvola missed dark ink under shadow on the left."
    assert np.all(binary[32:68, 81:84] == 0), "Sauvola missed ink under bright light on the right."
    # Background near ink lines should be 255
    assert np.all(binary[10:20, 15:20] == 255), "Background misclassified as ink."
    assert np.all(binary[10:20, 80:85] == 255), "Background misclassified as ink."


def test_compute_binarization_metrics():
    """Verify Precision, Recall, and F1 calculations on known binary masks."""
    # 4 pixels: GT ink is at [0, 1]
    gt_ink = np.array([True, True, False, False])
    # Pred ink is at [1, 2] -> TP=1 (idx 1), FP=1 (idx 2), FN=1 (idx 0)
    pred_ink = np.array([False, True, True, False])
    
    metrics = compute_binarization_metrics(pred_ink, gt_ink)
    
    # Precision = TP / (TP + FP) = 1 / 2 = 0.5
    # Recall = TP / (TP + FN) = 1 / 2 = 0.5
    # F1 = 2 * P * R / (P + R) = 0.5
    assert pytest.approx(metrics["precision"], abs=1e-4) == 0.5
    assert pytest.approx(metrics["recall"], abs=1e-4) == 0.5
    assert pytest.approx(metrics["f1"], abs=1e-4) == 0.5


def test_sauvola_parameter_sensitivity():
    """Higher k makes threshold more conservative (lower threshold, fewer pixels detected as ink)."""
    img = np.random.RandomState(42).randint(100, 200, size=(50, 50)).astype(np.uint8)
    _, binary_k02 = sauvola_threshold(img, window_size=25, k=0.2)
    _, binary_k05 = sauvola_threshold(img, window_size=25, k=0.5)
    
    # ink is binary == 0
    ink_count_k02 = np.sum(binary_k02 == 0)
    ink_count_k05 = np.sum(binary_k05 == 0)
    assert ink_count_k05 <= ink_count_k02
