"""Binarization: global Otsu, local Sauvola, and ink-mask metrics."""

import numpy as np
from scipy.ndimage import uniform_filter


def otsu_threshold(image: np.ndarray) -> tuple[float, np.ndarray]:
    """Pick the global threshold maximizing between-class variance; 255 = paper, 0 = ink."""
    img = np.clip(np.asarray(image, dtype=np.float64), 0.0, 255.0)
    hist, _ = np.histogram(img, bins=256, range=(0.0, 255.0))
    p = hist / max(hist.sum(), 1)
    levels = np.arange(256, dtype=np.float64)
    omega0 = np.cumsum(p)
    omega1 = 1.0 - omega0
    mu_total = float((levels * p).sum())
    cum = np.cumsum(levels * p)
    mu0 = np.divide(cum, omega0, out=np.zeros_like(cum), where=omega0 > 0)
    mu1 = np.divide(mu_total - cum, omega1, out=np.zeros_like(cum), where=omega1 > 0)
    sigma_b = omega0 * omega1 * (mu0 - mu1) ** 2
    best = np.flatnonzero(sigma_b >= sigma_b.max())
    threshold = float(best[0] + best[-1]) / 2.0  # midpoint of the optimal plateau
    binary = np.where(img > threshold, 255, 0).astype(np.uint8)
    return threshold, binary


def sauvola_threshold(
    image: np.ndarray, window_size: int = 25, k: float = 0.2, r: float = 128.0
) -> tuple[np.ndarray, np.ndarray]:
    """Local mean/std threshold map (Sauvola); 255 = paper, 0 = ink."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    img = np.asarray(image, dtype=np.float64)
    size = (window_size, window_size) if img.ndim == 2 else window_size
    m = uniform_filter(img, size=size, mode="reflect")
    m2 = uniform_filter(img**2, size=size, mode="reflect")
    s = np.sqrt(np.maximum(m2 - m**2, 0.0))
    threshold_map = m * (1.0 + k * (s / r - 1.0))
    binary = np.where(img > threshold_map, 255, 0).astype(np.uint8)
    return threshold_map, binary


def compute_binarization_metrics(pred_ink_mask: np.ndarray, gt_ink_mask: np.ndarray) -> dict:
    """Precision/recall/F1 of predicted ink against ground-truth ink."""
    pred = np.asarray(pred_ink_mask).astype(bool)
    gt = np.asarray(gt_ink_mask).astype(bool)
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
