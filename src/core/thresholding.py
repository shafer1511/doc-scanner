import cv2
import numpy as np


def otsu_threshold(gray_img: np.ndarray) -> dict:
    """Compute global Otsu threshold maximizing between-class variance from 256-bin histogram.

    Formula: sigma_b^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2.

    Args:
        gray_img: 2D grayscale image of shape (H, W).
    Returns:
        Dict with keys threshold (int), binary (uint8 0/255), hist (int64 shape (256,)).
    """
    img_u8 = np.clip(np.rint(gray_img), 0, 255).astype(np.uint8)
    hist = np.bincount(img_u8.ravel(), minlength=256).astype(np.int64)
    total = int(img_u8.size)

    if total == 0 or int(np.max(img_u8)) == int(np.min(img_u8)):
        thresh = int(img_u8.flat[0]) if total > 0 else 0
        binary = np.full(img_u8.shape, 255, dtype=np.uint8)
        return {"threshold": thresh, "binary": binary, "hist": hist}

    prob = hist.astype(np.float64) / float(total)
    bins = np.arange(256, dtype=np.float64)

    w0 = np.cumsum(prob)
    w1 = 1.0 - w0
    cum_mean = np.cumsum(prob * bins)
    total_mean = cum_mean[-1]

    valid = (w0 > 1e-12) & (w1 > 1e-12)
    between_var = np.zeros(256, dtype=np.float64)
    mu0 = np.zeros(256, dtype=np.float64)
    mu1 = np.zeros(256, dtype=np.float64)
    mu0[valid] = cum_mean[valid] / w0[valid]
    mu1[valid] = (total_mean - cum_mean[valid]) / w1[valid]
    between_var[valid] = w0[valid] * w1[valid] * (mu0[valid] - mu1[valid]) ** 2

    best_t = int(np.argmax(between_var))
    binary = np.where(img_u8 > best_t, np.uint8(255), np.uint8(0))
    return {"threshold": best_t, "binary": binary, "hist": hist}


def sauvola_threshold(
    gray_img: np.ndarray, window_size: int = 25, k: float = 0.2, R: float = 128.0
) -> dict:
    """Compute local adaptive Sauvola threshold map from local mean and std.

    Formula: T(x, y) = m(x, y) * [1 + k * (s(x, y) / R - 1)].

    Args:
        gray_img: 2D grayscale image of shape (H, W).
        window_size: Odd local window size in pixels.
        k: Positive sensitivity parameter (typically 0.1 to 0.5).
        R: Dynamic range of standard deviation (128.0 for 8-bit images).
    Returns:
        Dict with keys threshold_map (float64 shape (H, W)) and binary (uint8 0/255).
    """
    ws = int(window_size)
    if ws % 2 == 0:
        ws += 1
    ws = max(3, ws)

    src = np.clip(gray_img.astype(np.float64), 0.0, 255.0)
    mean = cv2.boxFilter(src, ddepth=cv2.CV_64F, ksize=(ws, ws), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(src * src, ddepth=cv2.CV_64F, ksize=(ws, ws), borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(0.0, mean_sq - mean * mean)
    std = np.sqrt(variance)

    threshold_map = mean * (1.0 + float(k) * ((std / float(R)) - 1.0))
    binary = np.where(src > threshold_map, np.uint8(255), np.uint8(0))
    return {"threshold_map": threshold_map, "binary": binary}
