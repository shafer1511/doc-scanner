"""Canny edge detector built from NumPy/SciPy primitives."""

from collections import deque

import numpy as np
from scipy.ndimage import convolve1d

_SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])


def gaussian_blur(image: np.ndarray, sigma: float = 1.0, kernel_size: int | None = None) -> np.ndarray:
    """Separable Gaussian blur with reflected borders; returns float64 of the same shape."""
    img = np.asarray(image, dtype=np.float64)
    if kernel_size is None:
        kernel_size = 2 * max(1, int(round(3.0 * sigma))) + 1
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return convolve1d(convolve1d(img, kernel, axis=0, mode="reflect"), kernel, axis=1, mode="reflect")


def compute_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradients; returns magnitude and direction (radians) as float64 arrays."""
    from scipy.signal import convolve2d

    img = np.asarray(image, dtype=np.float64)
    gx = convolve2d(img, _SOBEL_X, mode="same", boundary="symm")
    gy = convolve2d(img, _SOBEL_Y, mode="same", boundary="symm")
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    return magnitude, direction


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin ridges by keeping pixels that are maximal along the gradient direction."""
    mag = np.asarray(magnitude, dtype=np.float64)
    angle = np.mod(np.asarray(direction, dtype=np.float64), np.pi)
    padded = np.pad(mag, 1, mode="constant")

    center = padded[1:-1, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    up_right = padded[:-2, 2:]
    down_left = padded[2:, :-2]
    up_left = padded[:-2, :-2]
    down_right = padded[2:, 2:]

    horizontal = (angle < np.pi / 8) | (angle >= 7 * np.pi / 8)
    diagonal = (angle >= np.pi / 8) & (angle < 3 * np.pi / 8)
    vertical = (angle >= 3 * np.pi / 8) & (angle < 5 * np.pi / 8)
    anti_diagonal = (angle >= 5 * np.pi / 8) & (angle < 7 * np.pi / 8)

    keep = (
        (horizontal & (center >= left) & (center >= right))
        | (diagonal & (center >= up_right) & (center >= down_left))
        | (vertical & (center >= up) & (center >= down))
        | (anti_diagonal & (center >= up_left) & (center >= down_right))
    )
    return np.where(keep, center, 0.0)


def double_threshold(
    nms_magnitude: np.ndarray, low_threshold: float, high_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split NMS magnitudes into strong/weak masks plus a retained-magnitude map."""
    nms = np.asarray(nms_magnitude, dtype=np.float64)
    strong = nms >= high_threshold
    weak = (nms >= low_threshold) & ~strong
    threshold_map = np.where(strong | weak, nms, 0.0)
    return strong, weak, threshold_map


def hysteresis_tracking(strong_mask: np.ndarray, weak_mask: np.ndarray) -> np.ndarray:
    """8-connected flood fill from strong pixels through weak pixels; returns uint8 {0, 255}."""
    strong = np.asarray(strong_mask, dtype=bool)
    weak = np.asarray(weak_mask, dtype=bool)
    rows, cols = strong.shape
    queue = deque(map(tuple, np.argwhere(strong)))
    visited = strong.copy()

    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and weak[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    return (visited.astype(np.uint8)) * 255


def canny_edge_detector(
    image: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
    sigma: float = 1.0,
    apply_blur: bool = True,
) -> dict:
    """Run the full Canny pipeline and return every intermediate stage."""
    img = np.asarray(image, dtype=np.float64)
    blurred = gaussian_blur(img, sigma=sigma) if apply_blur else img
    grad_mag, grad_dir = compute_gradients(blurred)
    nms = non_maximum_suppression(grad_mag, grad_dir)
    strong, weak, _ = double_threshold(nms, low_threshold, high_threshold)
    edge_map = hysteresis_tracking(strong, weak)

    return {
        "blurred": blurred,
        "grad_mag": grad_mag,
        "grad_dir": grad_dir,
        "nms": nms,
        "strong": strong,
        "weak": weak,
        "edge_map": edge_map,
    }
