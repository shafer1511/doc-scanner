from collections import deque
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR or grayscale image to 2D float64 grayscale.

    Args:
        image: Input array of shape (H, W) or (H, W, 3).
    Returns:
        Grayscale array of shape (H, W), dtype float64.
    """
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0].astype(np.float64)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float64)


def gaussian_smooth(
    gray: np.ndarray, sigma: float = 1.4, ksize: int | None = None
) -> np.ndarray:
    """Apply 2D Gaussian smoothing to a grayscale image.

    Args:
        gray: 2D array of shape (H, W).
        sigma: Standard deviation of the Gaussian kernel.
        ksize: Odd kernel size; derived from sigma if None.
    Returns:
        Smoothed array of shape (H, W), dtype float64.
    """
    if sigma <= 0:
        return gray.astype(np.float64)
    if ksize is None:
        k = int(2 * np.ceil(3.0 * sigma) + 1)
    else:
        k = int(ksize)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    return cv2.GaussianBlur(gray.astype(np.float64), (k, k), sigmaX=sigma, sigmaY=sigma)


def compute_gradients(
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel gradients, magnitude M=hypot(gx,gy), and direction atan2(gy,gx).

    Args:
        img: 2D grayscale array of shape (H, W).
    Returns:
        Tuple (gx, gy, magnitude, direction) each of shape (H, W), dtype float64.
    """
    src = img.astype(np.float64)
    gx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    return gx, gy, magnitude, direction


def non_maximum_suppression(
    magnitude: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Suppress non-maximum gradient magnitudes along 4 quantized directions (0, 45, 90, 135 deg).

    Args:
        magnitude: Gradient magnitude array of shape (H, W), dtype float64.
        direction: Gradient angle array in radians of shape (H, W), dtype float64.
    Returns:
        Thinned magnitude array of shape (H, W), dtype float64.
    """
    h, w = magnitude.shape
    if h < 3 or w < 3:
        return np.zeros_like(magnitude, dtype=np.float64)

    angle = (np.rad2deg(direction) % 180.0)
    suppressed = np.zeros_like(magnitude, dtype=np.float64)

    center = magnitude[1:-1, 1:-1]
    ang = angle[1:-1, 1:-1]

    m0 = ((ang < 22.5) | (ang >= 157.5))
    m45 = (ang >= 22.5) & (ang < 67.5)
    m90 = (ang >= 67.5) & (ang < 112.5)
    m135 = (ang >= 112.5) & (ang < 157.5)

    keep0 = m0 & (center >= magnitude[1:-1, 2:]) & (center >= magnitude[1:-1, :-2])
    keep45 = m45 & (center >= magnitude[2:, 2:]) & (center >= magnitude[:-2, :-2])
    keep90 = m90 & (center >= magnitude[2:, 1:-1]) & (center >= magnitude[:-2, 1:-1])
    keep135 = m135 & (center >= magnitude[2:, :-2]) & (center >= magnitude[:-2, 2:])

    keep = keep0 | keep45 | keep90 | keep135
    suppressed[1:-1, 1:-1] = np.where(keep, center, 0.0)
    return suppressed


def double_threshold(
    nms: np.ndarray,
    low_ratio: float = 0.05,
    high_ratio: float = 0.15,
    low_thresh: float | None = None,
    high_thresh: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify NMS magnitudes into strong and weak edge masks using two thresholds.

    Args:
        nms: Thinned gradient magnitude array of shape (H, W).
        low_ratio: Fraction of max(nms) for low threshold when low_thresh is None.
        high_ratio: Fraction of max(nms) for high threshold when high_thresh is None.
        low_thresh: Optional absolute low threshold.
        high_thresh: Optional absolute high threshold.
    Returns:
        Tuple (strong, weak, classified) where strong/weak are bool (H, W) and classified is uint8 (H, W).
    """
    max_val = float(np.max(nms)) if nms.size > 0 else 0.0
    if max_val <= 0.0:
        empty_b = np.zeros(nms.shape, dtype=bool)
        empty_u = np.zeros(nms.shape, dtype=np.uint8)
        return empty_b, empty_b, empty_u

    hi = float(high_thresh) if high_thresh is not None else max_val * float(high_ratio)
    lo = float(low_thresh) if low_thresh is not None else max_val * float(low_ratio)
    if lo > hi:
        lo = hi

    strong = nms >= hi
    weak = (nms >= lo) & (nms < hi)
    classified = np.zeros(nms.shape, dtype=np.uint8)
    classified[weak] = 75
    classified[strong] = 255
    return strong, weak, classified


def hysteresis_8conn(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Trace 8-connected weak edge pixels linked to strong edge pixels via BFS.

    Args:
        strong: Boolean array of strong edges, shape (H, W).
        weak: Boolean array of weak edges, shape (H, W).
    Returns:
        Binary edge map of shape (H, W), dtype uint8 with values {0, 255}.
    """
    h, w = strong.shape
    edges = np.zeros((h, w), dtype=bool)
    edges[strong] = True

    ys, xs = np.nonzero(strong)
    queue = deque(zip(ys.tolist(), xs.tolist()))
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and weak[ny, nx] and not edges[ny, nx]:
                edges[ny, nx] = True
                queue.append((ny, nx))

    return np.where(edges, np.uint8(255), np.uint8(0))


def canny_edge_detection(
    image: np.ndarray,
    sigma: float = 1.4,
    low_ratio: float = 0.05,
    high_ratio: float = 0.15,
    use_gaussian: bool = True,
    low_thresh: float | None = None,
    high_thresh: float | None = None,
) -> dict[str, np.ndarray]:
    """Run full custom Canny edge detection pipeline returning all intermediate arrays.

    Args:
        image: Input BGR or grayscale array of shape (H, W) or (H, W, 3).
        sigma: Gaussian smoothing sigma.
        low_ratio: Low threshold ratio relative to max NMS magnitude.
        high_ratio: High threshold ratio relative to max NMS magnitude.
        use_gaussian: Whether to apply Gaussian smoothing before Sobel gradients.
        low_thresh: Optional absolute low threshold.
        high_thresh: Optional absolute high threshold.
    Returns:
        Dictionary with keys gray, blurred, magnitude, direction, nms, strong, weak, classified, edges.
    """
    gray = to_grayscale(image)
    blurred = gaussian_smooth(gray, sigma=sigma) if use_gaussian else gray.copy()
    _, _, magnitude, direction = compute_gradients(blurred)
    nms = non_maximum_suppression(magnitude, direction)
    strong, weak, classified = double_threshold(
        nms,
        low_ratio=low_ratio,
        high_ratio=high_ratio,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
    )
    edges = hysteresis_8conn(strong, weak)
    return {
        "gray": gray,
        "blurred": blurred,
        "magnitude": magnitude,
        "direction": direction,
        "nms": nms,
        "strong": strong,
        "weak": weak,
        "classified": classified,
        "edges": edges,
    }
