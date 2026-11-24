import cv2
import numpy as np


def bilinear_sample_gray(
    gray_img: np.ndarray, x: float | np.ndarray, y: float | np.ndarray
) -> float | np.ndarray:
    """Perform bilinear interpolation on a 2D grayscale image with border clamping.

    Formula: f(x,y) = (1-a)(1-b)I(x0,y0) + a(1-b)I(x1,y0) + (1-a)bI(x0,y1) + abI(x1,y1).

    Args:
        gray_img: 2D grayscale array of shape (H, W).
        x: Horizontal coordinate(s) (scalar or array).
        y: Vertical coordinate(s) (scalar or array).
    Returns:
        Interpolated intensity value(s) as float or float64 ndarray.
    """
    img = np.asarray(gray_img, dtype=np.float64)
    h, w = img.shape[:2]
    scalar = np.isscalar(x) and np.isscalar(y)

    xf = np.clip(np.asarray(x, dtype=np.float64), 0.0, float(w - 1))
    yf = np.clip(np.asarray(y, dtype=np.float64), 0.0, float(h - 1))

    x0 = np.floor(xf).astype(np.int32)
    y0 = np.floor(yf).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    a = xf - x0
    b = yf - y0

    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]

    val = (
        (1.0 - a) * (1.0 - b) * v00
        + a * (1.0 - b) * v10
        + (1.0 - a) * b * v01
        + a * b * v11
    )
    return float(val) if scalar else val


def compute_output_size(corners: np.ndarray, max_dim: int = 1200) -> tuple[int, int]:
    """Estimate rectified output (width, height) from ordered quad [TL, TR, BR, BL].

    Args:
        corners: Ordered corners array of shape (4, 2).
        max_dim: Maximum allowed output dimension in pixels.
    Returns:
        Tuple (out_w, out_h) of positive integers.
    """
    tl, tr, br, bl = np.asarray(corners, dtype=np.float64)
    w_top = float(np.hypot(*(tr - tl)))
    w_bot = float(np.hypot(*(br - bl)))
    h_left = float(np.hypot(*(bl - tl)))
    h_right = float(np.hypot(*(br - tr)))

    out_w = max(1, int(round(max(w_top, w_bot))))
    out_h = max(1, int(round(max(h_left, h_right))))

    if max(out_w, out_h) > max_dim:
        scale = float(max_dim) / float(max(out_w, out_h))
        out_w = max(1, int(round(out_w * scale)))
        out_h = max(1, int(round(out_h * scale)))
    return out_w, out_h


def warp_document(
    image: np.ndarray,
    corners: np.ndarray,
    target_size: tuple[int, int] | None = None,
    interpolation: str = "bilinear",
) -> tuple[np.ndarray, np.ndarray]:
    """Warp 4 ordered document corners to a fronto-parallel rectangle via homography.

    Args:
        image: Input BGR or grayscale image of shape (H, W) or (H, W, 3).
        corners: Ordered corners [TL, TR, BR, BL] of shape (4, 2).
        target_size: Optional (width, height); estimated from quad edges if None.
        interpolation: 'bilinear' or 'nearest'.
    Returns:
        Tuple (warped_image, homography_matrix) where H has shape (3, 3), dtype float64.
    """
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    out_w, out_h = target_size if target_size is not None else compute_output_size(src)

    dst = np.array(
        [[0.0, 0.0], [out_w - 1.0, 0.0], [out_w - 1.0, out_h - 1.0], [0.0, out_h - 1.0]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    interp_flag = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
    warped = cv2.warpPerspective(image, H, (out_w, out_h), flags=interp_flag)
    return warped, H.astype(np.float64)
