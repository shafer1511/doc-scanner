"""Bilinear sampling, homography estimation, and document perspective warping."""

import cv2
import numpy as np


def bilinear_sample_gray(image: np.ndarray, x: float, y: float) -> float:
    """Sample a 2D image at a continuous coordinate with bilinear interpolation."""
    height, width = image.shape[:2]
    x = min(max(float(x), 0.0), width - 1.0)
    y = min(max(float(y), 0.0), height - 1.0)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0

    top = (1.0 - dx) * float(image[y0, x0]) + dx * float(image[y0, x1])
    bottom = (1.0 - dx) * float(image[y1, x0]) + dx * float(image[y1, x1])
    return (1.0 - dy) * top + dy * bottom


def compute_destination_dimensions(corners: np.ndarray) -> tuple[int, int]:
    """Return the (width, height) of the axis-aligned rectangle spanned by the corners."""
    top_left, top_right, bottom_right, bottom_left = np.asarray(corners, dtype=np.float64)

    width_top = float(np.linalg.norm(top_right - top_left))
    width_bottom = float(np.linalg.norm(bottom_right - bottom_left))
    height_left = float(np.linalg.norm(bottom_left - top_left))
    height_right = float(np.linalg.norm(bottom_right - top_right))

    width = int(round(max(width_top, width_bottom))) + 1
    height = int(round(max(height_left, height_right))) + 1
    return width, height


def compute_homography(src_corners: np.ndarray, dst_corners: np.ndarray) -> np.ndarray:
    """Estimate the 3x3 perspective transform mapping source corners onto destination corners."""
    return cv2.getPerspectiveTransform(
        np.asarray(src_corners, dtype=np.float32),
        np.asarray(dst_corners, dtype=np.float32),
    )


def warp_document(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the quadrilateral region of an image into a front-facing rectangle."""
    width, height = output_size if output_size is not None else compute_destination_dimensions(corners)
    dst_corners = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )
    homography = compute_homography(corners, dst_corners)
    warped = cv2.warpPerspective(image, homography, (width, height))
    return warped, homography
