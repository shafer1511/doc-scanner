"""OpenCV reference (baseline) implementations for the document scanner pipeline."""

import time

import cv2
import numpy as np

from src.core.quadrilateral import (
    is_convex_quadrilateral,
    order_corners_perimeter,
    polygon_area,
)


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/BGR/BGRA or numeric input to a contiguous uint8 grayscale array."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] == 1:
        gray = arr[:, :, 0]
    elif arr.ndim == 3:
        if arr.dtype == np.uint8:
            code = cv2.COLOR_BGRA2GRAY if arr.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(arr, code)
        else:
            gray = arr[..., :3].mean(axis=2)
    else:
        raise ValueError("image must be a 2D or 3D array")

    if gray.dtype != np.uint8:
        if np.issubdtype(gray.dtype, np.floating) and gray.size and float(gray.max()) <= 1.0:
            gray = gray * 255.0
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(gray)


def baseline_canny(
    image: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
    aperture_size: int = 3,
) -> np.ndarray:
    """OpenCV Canny edge map; returns uint8 (H, W) with values in {0, 255}."""
    gray = _to_gray_uint8(image)
    if gray.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    return cv2.Canny(gray, low_threshold, high_threshold, apertureSize=aperture_size)


def baseline_hough_lines(
    edge_map: np.ndarray,
    rho_res: float = 1.0,
    theta_res: float = np.pi / 180.0,
    threshold: int = 50,
) -> list[tuple[float, float, int]]:
    """OpenCV standard Hough transform; returns [(rho, theta, votes)] sorted by votes."""
    edges = np.asarray(edge_map)
    if edges.ndim != 2 or edges.size == 0 or not np.any(edges):
        return []
    lines = cv2.HoughLines(_to_gray_uint8(edges), rho_res, theta_res, threshold)
    if lines is None:
        return []

    ys, xs = np.nonzero(edges)
    tolerance = 0.5 * rho_res
    results: list[tuple[float, float, int]] = []
    for line in lines.reshape(-1, lines.shape[-1]):
        rho, theta = float(line[0]), float(line[1])
        votes = int(np.count_nonzero(np.abs(xs * np.cos(theta) + ys * np.sin(theta) - rho) <= tolerance))
        results.append((rho, theta, votes))
    return sorted(results, key=lambda item: item[2], reverse=True)


def baseline_quadrilateral_from_contours(
    edge_map: np.ndarray,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.98,
) -> dict:
    """Largest convex 4-corner contour within the area bounds, as a status dict."""
    edges = np.asarray(edge_map)
    if edges.ndim != 2 or edges.size == 0:
        return {"status": "rejected", "reject_reason": "no_contour_found", "corners": None}
    edges = _to_gray_uint8(edges)
    height, width = edges.shape
    image_area = float(height * width)
    if image_area <= 0.0:
        return {"status": "rejected", "reject_reason": "no_contour_found", "corners": None}

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_corners: np.ndarray | None = None
    best_area = 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area_ratio = cv2.contourArea(contour) / image_area
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        corners = order_corners_perimeter(approx.reshape(4, 2).astype(np.float64))
        if not is_convex_quadrilateral(corners):
            continue
        area = polygon_area(corners)
        if area > best_area:
            best_area, best_corners = area, corners

    if best_corners is None:
        return {"status": "rejected", "reject_reason": "no_contour_found", "corners": None}
    return {"status": "accepted", "reject_reason": None, "corners": best_corners}


def baseline_warp(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the outlined document to a front-facing rectangle; returns (image, 3x3 homography)."""
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if output_size is None:
        top_left, top_right, bottom_right, bottom_left = src.astype(np.float64)
        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)
        output_size = (int(round(max(width_top, width_bottom))), int(round(max(height_left, height_right))))
    width, height = output_size
    width, height = max(int(width), 1), max(int(height), 1)

    dst = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(np.asarray(image), homography, (width, height))
    return warped, homography


def baseline_threshold_otsu(image: np.ndarray) -> tuple[float, np.ndarray]:
    """OpenCV global Otsu binarization; returns (threshold, uint8 binary image)."""
    gray = _to_gray_uint8(image)
    if gray.size == 0:
        return 0.0, np.zeros(gray.shape, dtype=np.uint8)
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold), binary


def baseline_threshold_adaptive(
    image: np.ndarray,
    window_size: int = 25,
    c: float = 10.0,
) -> np.ndarray:
    """OpenCV Gaussian adaptive threshold; returns uint8 binary image."""
    gray = _to_gray_uint8(image)
    if gray.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    limit = min(gray.shape[0], gray.shape[1]) | 1
    window_size = min(max(int(window_size) | 1, 3), limit)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        window_size,
        c,
    )


def run_baseline_pipeline(image: np.ndarray) -> dict:
    """Run Canny, contour quad detection, warp, and both thresholds; timings include only CPU."""
    start = time.perf_counter()
    empty_result = {
        "status": "rejected",
        "reject_reason": "invalid_image",
        "corners": None,
        "warped": None,
        "binary_otsu": None,
        "binary_adaptive": None,
    }
    try:
        arr = np.asarray(image)
        if arr.size == 0 or arr.ndim not in (2, 3):
            return {**empty_result, "runtime_ms": (time.perf_counter() - start) * 1000.0}
        gray = _to_gray_uint8(arr)
    except (TypeError, ValueError):
        return {**empty_result, "runtime_ms": (time.perf_counter() - start) * 1000.0}

    edge_map = baseline_canny(gray)
    quad = baseline_quadrilateral_from_contours(edge_map)
    if quad["status"] != "accepted":
        return {**quad, "warped": None, "binary_otsu": None, "binary_adaptive": None,
                "runtime_ms": (time.perf_counter() - start) * 1000.0}

    warped, _ = baseline_warp(gray, quad["corners"])
    _, binary_otsu = baseline_threshold_otsu(warped)
    binary_adaptive = baseline_threshold_adaptive(warped)
    return {
        "status": "accepted",
        "reject_reason": None,
        "corners": quad["corners"],
        "warped": warped,
        "binary_otsu": binary_otsu,
        "binary_adaptive": binary_adaptive,
        "runtime_ms": (time.perf_counter() - start) * 1000.0,
    }
