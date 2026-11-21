"""Hough line transform built from NumPy/SciPy primitives."""

import numpy as np
from scipy.ndimage import maximum_filter


def hough_accumulator(
    edge_map: np.ndarray,
    rho_res: float = 1.0,
    theta_res: float = np.pi / 180.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate votes for rho = x cos(theta) + y sin(theta); blank maps give zeros."""
    edge_map = np.asarray(edge_map)
    height, width = edge_map.shape
    diag = float(np.ceil(np.hypot(height, width)))
    n_rho = int(np.ceil(2.0 * diag / rho_res)) + 1
    rhos = -diag + np.arange(n_rho) * rho_res
    thetas = np.arange(-np.pi / 2.0, np.pi / 2.0, theta_res)
    accumulator = np.zeros((n_rho, thetas.size), dtype=np.int32)

    ys, xs = np.nonzero(edge_map)
    if ys.size == 0 or thetas.size == 0:
        return accumulator, rhos, thetas

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    n_theta = thetas.size
    theta_offsets = np.arange(n_theta, dtype=np.intp)
    inv_res = 1.0 / rho_res
    step = max(1, 4_000_000 // n_theta)
    for start in range(0, ys.size, step):
        gx = xs[start : start + step, None].astype(np.float64)
        gy = ys[start : start + step, None].astype(np.float64)
        rho = gx * cos_t + gy * sin_t
        idx = np.rint((rho + diag) * inv_res).astype(np.intp)
        np.clip(idx, 0, n_rho - 1, out=idx)
        idx *= n_theta
        idx += theta_offsets
        counts = np.bincount(idx.ravel(), minlength=n_rho * n_theta)
        accumulator += counts.reshape(n_rho, n_theta).astype(np.int32)

    return accumulator, rhos, thetas


def find_hough_peaks(
    accumulator: np.ndarray,
    rhos: np.ndarray,
    thetas: np.ndarray,
    threshold: int = 50,
    neighborhood_size: tuple[int, int] = (5, 5),
) -> list[tuple[float, float, int]]:
    """Local maxima with votes >= threshold, sorted by votes descending."""
    accumulator = np.asarray(accumulator)
    if accumulator.size == 0:
        return []
    local_max = maximum_filter(accumulator, size=neighborhood_size, mode="constant", cval=0)
    rows, cols = np.nonzero((accumulator == local_max) & (accumulator >= threshold))
    order = np.argsort(-accumulator[rows, cols], kind="stable")
    return [
        (float(rhos[rows[i]]), float(thetas[cols[i]]), int(accumulator[rows[i], cols[i]]))
        for i in order
    ]


def suppress_similar_lines(
    lines: list[tuple[float, float, int]],
    min_rho_diff: float = 20.0,
    min_theta_diff: float = np.radians(10.0),
    top_k: int | None = None,
) -> list[tuple[float, float, int]]:
    """Greedy vote-ordered suppression; equal lines under (rho, theta+pi) -> -rho are merged."""
    accepted: list[tuple[float, float, int]] = []
    for rho, theta, votes in sorted(lines, key=lambda line: -line[2]):
        for acc_rho, acc_theta, _ in accepted:
            dtheta = abs(theta - acc_theta) % np.pi
            dtheta = min(dtheta, np.pi - dtheta)
            if dtheta < min_theta_diff and (
                abs(rho - acc_rho) < min_rho_diff or abs(rho + acc_rho) < min_rho_diff
            ):
                break
        else:
            accepted.append((rho, theta, votes))
            if top_k is not None and len(accepted) >= top_k:
                break
    return accepted


def hough_line_to_points(
    rho: float,
    theta: float,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Clip x cos(theta) + y sin(theta) = rho to the pixel rectangle; None if no chord."""
    if width <= 0 or height <= 0:
        return None
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    candidates: list[tuple[float, float]] = []
    if abs(sin_t) > 1e-12:
        for x in (0, width - 1):
            candidates.append((float(x), (rho - x * cos_t) / sin_t))
    if abs(cos_t) > 1e-12:
        for y in (0, height - 1):
            candidates.append(((rho - y * sin_t) / cos_t, float(y)))

    points: list[tuple[int, int]] = []
    for x, y in candidates:
        if -1e-9 <= x <= width - 1 + 1e-9 and -1e-9 <= y <= height - 1 + 1e-9:
            px = min(max(int(np.floor(x + 0.5)), 0), width - 1)
            py = min(max(int(np.floor(y + 0.5)), 0), height - 1)
            if (px, py) not in points:
                points.append((px, py))
        if len(points) == 2:
            break
    if len(points) < 2:
        return None
    return points[0], points[1]


def detect_hough_lines(
    edge_map: np.ndarray,
    rho_res: float = 1.0,
    theta_res: float = np.pi / 180.0,
    threshold: int = 50,
    min_rho_diff: float = 20.0,
    min_theta_diff: float = np.radians(10.0),
    top_k: int = 20,
    suppress_duplicates: bool = True,
) -> dict:
    """Accumulate, find peaks, optionally suppress duplicates, and clip line endpoints."""
    edge_map = np.asarray(edge_map)
    height, width = edge_map.shape
    accumulator, rhos, thetas = hough_accumulator(edge_map, rho_res, theta_res)
    raw_peaks = find_hough_peaks(accumulator, rhos, thetas, threshold)
    if suppress_duplicates:
        filtered_lines = suppress_similar_lines(raw_peaks, min_rho_diff, min_theta_diff, top_k)
    else:
        filtered_lines = raw_peaks if top_k is None else raw_peaks[:top_k]

    line_endpoints = []
    for rho, theta, _ in filtered_lines:
        pts = hough_line_to_points(rho, theta, width, height)
        if pts is not None:
            line_endpoints.append(pts)

    return {
        "accumulator": accumulator,
        "rhos": rhos,
        "thetas": thetas,
        "raw_peaks": raw_peaks,
        "filtered_lines": filtered_lines,
        "line_endpoints": line_endpoints,
    }
