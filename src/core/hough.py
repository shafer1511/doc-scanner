import numpy as np


def hough_transform(
    edge_map: np.ndarray, rho_res: float = 1.0, theta_res_deg: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute standard Hough accumulator for rho = x*cos(theta) + y*sin(theta), theta in [0, pi).

    Args:
        edge_map: Binary edge map of shape (H, W) where nonzero pixels vote.
        rho_res: Quantization step for rho in pixels.
        theta_res_deg: Quantization step for theta in degrees.
    Returns:
        Tuple (accumulator, rhos, thetas) with shapes (R, T), (R,), and (T,) in radians.
    """
    h, w = edge_map.shape[:2]
    max_rho = float(np.ceil(np.hypot(w, h)))
    num_rhos = int(np.floor((2.0 * max_rho) / rho_res)) + 1
    rhos = np.linspace(-max_rho, max_rho, num_rhos, dtype=np.float64)

    num_thetas = max(1, int(round(180.0 / theta_res_deg)))
    thetas = np.deg2rad(np.arange(num_thetas, dtype=np.float64) * (180.0 / num_thetas))

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    ys, xs = np.nonzero(edge_map)
    if ys.size == 0:
        return accumulator, rhos, thetas

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    rho_vals = xs[:, None].astype(np.float64) * cos_t[None, :] + ys[:, None].astype(np.float64) * sin_t[None, :]
    rho_indices = np.rint((rho_vals - rhos[0]) / (rhos[1] - rhos[0])).astype(np.int32)
    rho_indices = np.clip(rho_indices, 0, len(rhos) - 1)

    for t_idx in range(len(thetas)):
        accumulator[:, t_idx] = np.bincount(rho_indices[:, t_idx], minlength=len(rhos))

    return accumulator, rhos, thetas


def _circular_theta_diff_deg(theta_a_rad: float, theta_b_rad: float) -> float:
    """Compute minimum angular distance in degrees modulo 180 degrees."""
    diff = abs(np.rad2deg(theta_a_rad - theta_b_rad)) % 180.0
    return float(min(diff, 180.0 - diff))


def find_hough_peaks(
    accumulator: np.ndarray,
    rhos: np.ndarray,
    thetas: np.ndarray,
    threshold: int,
    top_k: int = 20,
    suppress: bool = True,
    rho_supp: float = 15.0,
    theta_supp_deg: float = 10.0,
) -> list[dict]:
    """Select top Hough peaks above vote threshold with optional near-duplicate suppression.

    Args:
        accumulator: 2D vote array of shape (R, T).
        rhos: 1D rho bin centers of shape (R,).
        thetas: 1D theta angles in radians of shape (T,).
        threshold: Minimum accumulator votes required for a peak.
        top_k: Maximum number of peaks returned.
        suppress: Whether to suppress near-duplicate (rho, theta) lines (also handles theta periodicity).
        rho_supp: Minimum rho separation in pixels when suppress=True.
        theta_supp_deg: Minimum theta separation in degrees when suppress=True.
    Returns:
        List of peak dicts with keys rho, theta, votes, rho_idx, theta_idx.
    """
    if accumulator.size == 0 or int(np.max(accumulator)) < threshold:
        return []

    r_idx, t_idx = np.nonzero(accumulator >= threshold)
    if r_idx.size == 0:
        return []

    votes = accumulator[r_idx, t_idx]
    order = np.argsort(-votes, kind="mergesort")

    peaks: list[dict] = []
    for idx in order:
        ri = int(r_idx[idx])
        ti = int(t_idx[idx])
        rho = float(rhos[ri])
        theta = float(thetas[ti])
        v = int(votes[idx])

        if suppress:
            duplicate = False
            for p in peaks:
                t_diff = _circular_theta_diff_deg(theta, p["theta"])
                # Handle theta wrap-around near 0/pi where rho flips sign
                raw_diff = abs(np.rad2deg(theta - p["theta"])) % 180.0
                if raw_diff > 90.0:
                    r_diff = abs(rho + p["rho"])
                else:
                    r_diff = abs(rho - p["rho"])
                if r_diff < rho_supp and t_diff < theta_supp_deg:
                    duplicate = True
                    break
            if duplicate:
                continue

        peaks.append(
            {
                "rho": rho,
                "theta": theta,
                "votes": v,
                "rho_idx": ri,
                "theta_idx": ti,
            }
        )
        if len(peaks) >= top_k:
            break

    return peaks


def polar_to_cartesian_segment(
    rho: float, theta: float, width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert polar line (rho, theta) to clipped segment endpoints inside [0, width) x [0, height).

    Args:
        rho: Perpendicular distance from origin in pixels.
        theta: Normal angle in radians in [0, pi).
        width: Image width in pixels.
        height: Image height in pixels.
    Returns:
        Two endpoints ((x1, y1), (x2, y2)) clipped to image bounds.
    """
    a = float(np.cos(theta))
    b = float(np.sin(theta))
    candidates: list[tuple[float, float]] = []

    if abs(b) > 1e-8:
        for x in (0.0, float(width - 1)):
            y = (rho - x * a) / b
            if -1e-6 <= y <= height - 1 + 1e-6:
                candidates.append((x, float(np.clip(y, 0.0, height - 1))))
    if abs(a) > 1e-8:
        for y in (0.0, float(height - 1)):
            x = (rho - y * b) / a
            if -1e-6 <= x <= width - 1 + 1e-6:
                candidates.append((float(np.clip(x, 0.0, width - 1)), y))

    unique: list[tuple[float, float]] = []
    for pt in candidates:
        if not any(np.hypot(pt[0] - u[0], pt[1] - u[1]) < 1e-3 for u in unique):
            unique.append(pt)

    if len(unique) >= 2:
        best_pair = (unique[0], unique[1])
        best_dist = -1.0
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                d = float(np.hypot(unique[i][0] - unique[j][0], unique[i][1] - unique[j][1]))
                if d > best_dist:
                    best_dist = d
                    best_pair = (unique[i], unique[j])
        return (
            (int(round(best_pair[0][0])), int(round(best_pair[0][1]))),
            (int(round(best_pair[1][0])), int(round(best_pair[1][1]))),
        )

    x0 = a * rho
    y0 = b * rho
    scale = float(max(width, height)) * 2.0
    p1 = (int(round(x0 + scale * (-b))), int(round(y0 + scale * a)))
    p2 = (int(round(x0 - scale * (-b))), int(round(y0 - scale * a)))
    return p1, p2
