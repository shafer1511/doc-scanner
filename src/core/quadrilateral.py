from itertools import combinations
import cv2
import numpy as np


def intersect_lines(
    line1: dict, line2: dict, parallel_sin_tol: float = 0.15
) -> np.ndarray | None:
    """Solve 2x2 linear system for intersection of two polar lines (rho, theta).

    Args:
        line1: Dict with keys 'rho' and 'theta' (radians).
        line2: Dict with keys 'rho' and 'theta' (radians).
        parallel_sin_tol: Minimum |sin(theta1 - theta2)| to reject near-parallel pairs.
    Returns:
        Point array [x, y] of shape (2,), dtype float64, or None if near-parallel.
    """
    t1, t2 = float(line1["theta"]), float(line2["theta"])
    if abs(np.sin(t1 - t2)) < parallel_sin_tol:
        return None
    A = np.array(
        [[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]], dtype=np.float64
    )
    b = np.array([float(line1["rho"]), float(line2["rho"])], dtype=np.float64)
    try:
        pt = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return pt


def order_corners_perimeter(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners clockwise along perimeter starting from top-left [TL, TR, BR, BL].

    Args:
        pts: Array of shape (4, 2).
    Returns:
        Ordered corner array of shape (4, 2), dtype float32.
    """
    pts_f = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    centroid = np.mean(pts_f, axis=0)
    angles = np.arctan2(pts_f[:, 1] - centroid[1], pts_f[:, 0] - centroid[0])
    cw_pts = pts_f[np.argsort(angles)]
    tl_idx = int(np.argmin(cw_pts[:, 0] + cw_pts[:, 1]))
    ordered = np.roll(cw_pts, -tl_idx, axis=0)
    return ordered.astype(np.float32)


def polygon_area(pts: np.ndarray) -> float:
    """Compute polygon area via shoelace formula.

    Args:
        pts: Ordered vertex array of shape (N, 2).
    Returns:
        Non-negative area as float.
    """
    p = np.asarray(pts, dtype=np.float64)
    x = p[:, 0]
    y = p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def is_convex_quad(pts: np.ndarray, colinear_eps: float = 1e-3) -> bool:
    """Check whether 4 perimeter-ordered vertices form a strict non-degenerate convex quad.

    Args:
        pts: Array of shape (4, 2) (will be perimeter-ordered internally).
        colinear_eps: Minimum cross-product magnitude relative to edge lengths to reject collinearity.
    Returns:
        True if strictly convex with no 3 collinear vertices.
    """
    ordered = order_corners_perimeter(pts).astype(np.float64)
    signs = []
    for i in range(4):
        p0 = ordered[i]
        p1 = ordered[(i + 1) % 4]
        p2 = ordered[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = float(np.hypot(v1[0], v1[1]))
        n2 = float(np.hypot(v2[0], v2[1]))
        if n1 < 1e-3 or n2 < 1e-3:
            return False
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        if abs(cross) / (n1 * n2) <= colinear_eps:
            return False
        signs.append(cross > 0.0)
    return all(signs) or not any(signs)


def _min_corner_angle_deg(ordered: np.ndarray) -> tuple[float, float]:
    """Return (min_angle_deg, max_angle_deg) across 4 ordered vertices."""
    angles = []
    for i in range(4):
        p_prev = ordered[(i - 1) % 4]
        p_curr = ordered[i]
        p_next = ordered[(i + 1) % 4]
        u = p_prev - p_curr
        v = p_next - p_curr
        nu = float(np.hypot(u[0], u[1]))
        nv = float(np.hypot(v[0], v[1]))
        if nu < 1e-6 or nv < 1e-6:
            return 0.0, 180.0
        cos_a = np.clip(float(np.dot(u, v) / (nu * nv)), -1.0, 1.0)
        angles.append(float(np.rad2deg(np.arccos(cos_a))))
    return min(angles), max(angles)


def compute_edge_support(
    pts: np.ndarray, edge_map: np.ndarray, sample_step: float = 1.0, band: int = 2
) -> float:
    """Compute fraction of perimeter sample points within `band` px of an edge pixel.

    Args:
        pts: Ordered 4 corners of shape (4, 2).
        edge_map: Binary edge map of shape (H, W).
        sample_step: Sampling spacing along edges in pixels.
        band: Radius in pixels for edge neighborhood dilation.
    Returns:
        Mean edge support ratio across the 4 sides in [0.0, 1.0].
    """
    return float(np.mean(compute_side_edge_supports(pts, edge_map, sample_step=sample_step, band=band)))


def compute_side_edge_supports(
    pts: np.ndarray, edge_map: np.ndarray, sample_step: float = 2.0, band: int = 1
) -> list[float]:
    """Compute edge support ratio separately for each of the 4 quad sides.

    Args:
        pts: Ordered 4 corners of shape (4, 2).
        edge_map: Binary edge map of shape (H, W).
        sample_step: Sampling spacing along edges in pixels.
        band: Radius in pixels for edge neighborhood dilation.
    Returns:
        List of 4 support ratios in [0.0, 1.0].
    """
    h, w = edge_map.shape[:2]
    if band > 0:
        k = 2 * band + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        dilated = cv2.dilate((edge_map > 0).astype(np.uint8), kernel)
    else:
        dilated = (edge_map > 0).astype(np.uint8)

    ordered = np.asarray(pts, dtype=np.float64)
    side_supports: list[float] = []
    for i in range(4):
        p0 = ordered[i]
        p1 = ordered[(i + 1) % 4]
        length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        n_samples = max(4, int(np.ceil(length / max(0.5, sample_step))))
        ts = np.linspace(0.0, 1.0, n_samples)
        xs = np.clip(np.rint(p0[0] + ts * (p1[0] - p0[0])).astype(np.int32), 0, w - 1)
        ys = np.clip(np.rint(p0[1] + ts * (p1[1] - p0[1])).astype(np.int32), 0, h - 1)
        side_supports.append(float(np.mean(dilated[ys, xs] > 0)))

    return side_supports


def score_quadrilateral(
    pts: np.ndarray,
    edge_map: np.ndarray,
    img_shape: tuple[int, ...],
    use_edge_support: bool = True,
    area_weight: float = 0.35,
    support_weight: float = 0.65,
    dilated_edges: np.ndarray | None = None,
) -> dict[str, float | list[float]]:
    """Score a valid quadrilateral using edge support and area ratio.

    Args:
        pts: Ordered corners of shape (4, 2).
        edge_map: Binary edge map of shape (H, W).
        img_shape: Image shape tuple (H, W, ...).
        use_edge_support: Include perimeter edge support if True, else score by area only.
        area_weight: Weight of normalized area ratio.
        support_weight: Weight of perimeter edge support.
        dilated_edges: Optional pre-dilated binary edge map of shape (H, W) for speed.
    Returns:
        Dict with keys score, edge_support, min_side_support, area_ratio.
    """
    h, w = img_shape[:2]
    area_ratio = polygon_area(pts) / float(max(1, h * w))
    if dilated_edges is not None:
        side_sups = compute_side_edge_supports(pts, dilated_edges, band=0)
    else:
        side_sups = compute_side_edge_supports(pts, edge_map, band=1)
    support = float(np.mean(side_sups))
    min_side = float(min(side_sups))

    if use_edge_support:
        score = float(support_weight * support + area_weight * area_ratio)
    else:
        score = float(area_ratio)

    return {
        "score": score,
        "edge_support": support,
        "min_side_support": min_side,
        "area_ratio": float(area_ratio),
    }


def find_document_quad(
    lines: list[dict],
    edge_map: np.ndarray,
    img_shape: tuple[int, ...],
    use_edge_support: bool = True,
    min_area_ratio: float = 0.14,
    max_area_ratio: float = 0.98,
    min_edge_support: float = 0.55,
    min_side_support: float = 0.35,
    min_angle_deg: float = 45.0,
    max_angle_deg: float = 135.0,
    min_side_ratio: float = 0.12,
    border_margin_px: float = 3.0,
    max_candidates: int = 300,
) -> dict:
    """Group Hough lines into 4-sided document candidates, validate geometry, and return best quad or rejected.

    Args:
        lines: List of Hough line dicts with 'rho', 'theta', 'votes'.
        edge_map: Binary edge map of shape (H, W).
        img_shape: Image shape tuple (H, W, ...).
        use_edge_support: Whether edge support participates in scoring (Ablation 3 flag).
        min_area_ratio: Minimum quad area / image area.
        max_area_ratio: Maximum quad area / image area.
        min_edge_support: Minimum mean perimeter edge support to accept.
        min_side_support: Minimum per-side edge support to accept.
        min_angle_deg: Minimum interior corner angle in degrees.
        max_angle_deg: Maximum interior corner angle in degrees.
        min_side_ratio: Minimum side length divided by min(H, W).
        border_margin_px: Allowed overshoot outside image bounds in pixels.
        max_candidates: Maximum 4-line combinations evaluated.
    Returns:
        Dict with status ('accepted'|'rejected'), reject_reason, corners, score, edge_support, area_ratio, candidates.
    """
    h, w = img_shape[:2]
    if len(lines) < 4:
        return {
            "status": "rejected",
            "reject_reason": "insufficient_hough_lines",
            "corners": None,
            "score": 0.0,
            "edge_support": 0.0,
            "area_ratio": 0.0,
            "candidates": [],
        }

    # Pre-dilate once (3x3) for tight boundary alignment
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_edges = cv2.dilate((edge_map > 0).astype(np.uint8), kernel)

    pair_groups: list[tuple[tuple[int, int], tuple[int, int]]] = []
    n = len(lines)
    all_pairs = list(combinations(range(n), 2))

    # Build opposite-side pairs that do not intersect inside a tight center and have compatible angle
    valid_pairs = []
    for i, j in all_pairs:
        ang_diff = abs(np.rad2deg(lines[i]["theta"] - lines[j]["theta"])) % 180.0
        ang_diff = min(ang_diff, 180.0 - ang_diff)
        if ang_diff <= 35.0:
            valid_pairs.append((i, j))

    for idx_a in range(len(valid_pairs)):
        p_a = valid_pairs[idx_a]
        for idx_b in range(idx_a + 1, len(valid_pairs)):
            p_b = valid_pairs[idx_b]
            if len({p_a[0], p_a[1], p_b[0], p_b[1]}) < 4:
                continue
            ang_ab = abs(np.rad2deg(lines[p_a[0]]["theta"] - lines[p_b[0]]["theta"])) % 180.0
            ang_ab = min(ang_ab, 180.0 - ang_ab)
            if ang_ab < 35.0:
                continue
            pair_groups.append((p_a, p_b))
            if len(pair_groups) >= max_candidates:
                break
        if len(pair_groups) >= max_candidates:
            break

    if not pair_groups:
        return {
            "status": "rejected",
            "reject_reason": "no_orthogonal_line_pairs",
            "corners": None,
            "score": 0.0,
            "edge_support": 0.0,
            "area_ratio": 0.0,
            "candidates": [],
        }

    candidates = []
    reject_counts = {
        "out_of_bounds": 0,
        "non_convex": 0,
        "bad_angle": 0,
        "area_out_of_range": 0,
        "short_side": 0,
        "low_edge_support": 0,
    }
    min_side_px = min_side_ratio * float(min(h, w))

    for (a1, a2), (b1, b2) in pair_groups:
        c0 = intersect_lines(lines[a1], lines[b1])
        c1 = intersect_lines(lines[a1], lines[b2])
        c2 = intersect_lines(lines[a2], lines[b2])
        c3 = intersect_lines(lines[a2], lines[b1])
        if c0 is None or c1 is None or c2 is None or c3 is None:
            continue

        raw_pts = np.vstack([c0, c1, c2, c3])
        if (
            np.any(raw_pts[:, 0] < -border_margin_px)
            or np.any(raw_pts[:, 0] > w - 1 + border_margin_px)
            or np.any(raw_pts[:, 1] < -border_margin_px)
            or np.any(raw_pts[:, 1] > h - 1 + border_margin_px)
        ):
            reject_counts["out_of_bounds"] += 1
            continue

        ordered = order_corners_perimeter(raw_pts)
        if not is_convex_quad(ordered):
            reject_counts["non_convex"] += 1
            continue

        side_lengths = [
            float(np.hypot(*(ordered[(k + 1) % 4] - ordered[k]))) for k in range(4)
        ]
        if min(side_lengths) < min_side_px:
            reject_counts["short_side"] += 1
            continue

        min_ang, max_ang = _min_corner_angle_deg(ordered)
        if min_ang < min_angle_deg or max_ang > max_angle_deg:
            reject_counts["bad_angle"] += 1
            continue

        area_ratio = polygon_area(ordered) / float(max(1, h * w))
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            reject_counts["area_out_of_range"] += 1
            continue

        metrics = score_quadrilateral(
            ordered,
            edge_map,
            img_shape,
            use_edge_support=use_edge_support,
            dilated_edges=dilated_edges,
        )
        if use_edge_support and (
            metrics["edge_support"] < min_edge_support
            or metrics["min_side_support"] < min_side_support
        ):
            reject_counts["low_edge_support"] += 1
            continue

        candidates.append(
            {
                "corners": ordered,
                "score": metrics["score"],
                "edge_support": metrics["edge_support"],
                "area_ratio": metrics["area_ratio"],
            }
        )

    if not candidates:
        dominant_reason = max(reject_counts, key=lambda k: reject_counts[k])
        if reject_counts[dominant_reason] == 0:
            dominant_reason = "no_valid_intersections"
        return {
            "status": "rejected",
            "reject_reason": dominant_reason,
            "corners": None,
            "score": 0.0,
            "edge_support": 0.0,
            "area_ratio": 0.0,
            "candidates": [],
        }

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    return {
        "status": "accepted",
        "reject_reason": "",
        "corners": best["corners"],
        "score": float(best["score"]),
        "edge_support": float(best["edge_support"]),
        "area_ratio": float(best["area_ratio"]),
        "candidates": candidates[:5],
    }
