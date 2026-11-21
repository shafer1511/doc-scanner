"""Document quadrilateral selection from Hough lines and an edge map."""

import itertools

import numpy as np


def intersect_lines(
    line1: tuple[float, float] | tuple[float, float, int],
    line2: tuple[float, float] | tuple[float, float, int],
    min_angle_diff: float = np.radians(10.0),
) -> tuple[float, float] | None:
    """Solve the two rho = x cos(theta) + y sin(theta) equations; near-parallel lines give None."""
    rho1, theta1 = float(line1[0]), float(line1[1])
    rho2, theta2 = float(line2[0]), float(line2[1])
    matrix = np.array(
        [[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]],
        dtype=np.float64,
    )
    if abs(float(np.linalg.det(matrix))) < 1e-4:
        return None
    x, y = np.linalg.solve(matrix, np.array([rho1, rho2], dtype=np.float64))
    return float(x), float(y)


def order_corners_perimeter(corners: np.ndarray) -> np.ndarray:
    """Order the four vertices clockwise on screen (y down) with the top-left corner first."""
    corners = np.asarray(corners, dtype=np.float64)
    centroid = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    ordered = corners[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)


def is_convex_quadrilateral(corners: np.ndarray) -> bool:
    """True when consecutive edge cross-products share one strictly positive/negative sign."""
    corners = np.asarray(corners, dtype=np.float64)
    edges = np.roll(corners, -1, axis=0) - corners
    nxt = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * nxt[:, 1] - edges[:, 1] * nxt[:, 0]
    if np.any(np.abs(crosses) <= 1e-5):
        return False
    return bool(np.all(crosses > 0.0) or np.all(crosses < 0.0))


def polygon_area(corners: np.ndarray) -> float:
    """Shoelace area of the polygon described by the given vertices."""
    corners = np.asarray(corners, dtype=np.float64)
    x, y = corners[:, 0], corners[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def compute_edge_support(
    corners: np.ndarray,
    edge_map: np.ndarray,
    num_samples: int = 50,
    tolerance: int = 2,
) -> float:
    """Fraction of edge samples whose square tolerance window touches an edge pixel."""
    edge_map = np.asarray(edge_map)
    height, width = edge_map.shape
    corners = np.asarray(corners, dtype=np.float64)
    starts = corners
    ends = np.roll(corners, -1, axis=0)
    t = np.linspace(0.0, 1.0, num_samples, endpoint=False)[:, None, None]
    points = starts[None, :, :] + t * (ends - starts)[None, :, :]
    xs = np.rint(points[..., 0]).astype(np.intp)
    ys = np.rint(points[..., 1]).astype(np.intp)

    supported = np.zeros(xs.shape, dtype=bool)
    for dy in range(-tolerance, tolerance + 1):
        yy = np.clip(ys + dy, 0, height - 1)
        for dx in range(-tolerance, tolerance + 1):
            xx = np.clip(xs + dx, 0, width - 1)
            supported |= edge_map[yy, xx] != 0
    return float(supported.mean())


def score_quadrilateral(
    corners: np.ndarray,
    edge_map: np.ndarray,
    image_shape: tuple[int, int],
    use_edge_support: bool = True,
    alpha: float = 0.5,
) -> float:
    """Blend edge support with the quad's share of the image area; ablation drops the support term."""
    height, width = image_shape
    area_ratio = polygon_area(corners) / float(height * width)
    if not use_edge_support:
        return area_ratio
    support = compute_edge_support(corners, edge_map)
    return alpha * support + (1.0 - alpha) * area_ratio


def find_document_quadrilateral(
    lines: list,
    edge_map: np.ndarray,
    image_shape: tuple[int, int],
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.98,
    min_edge_support: float = 0.15,
    use_edge_support: bool = True,
    max_candidates: int = 150,
) -> dict:
    """Search 4-line combinations for the best in-bounds convex quadrilateral."""
    if len(lines) < 4:
        return {
            "status": "rejected",
            "reject_reason": "insufficient_lines",
            "corners": None,
            "score": 0.0,
        }

    height, width = image_shape
    pool = sorted(lines, key=lambda line: line[2] if len(line) > 2 else 0, reverse=True)[:16]
    best_corners: np.ndarray | None = None
    best_score, best_area_ratio, best_support = -1.0, 0.0, 0.0

    for index, combo in enumerate(itertools.combinations(pool, 4)):
        if index >= max_candidates:
            break
        points = []
        for line_a, line_b in itertools.combinations(combo, 2):
            point = intersect_lines(line_a, line_b)
            if point is None:
                continue
            x, y = point
            if -1e-6 <= x <= width + 1e-6 and -1e-6 <= y <= height + 1e-6:
                points.append(point)
        if len(points) != 4:
            continue

        corners = order_corners_perimeter(np.asarray(points, dtype=np.float64))
        if not is_convex_quadrilateral(corners):
            continue
        area_ratio = polygon_area(corners) / float(height * width)
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue

        support = compute_edge_support(corners, edge_map) if use_edge_support else 0.0
        if use_edge_support and support < min_edge_support:
            continue

        score = score_quadrilateral(corners, edge_map, image_shape, use_edge_support)
        if score > best_score:
            best_corners, best_score = corners, score
            best_area_ratio, best_support = area_ratio, support

    if best_corners is None:
        return {
            "status": "rejected",
            "reject_reason": "no_convex_quad_found",
            "corners": None,
            "score": 0.0,
        }
    return {
        "status": "accepted",
        "reject_reason": None,
        "corners": best_corners,
        "score": best_score,
        "area_ratio": best_area_ratio,
        "edge_support": best_support,
    }
