import argparse
import csv
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np

from src.core.canny import canny_edge_detection, to_grayscale
from src.core.hough import find_hough_peaks, hough_transform
from src.core.quadrilateral import find_document_quad, order_corners_perimeter, polygon_area
from src.core.thresholding import otsu_threshold, sauvola_threshold
from src.core.warping import warp_document


CSV_COLUMNS = [
    "image_id",
    "split",
    "condition",
    "status",
    "reject_reason",
    "iou",
    "corner_error_norm",
    "precision",
    "recall",
    "f1",
    "runtime_ms",
    "resolution",
]

DEFAULT_CONFIG = {
    "work_max_dim": 640,
    "sigma": 1.4,
    "low_ratio": 0.06,
    "high_ratio": 0.16,
    "use_gaussian": True,
    "rho_res": 1.0,
    "theta_res_deg": 1.0,
    "hough_thresh_ratio": 0.24,
    "hough_min_votes": 45,
    "top_k_lines": 20,
    "hough_suppress": True,
    "rho_supp": 18.0,
    "theta_supp_deg": 12.0,
    "use_edge_support": True,
    "min_area_ratio": 0.14,
    "max_area_ratio": 0.98,
    "min_edge_support": 0.72,
    "min_side_support": 0.50,
    "min_angle_deg": 45.0,
    "max_angle_deg": 135.0,
    "sauvola_window": 25,
    "sauvola_k": 0.2,
    "warp_max_dim": 1200,
}


def _clip_convex_polygon(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Clip subject polygon against convex clipper polygon via Sutherland-Hodgman."""
    def _inside(p: np.ndarray, a: np.ndarray, b: np.ndarray, sign: float) -> bool:
        return sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -1e-9

    def _intersect(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dc = a - b
        dp = p1 - p2
        n1 = a[0] * b[1] - a[1] * b[0]
        n2 = p1[0] * p2[1] - p1[1] * p2[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(denom) < 1e-12:
            return (p1 + p2) * 0.5
        return np.array([(n1 * dp[0] - n2 * dc[0]) / denom, (n1 * dp[1] - n2 * dc[1]) / denom], dtype=np.float64)

    clip_f = np.asarray(clipper, dtype=np.float64)
    c_area_signed = 0.5 * float(np.dot(clip_f[:, 0], np.roll(clip_f[:, 1], -1)) - np.dot(clip_f[:, 1], np.roll(clip_f[:, 0], -1)))
    sign = 1.0 if c_area_signed >= 0.0 else -1.0

    output = [np.asarray(pt, dtype=np.float64) for pt in subject]
    n_c = len(clip_f)
    for i in range(n_c):
        a = clip_f[i]
        b = clip_f[(i + 1) % n_c]
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if _inside(e, a, b, sign):
                if not _inside(s, a, b, sign):
                    output.append(_intersect(s, e, a, b))
                output.append(e)
            elif _inside(s, a, b, sign):
                output.append(_intersect(s, e, a, b))
            s = e
    return np.array(output, dtype=np.float64) if output else np.empty((0, 2), dtype=np.float64)


def polygon_iou(pred_quad: np.ndarray | None, gt_quad: np.ndarray | None) -> float:
    """Compute IoU between two convex quadrilaterals; returns 0.0 if pred_quad is None.

    Args:
        pred_quad: Predicted corners (4, 2) or None.
        gt_quad: Ground-truth corners (4, 2).
    Returns:
        Intersection-over-Union in [0.0, 1.0].
    """
    if pred_quad is None or gt_quad is None:
        return 0.0
    p = order_corners_perimeter(pred_quad).astype(np.float64)
    g = order_corners_perimeter(gt_quad).astype(np.float64)
    inter_poly = _clip_convex_polygon(p, g)
    inter_area = polygon_area(inter_poly) if len(inter_poly) >= 3 else 0.0
    union_area = polygon_area(p) + polygon_area(g) - inter_area
    if union_area <= 1e-9:
        return 0.0
    return float(np.clip(inter_area / union_area, 0.0, 1.0))


def normalized_corner_error(
    pred_quad: np.ndarray | None, gt_quad: np.ndarray | None, img_shape: tuple[int, ...]
) -> float | None:
    """Compute minimum perimeter-ordered corner error normalized by image diagonal.

    Args:
        pred_quad: Predicted corners (4, 2) or None.
        gt_quad: Ground-truth corners (4, 2).
        img_shape: Image shape (H, W, ...).
    Returns:
        Mean Euclidean corner error / sqrt(W^2 + H^2), or None if pred_quad/gt_quad is None.
    """
    if pred_quad is None or gt_quad is None:
        return None
    p = order_corners_perimeter(pred_quad).astype(np.float64)
    g = order_corners_perimeter(gt_quad).astype(np.float64)
    h, w = img_shape[:2]
    diag = float(np.hypot(w, h))
    if diag <= 1e-9:
        return None
    best = float("inf")
    for seq in (p, p[::-1]):
        for shift in range(4):
            rolled = np.roll(seq, shift, axis=0)
            err = float(np.mean(np.linalg.norm(rolled - g, axis=1)))
            if err < best:
                best = err
    return best / diag


def ink_prf1(pred_binary: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 for ink class (pixel < 128).

    Args:
        pred_binary: Predicted binary image (0=ink, 255=bg), shape (H, W).
        gt_mask: Ground-truth mask (0=ink, 255=bg), shape (H, W).
    Returns:
        Tuple (precision, recall, f1) in [0.0, 1.0].
    """
    pred_ink = np.asarray(pred_binary) < 128
    gt_ink = np.asarray(gt_mask) < 128
    tp = int(np.sum(pred_ink & gt_ink))
    fp = int(np.sum(pred_ink & (~gt_ink)))
    fn = int(np.sum((~pred_ink) & gt_ink))
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def resize_for_work(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Resize image so max(H, W) <= max_dim, returning (resized, scale_to_orig)."""
    h, w = image.shape[:2]
    cur_max = max(h, w)
    if cur_max <= max_dim:
        return image, 1.0
    scale = float(max_dim) / float(cur_max)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, 1.0 / scale


def process_single_image(image: np.ndarray, config: dict | None = None) -> dict:
    """Run full custom document scanner pipeline on an in-memory BGR/gray image.

    Args:
        image: Loaded image array of shape (H, W, 3) or (H, W).
        config: Parameter dictionary; defaults to DEFAULT_CONFIG.
    Returns:
        Dict with status, reject_reason, corners, runtime_ms, intermediates, warped, otsu, sauvola.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    t0 = time.perf_counter()

    work_img, scale_to_orig = resize_for_work(image, int(cfg["work_max_dim"]))
    canny_res = canny_edge_detection(
        work_img,
        sigma=float(cfg["sigma"]),
        low_ratio=float(cfg["low_ratio"]),
        high_ratio=float(cfg["high_ratio"]),
        use_gaussian=bool(cfg["use_gaussian"]),
    )
    edges = canny_res["edges"]
    acc, rhos, thetas = hough_transform(
        edges, rho_res=float(cfg["rho_res"]), theta_res_deg=float(cfg["theta_res_deg"])
    )
    max_votes = int(np.max(acc)) if acc.size > 0 else 0
    vote_thresh = max(int(cfg["hough_min_votes"]), int(round(float(cfg["hough_thresh_ratio"]) * max_votes)))
    peaks = find_hough_peaks(
        acc,
        rhos,
        thetas,
        threshold=vote_thresh,
        top_k=int(cfg["top_k_lines"]),
        suppress=bool(cfg["hough_suppress"]),
        rho_supp=float(cfg["rho_supp"]),
        theta_supp_deg=float(cfg["theta_supp_deg"]),
    )
    quad_res = find_document_quad(
        peaks,
        edges,
        work_img.shape,
        use_edge_support=bool(cfg["use_edge_support"]),
        min_area_ratio=float(cfg["min_area_ratio"]),
        max_area_ratio=float(cfg["max_area_ratio"]),
        min_edge_support=float(cfg["min_edge_support"]),
        min_side_support=float(cfg.get("min_side_support", 0.35)),
        min_angle_deg=float(cfg["min_angle_deg"]),
        max_angle_deg=float(cfg["max_angle_deg"]),
    )

    warped_bgr = None
    warped_gray = None
    otsu_res = None
    sauvola_res = None
    orig_corners = None

    if quad_res["status"] == "accepted" and quad_res["corners"] is not None:
        orig_corners = (quad_res["corners"] * scale_to_orig).astype(np.float32)
        warped_bgr, _ = warp_document(
            image, orig_corners, interpolation="bilinear"
        )
        warped_gray = to_grayscale(warped_bgr)
        otsu_res = otsu_threshold(warped_gray)
        sauvola_res = sauvola_threshold(
            warped_gray,
            window_size=int(cfg["sauvola_window"]),
            k=float(cfg["sauvola_k"]),
        )

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "status": quad_res["status"],
        "reject_reason": quad_res["reject_reason"],
        "corners": orig_corners,
        "work_corners": quad_res["corners"],
        "scale_to_orig": scale_to_orig,
        "score": quad_res["score"],
        "edge_support": quad_res["edge_support"],
        "area_ratio": quad_res["area_ratio"],
        "candidates": quad_res["candidates"],
        "canny": canny_res,
        "accumulator": acc,
        "rhos": rhos,
        "thetas": thetas,
        "peaks": peaks,
        "warped_bgr": warped_bgr,
        "warped_gray": warped_gray,
        "otsu": otsu_res,
        "sauvola": sauvola_res,
        "runtime_ms": runtime_ms,
    }


def load_labels(labels_path: str | None) -> dict:
    """Load corner ground-truth JSON mapping image_id -> 4x2 list if present."""
    if not labels_path or not os.path.exists(labels_path):
        return {}
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_image_intermediates(
    out_dir: Path, img_id: str, image: np.ndarray, res: dict, gt_corners: np.ndarray | None
) -> None:
    """Save compact intermediate visualizations after runtime measurement."""
    case_dir = out_dir / img_id
    case_dir.mkdir(parents=True, exist_ok=True)
    edges = res["canny"]["edges"]
    cv2.imwrite(str(case_dir / "01_edges.png"), edges)

    # Work-scale overlay of Hough lines + predicted/GT quad
    h_w, w_w = edges.shape[:2]
    work_vis = cv2.resize(image, (w_w, h_w), interpolation=cv2.INTER_AREA)
    for pk in res["peaks"]:
        rho, theta = float(pk["rho"]), float(pk["theta"])
        a, b = float(np.cos(theta)), float(np.sin(theta))
        x0, y0 = a * rho, b * rho
        pt1 = (int(round(x0 + 2000 * (-b))), int(round(y0 + 2000 * a)))
        pt2 = (int(round(x0 - 2000 * (-b))), int(round(y0 - 2000 * a)))
        cv2.line(work_vis, pt1, pt2, (0, 180, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(case_dir / "02_hough_lines.png"), work_vis)

    overlay = image.copy()
    if gt_corners is not None:
        cv2.polylines(overlay, [gt_corners.astype(np.int32)], True, (255, 0, 255), 4, cv2.LINE_AA)
    if res["corners"] is not None:
        cv2.polylines(overlay, [res["corners"].astype(np.int32)], True, (0, 220, 0), 4, cv2.LINE_AA)
    overlay_small = cv2.resize(overlay, (w_w, h_w), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(case_dir / "03_quad_overlay.png"), overlay_small)

    if res["warped_bgr"] is not None:
        cv2.imwrite(str(case_dir / "04_warped.png"), res["warped_bgr"])
    if res["otsu"] is not None:
        otsu_img = res["otsu"]["binary"] if isinstance(res["otsu"], dict) else res["otsu"]
        cv2.imwrite(str(case_dir / "05_otsu.png"), otsu_img)
    if res["sauvola"] is not None:
        sauvola_img = res["sauvola"]["binary"] if isinstance(res["sauvola"], dict) else res["sauvola"]
        cv2.imwrite(str(case_dir / "06_sauvola.png"), sauvola_img)


def run_batch(
    manifest_path: str,
    images_dir: str,
    output_csv: str,
    split_name: str,
    config: dict,
    labels_path: str | None = None,
    seed: int = 42,
    save_intermediates_dir: str | None = None,
    intermediate_ids: set[str] | None = None,
) -> list[dict]:
    """Run pipeline over all rows in a manifest CSV and write 12-column results CSV."""
    np.random.seed(seed)
    labels = load_labels(labels_path)
    rows_out: list[dict] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)

    for row in manifest_rows:
        img_id = row["image_id"]
        split = row.get("split", split_name)
        condition = row.get("condition", "")
        is_pos = str(row.get("is_positive", "1")) == "1"
        img_path = os.path.join(images_dir, row["filename"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Missing image: {img_path}")

        h, w = img.shape[:2]
        res = process_single_image(img, config)

        gt_entry = labels.get(img_id)
        gt_corners = np.array(gt_entry["corners"], dtype=np.float32) if (gt_entry and "corners" in gt_entry) else None

        if save_intermediates_dir and (intermediate_ids is None or img_id in intermediate_ids):
            _save_image_intermediates(Path(save_intermediates_dir), img_id, img, res, gt_corners)

        if not is_pos:
            iou_val = "0.0000" if res["status"] == "accepted" else "N/A"
            ce_val = "N/A"
        elif gt_corners is not None:
            if res["status"] == "accepted":
                iou_val = f"{polygon_iou(res['corners'], gt_corners):.4f}"
                ce = normalized_corner_error(res["corners"], gt_corners, img.shape)
                ce_val = f"{ce:.4f}" if ce is not None else "N/A"
            else:
                iou_val = "0.0000"
                ce_val = "N/A"
        else:
            iou_val = "0.0000" if res["status"] == "rejected" else "N/A"
            ce_val = "N/A"

        rows_out.append(
            {
                "image_id": img_id,
                "split": split,
                "condition": condition,
                "status": res["status"],
                "reject_reason": res["reject_reason"],
                "iou": iou_val,
                "corner_error_norm": ce_val,
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
                "runtime_ms": f"{res['runtime_ms']:.2f}",
                "resolution": f"{w}x{h}",
            }
        )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_out)

    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run custom Document Scanner pipeline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--config", default=None)
    parser.add_argument("--labels-json", default="data/labels.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-intermediates", default=None)
    parser.add_argument("--no-gaussian", action="store_true")
    parser.add_argument("--no-hough-suppression", action="store_true")
    parser.add_argument("--no-edge-support", action="store_true")
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    if args.no_gaussian:
        cfg["use_gaussian"] = False
    if args.no_hough_suppression:
        cfg["hough_suppress"] = False
    if args.no_edge_support:
        cfg["use_edge_support"] = False

    run_batch(
        args.manifest,
        args.images_dir,
        args.output_csv,
        args.split,
        cfg,
        labels_path=args.labels_json,
        seed=args.seed,
        save_intermediates_dir=args.save_intermediates,
    )


if __name__ == "__main__":
    main()
