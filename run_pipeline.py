"""Batch document-scanner pipeline: Canny -> Hough -> quadrilateral -> warp -> threshold."""

import argparse
import csv
import math
import os
import time

import cv2
import numpy as np
from shapely.geometry import Polygon

from src.core.canny import canny_edge_detector
from src.core.hough import detect_hough_lines
from src.core.quadrilateral import find_document_quadrilateral, order_corners_perimeter
from src.core.thresholding import compute_binarization_metrics, otsu_threshold, sauvola_threshold
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

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

DEFAULT_PARAMS = {
    "apply_blur": True,
    "suppress_duplicates": True,
    "use_edge_support": True,
    "top_k_lines": 16,
    "canny_low": 40.0,
    "canny_high": 120.0,
    "canny_sigma": 1.2,
    "hough_threshold": 50,
    "sauvola_window": 25,
    "sauvola_k": 0.2,
}


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Convert BGR/BGRA/grayscale input to contiguous uint8 grayscale."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] == 1:
        gray = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[2] == 4:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    elif arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("image must be a 2D or 3D array")
    if gray.dtype != np.uint8:
        if np.issubdtype(gray.dtype, np.floating) and gray.size and float(gray.max()) <= 1.0:
            gray = gray * 255.0
        gray = np.clip(gray, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(gray)


def compute_polygon_iou(pred_corners: np.ndarray, gt_corners: np.ndarray) -> float:
    """IoU of two 4-vertex polygons via shapely; invalid/empty/zero-union polygons give 0.0."""
    pred = np.asarray(pred_corners, dtype=np.float64)
    gt = np.asarray(gt_corners, dtype=np.float64)
    if pred.shape != (4, 2) or gt.shape != (4, 2):
        return 0.0
    poly_pred = Polygon(pred)
    poly_gt = Polygon(gt)
    if poly_pred.is_empty or poly_gt.is_empty:
        return 0.0
    if not poly_pred.is_valid or not poly_gt.is_valid:
        return 0.0
    union_area = poly_pred.union(poly_gt).area
    if union_area <= 0.0:
        return 0.0
    iou = poly_pred.intersection(poly_gt).area / union_area
    return float(min(max(iou, 0.0), 1.0))


def compute_corner_error_norm(
    pred_corners: np.ndarray,
    gt_corners: np.ndarray,
    image_shape: tuple[int, int],
) -> float:
    """Min mean corner distance over 4 cyclic rolls, divided by the image diagonal."""
    pred = np.asarray(pred_corners, dtype=np.float64)
    gt = np.asarray(gt_corners, dtype=np.float64)
    if pred.shape != (4, 2) or gt.shape != (4, 2):
        return float("nan")
    pred = order_corners_perimeter(pred)
    gt = order_corners_perimeter(gt)
    height, width = image_shape
    diagonal = math.hypot(float(height), float(width))
    if diagonal <= 0.0:
        return float("nan")
    min_dist = min(
        float(np.mean(np.linalg.norm(np.roll(pred, shift, axis=0) - gt, axis=1)))
        for shift in range(4)
    )
    return min_dist / diagonal


def run_single_image(
    image_input: np.ndarray,
    params: dict,
    gt_corners: np.ndarray | None = None,
    gt_ink_mask: np.ndarray | None = None,
) -> dict:
    """Run the five core stages on one image and return intermediates plus metrics.

    Args:
        image_input: grayscale/BGR/BGRA array (H, W) or (H, W, C).
        params: keys of DEFAULT_PARAMS; missing keys fall back to defaults.
        gt_corners: optional (4, 2) ground-truth corners in image coordinates.
        gt_ink_mask: optional binary ink mask aligned with the warped output.
    Returns:
        dict with edge/hough/quad/warp/threshold intermediates, metrics, and runtime_ms.
    """
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    gray = _to_gray_uint8(image_input)
    height, width = gray.shape[:2]

    start = time.perf_counter()
    canny = canny_edge_detector(
        gray,
        low_threshold=float(cfg["canny_low"]),
        high_threshold=float(cfg["canny_high"]),
        sigma=float(cfg["canny_sigma"]),
        apply_blur=bool(cfg["apply_blur"]),
    )
    hough = detect_hough_lines(
        canny["edge_map"],
        threshold=int(cfg["hough_threshold"]),
        top_k=int(cfg["top_k_lines"]),
        suppress_duplicates=bool(cfg["suppress_duplicates"]),
    )
    quad = find_document_quadrilateral(
        hough["filtered_lines"],
        canny["edge_map"],
        (height, width),
        use_edge_support=bool(cfg["use_edge_support"]),
    )

    warped = None
    homography = None
    otsu_value = None
    otsu_binary = None
    sauvola_map = None
    sauvola_binary = None

    if quad["status"] == "accepted":
        warped, homography = warp_document(gray, quad["corners"])
        otsu_value, otsu_binary = otsu_threshold(warped)
        sauvola_map, sauvola_binary = sauvola_threshold(
            warped,
            window_size=int(cfg["sauvola_window"]),
            k=float(cfg["sauvola_k"]),
        )
    runtime_ms = (time.perf_counter() - start) * 1000.0

    accepted = quad["status"] == "accepted"
    iou = float("nan")
    corner_error = float("nan")
    if gt_corners is not None:
        if accepted:
            iou = compute_polygon_iou(quad["corners"], gt_corners)
            corner_error = compute_corner_error_norm(quad["corners"], gt_corners, (height, width))
        else:
            iou = 0.0

    precision = recall = f1 = float("nan")
    if gt_ink_mask is not None and sauvola_binary is not None:
        gt_mask = np.asarray(gt_ink_mask)
        if gt_mask.shape != sauvola_binary.shape:
            gt_mask = cv2.resize(
                gt_mask.astype(np.uint8),
                (sauvola_binary.shape[1], sauvola_binary.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        ink_metrics = compute_binarization_metrics(sauvola_binary == 0, gt_mask > 0)
        precision = ink_metrics["precision"]
        recall = ink_metrics["recall"]
        f1 = ink_metrics["f1"]

    return {
        "image_id": None,
        "status": quad["status"],
        "reject_reason": quad["reject_reason"],
        "runtime_ms": runtime_ms,
        "resolution": f"{width}x{height}",
        "iou": iou,
        "corner_error_norm": corner_error,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gray": gray,
        "edge_map": canny["edge_map"],
        "grad_mag": canny["grad_mag"],
        "hough": hough,
        "quad": quad,
        "corners": quad["corners"],
        "warped": warped,
        "homography": homography,
        "otsu_threshold": otsu_value,
        "otsu_binary": otsu_binary,
        "sauvola_map": sauvola_map,
        "sauvola_binary": sauvola_binary,
    }


def _draw_hough_lines(image: np.ndarray, endpoints: list) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for (x0, y0), (x1, y1) in endpoints:
        cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2, cv2.LINE_AA)
    return canvas


def _draw_quadrilateral(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    points = np.rint(np.asarray(corners, dtype=np.float64)).astype(np.int32)
    cv2.polylines(canvas, [points], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    for index, (x, y) in enumerate(points):
        origin = (int(x) + 6, int(y) - 6)
        cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def visualize_and_save_intermediates(
    result: dict,
    output_dir: str,
    image_id: str,
    original_image: np.ndarray,
) -> None:
    """Write the seven numbered checkpoints for one image into output_dir/image_id/."""
    image_dir = os.path.join(output_dir, image_id)
    os.makedirs(image_dir, exist_ok=True)

    gray = result["gray"]
    original = _to_gray_uint8(original_image)
    if original.shape != gray.shape:
        original = cv2.resize(original, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(image_dir, "01_gray.png"), gray)
    cv2.imwrite(os.path.join(image_dir, "02_canny_edge.png"), result["edge_map"])
    cv2.imwrite(
        os.path.join(image_dir, "03_hough_lines.png"),
        _draw_hough_lines(original, result["hough"]["line_endpoints"]),
    )

    if result["corners"] is not None:
        quad_image = _draw_quadrilateral(original, result["corners"])
    else:
        quad_image = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            quad_image,
            f"rejected: {result['reject_reason']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(os.path.join(image_dir, "04_quadrilateral.png"), quad_image)

    if result["warped"] is not None:
        cv2.imwrite(os.path.join(image_dir, "05_warped.png"), result["warped"])
    if result["otsu_binary"] is not None:
        cv2.imwrite(os.path.join(image_dir, "06_binary_otsu.png"), result["otsu_binary"])
    if result["sauvola_binary"] is not None:
        cv2.imwrite(os.path.join(image_dir, "07_binary_sauvola.png"), result["sauvola_binary"])


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _resolve_image_path(images_dir: str, filename: str, original_path: str | None) -> str | None:
    candidates = [os.path.join(images_dir, filename)]
    if original_path:
        candidates.append(original_path)
        candidates.append(os.path.join(images_dir, os.path.basename(original_path)))
        candidates.append(os.path.join("filedata", os.path.basename(original_path)))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _iter_manifest(manifest_path: str, images_dir: str):
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = (row.get("image_id") or "").strip()
            filename = (row.get("filename") or "").strip()
            if not image_id or not filename:
                continue
            path = _resolve_image_path(images_dir, filename, row.get("original_path"))
            gt_corners = None
            if "x0" in row and row["x0"]:
                try:
                    gt_corners = np.array([
                        [float(row["x0"]), float(row["y0"])],
                        [float(row["x1"]), float(row["y1"])],
                        [float(row["x2"]), float(row["y2"])],
                        [float(row["x3"]), float(row["y3"])],
                    ], dtype=np.float64)
                except (ValueError, KeyError):
                    gt_corners = None
            mask_filename = row.get("mask_filename", "").strip()
            mask_path = None
            if mask_filename and path:
                cand_mask = os.path.join(os.path.dirname(path), mask_filename)
                if os.path.isfile(cand_mask):
                    mask_path = cand_mask
            yield image_id, row.get("split", ""), row.get("condition", ""), row.get("is_positive", ""), path, gt_corners, mask_path


def _iter_directory(images_dir: str):
    for filename in sorted(os.listdir(images_dir)):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
        image_id = os.path.splitext(filename)[0]
        path = os.path.join(images_dir, filename)
        yield image_id, "", "", "", path, None, None


def process_batch(
    manifest_path: str,
    images_dir: str,
    output_csv: str,
    intermediate_dir: str | None = None,
    ablation_flags: dict | None = None,
    gt_corners_map: dict | None = None,
) -> list[dict]:
    """Run run_single_image over a manifest CSV (or images_dir), save intermediates and CSV.

    Returns the list of per-image result dictionaries, including rejected cases.
    """
    params = {**DEFAULT_PARAMS, **(ablation_flags or {})}
    entries = (
        list(_iter_manifest(manifest_path, images_dir))
        if manifest_path and os.path.isfile(manifest_path)
        else list(_iter_directory(images_dir))
    )

    output_dir = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(output_dir, exist_ok=True)
    if intermediate_dir:
        os.makedirs(intermediate_dir, exist_ok=True)

    rows: list[dict] = []
    for image_id, split, condition, is_positive, image_path, row_gt_corners, mask_path in entries:
        row = {
            "image_id": image_id,
            "split": split,
            "condition": condition,
            "status": "rejected",
            "reject_reason": "image_not_found",
            "iou": float("nan"),
            "corner_error_norm": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "runtime_ms": float("nan"),
            "resolution": "",
        }
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) if image_path else None
        if image is not None:
            gt_corners = row_gt_corners
            if gt_corners is None and gt_corners_map and image_id in gt_corners_map:
                gt_corners = np.asarray(gt_corners_map[image_id], dtype=np.float64)
            gt_ink_mask = None
            if mask_path and os.path.isfile(mask_path):
                gt_ink_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            result = run_single_image(image, params, gt_corners=gt_corners, gt_ink_mask=gt_ink_mask)
            result = run_single_image(image, params, gt_corners=gt_corners, gt_ink_mask=gt_ink_mask)
            row.update(
                {
                    "image_id": image_id,
                    "split": split,
                    "condition": condition,
                    "status": result["status"],
                    "reject_reason": result["reject_reason"] or "",
                    "iou": result["iou"],
                    "corner_error_norm": result["corner_error_norm"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1": result["f1"],
                    "runtime_ms": result["runtime_ms"],
                    "resolution": result["resolution"],
                }
            )
            if intermediate_dir:
                visualize_and_save_intermediates(result, intermediate_dir, image_id, image)
            result.update(
                {"image_id": image_id, "split": split, "condition": condition, "csv_row": row}
            )
            rows.append(result)
        else:
            rows.append({"image_id": image_id, "csv_row": row, **row})

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for result in rows:
            source = result.get("csv_row", result)
            writer.writerow({column: _fmt(source.get(column)) for column in CSV_COLUMNS})

    return rows


def build_params(args: argparse.Namespace) -> dict:
    return {
        "apply_blur": not args.no_gaussian,
        "suppress_duplicates": not args.no_hough_suppression,
        "use_edge_support": not args.no_edge_support,
        "top_k_lines": args.top_k_lines,
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "canny_sigma": args.canny_sigma,
        "sauvola_window": args.sauvola_window,
        "sauvola_k": args.sauvola_k,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Document scanner pipeline (Canny/Hough/quad/warp/threshold)")
    parser.add_argument("--manifest", default=None, help="manifest CSV (e.g. data/manifest_development.csv)")
    parser.add_argument("--images-dir", required=True, help="directory containing the images")
    parser.add_argument("--output-csv", required=True, help="destination results CSV")
    parser.add_argument("--intermediate-dir", default=None, help="folder for intermediate PNGs")
    parser.add_argument("--no-gaussian", action="store_true", help="ablation 1: skip Gaussian blur")
    parser.add_argument("--no-hough-suppression", action="store_true", help="ablation 2: keep duplicate Hough lines")
    parser.add_argument("--no-edge-support", action="store_true", help="ablation 3: drop edge-support score term")
    parser.add_argument("--top-k-lines", type=int, default=16)
    parser.add_argument("--canny-low", type=float, default=40.0)
    parser.add_argument("--canny-high", type=float, default=120.0)
    parser.add_argument("--canny-sigma", type=float, default=1.2)
    parser.add_argument("--sauvola-window", type=int, default=25)
    parser.add_argument("--sauvola-k", type=float, default=0.2)
    args = parser.parse_args(argv)

    rows = process_batch(
        manifest_path=args.manifest,
        images_dir=args.images_dir,
        output_csv=args.output_csv,
        intermediate_dir=args.intermediate_dir,
        ablation_flags=build_params(args),
    )
    accepted = sum(1 for row in rows if row.get("status") == "accepted")
    print(f"processed={len(rows)} accepted={accepted} csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
