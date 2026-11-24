import argparse
import csv
import os
import time
from pathlib import Path
import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from run_pipeline import (
    CSV_COLUMNS,
    DEFAULT_CONFIG,
    load_labels,
    normalized_corner_error,
    polygon_iou,
    resize_for_work,
)
from src.core.quadrilateral import find_document_quad
from src.core.warping import warp_document


def run_baseline_on_image(image: np.ndarray, config: dict | None = None) -> dict:
    """Run OpenCV/skimage reference baseline (cv2.Canny, cv2.HoughLines, cv2 Otsu, skimage Sauvola).

    Args:
        image: Input BGR or grayscale array of shape (H, W, 3) or (H, W).
        config: Optional parameter dictionary.
    Returns:
        Dict with status, reject_reason, corners, runtime_ms, otsu_binary, sauvola_binary.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    t0 = time.perf_counter()

    work_img, scale_to_orig = resize_for_work(image, int(cfg["work_max_dim"]))
    gray_w = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY) if work_img.ndim == 3 else work_img
    blurred = cv2.GaussianBlur(gray_w, (9, 9), sigmaX=float(cfg["sigma"]))
    edges = cv2.Canny(blurred, 50, 140)

    raw_lines = cv2.HoughLines(edges, rho=1.0, theta=np.pi / 180.0, threshold=60)
    peaks: list[dict] = []
    if raw_lines is not None:
        for item in raw_lines[: int(cfg["top_k_lines"])]:
            rho, theta = float(item[0][0]), float(item[0][1])
            peaks.append({"rho": rho, "theta": theta, "votes": 1})

    quad_res = find_document_quad(
        peaks,
        edges,
        work_img.shape,
        use_edge_support=True,
        min_area_ratio=float(cfg["min_area_ratio"]),
        max_area_ratio=float(cfg["max_area_ratio"]),
        min_edge_support=float(cfg["min_edge_support"]),
        min_side_support=float(cfg.get("min_side_support", 0.50)),
        min_angle_deg=float(cfg["min_angle_deg"]),
        max_angle_deg=float(cfg["max_angle_deg"]),
    )

    orig_corners = None
    otsu_bin = None
    sauvola_bin = None
    if quad_res["status"] == "accepted" and quad_res["corners"] is not None:
        orig_corners = (quad_res["corners"] * scale_to_orig).astype(np.float32)
        warped_bgr, _ = warp_document(image, orig_corners, interpolation="bilinear")
        warped_gray = (
            cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
            if warped_bgr.ndim == 3
            else warped_bgr
        )
        _, otsu_bin = cv2.threshold(
            warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        t_map = threshold_sauvola(
            warped_gray,
            window_size=int(cfg["sauvola_window"]),
            k=float(cfg["sauvola_k"]),
        )
        sauvola_bin = np.where(warped_gray > t_map, np.uint8(255), np.uint8(0))

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "status": quad_res["status"],
        "reject_reason": quad_res["reject_reason"],
        "corners": orig_corners,
        "edges": edges,
        "otsu_binary": otsu_bin,
        "sauvola_binary": sauvola_bin,
        "runtime_ms": runtime_ms,
    }


def run_baseline_batch(
    manifest_path: str,
    images_dir: str,
    output_csv: str,
    split_name: str,
    labels_path: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """Run reference baseline across manifest and write 12-column CSV."""
    np.random.seed(seed)
    labels = load_labels(labels_path)
    rows_out: list[dict] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

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
        res = run_baseline_on_image(img)
        gt_entry = labels.get(img_id)
        gt_corners = np.array(gt_entry["corners"], dtype=np.float32) if (gt_entry and "corners" in gt_entry) else None

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
    parser = argparse.ArgumentParser(description="Run OpenCV/skimage baseline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--labels-json", default="data/labels.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_baseline_batch(
        args.manifest, args.images_dir, args.output_csv, args.split, args.labels_json, args.seed
    )


if __name__ == "__main__":
    main()
