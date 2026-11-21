"""Run OpenCV baseline pipeline over manifest and output CSV matching RULE.md schema."""
import csv
import math
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from src.baseline.baseline import run_baseline_pipeline
from run_pipeline import compute_polygon_iou, compute_corner_error_norm, compute_binarization_metrics, CSV_COLUMNS, _fmt, _resolve_image_path

def run_baseline_batch(manifest_path, images_dir, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rows = []
    
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        
    for r in reader:
        image_id = r["image_id"]
        split = r.get("split", "")
        condition = r.get("condition", "")
        is_positive = (r.get("is_positive") == "1")
        filename = r.get("filename", "")
        path = _resolve_image_path(images_dir, filename, r.get("original_path"))
        
        gt_corners = None
        if "x0" in r and r["x0"]:
            try:
                gt_corners = np.array([
                    [float(r["x0"]), float(r["y0"])],
                    [float(r["x1"]), float(r["y1"])],
                    [float(r["x2"]), float(r["y2"])],
                    [float(r["x3"]), float(r["y3"])],
                ], dtype=np.float64)
            except (ValueError, KeyError):
                gt_corners = None
                
        mask_path = None
        mask_filename = r.get("mask_filename", "").strip()
        if mask_filename and path:
            cand = os.path.join(os.path.dirname(path), mask_filename)
            if os.path.isfile(cand):
                mask_path = cand
                
        img = cv2.imread(path) if path else None
        if img is None:
            rows.append({
                "image_id": image_id, "split": split, "condition": condition,
                "status": "rejected", "reject_reason": "image_not_found",
                "iou": float("nan"), "corner_error_norm": float("nan"),
                "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                "runtime_ms": float("nan"), "resolution": "",
            })
            continue
            
        H, W = img.shape[:2]
        res = run_baseline_pipeline(img)
        
        accepted = (res["status"] == "accepted")
        iou = float("nan")
        corner_error = float("nan")
        if gt_corners is not None:
            if accepted:
                iou = compute_polygon_iou(res["corners"], gt_corners)
                corner_error = compute_corner_error_norm(res["corners"], gt_corners, (H, W))
            else:
                iou = 0.0
                
        precision = recall = f1 = float("nan")
        if mask_path and res["binary_otsu"] is not None:
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_mask = cv2.resize(gt_mask, (res["binary_otsu"].shape[1], res["binary_otsu"].shape[0]), interpolation=cv2.INTER_NEAREST)
                metrics = compute_binarization_metrics(res["binary_otsu"] == 0, gt_mask > 0)
                precision = metrics["precision"]
                recall = metrics["recall"]
                f1 = metrics["f1"]
                
        rows.append({
            "image_id": image_id,
            "split": split,
            "condition": condition,
            "status": res["status"],
            "reject_reason": res["reject_reason"] or "",
            "iou": iou,
            "corner_error_norm": corner_error,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "runtime_ms": res["runtime_ms"],
            "resolution": f"{W}x{H}",
        })
        
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _fmt(row[col]) for col in CSV_COLUMNS})
            
    accepted_count = sum(1 for r in rows if r["status"] == "accepted")
    print(f"Baseline processed {len(rows)} images -> {accepted_count} accepted. Output: {output_csv}")

if __name__ == "__main__":
    import sys
    manifest = sys.argv[1] if len(sys.argv) > 1 else "data/manifest_development_full.csv"
    img_dir = sys.argv[2] if len(sys.argv) > 2 else "data/development"
    out_csv = sys.argv[3] if len(sys.argv) > 3 else "outputs/results_baseline_dev.csv"
    run_baseline_batch(manifest, img_dir, out_csv)
