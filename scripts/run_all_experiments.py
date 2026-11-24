import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from run_pipeline import (
    DEFAULT_CONFIG,
    ink_prf1,
    load_labels,
    normalized_corner_error,
    polygon_iou,
    process_single_image,
    run_batch,
)
from src.baseline.baseline import run_baseline_batch
from src.core.thresholding import otsu_threshold, sauvola_threshold


def _load_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _eval_real_dev(images_cache: list[tuple[dict, np.ndarray]], cfg: dict, labels: dict) -> dict:
    pos_acc, pos_tot, neg_fp, neg_tot = 0, 0, 0, 0
    ious = []
    ces = []
    runtimes = []
    for row, img in images_cache:
        res = process_single_image(img, cfg)
        runtimes.append(res["runtime_ms"])
        acc = res["status"] == "accepted"
        if row["is_positive"] == "1":
            pos_tot += 1
            pos_acc += int(acc)
            gt_entry = labels.get(row["image_id"])
            gt_corners = np.array(gt_entry["corners"], dtype=np.float32) if gt_entry else None
            if acc and gt_corners is not None:
                iou = polygon_iou(res["corners"], gt_corners)
                ious.append(iou)
                ce = normalized_corner_error(res["corners"], gt_corners, img.shape)
                if ce is not None:
                    ces.append(ce)
            else:
                ious.append(0.0)
        else:
            neg_tot += 1
            neg_fp += int(acc)
    succ_08 = sum(1 for v in ious if v >= 0.8)
    return {
        "pos_accepted": pos_acc,
        "pos_total": pos_tot,
        "pos_rate": pos_acc / max(1, pos_tot),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "succ_08": succ_08,
        "succ_08_rate": succ_08 / max(1, pos_tot),
        "mean_corner_err": float(np.mean(ces)) if ces else 0.0,
        "neg_fp": neg_fp,
        "neg_total": neg_tot,
        "neg_fpr": neg_fp / max(1, neg_tot),
        "mean_runtime_ms": float(np.mean(runtimes)) if runtimes else 0.0,
    }


def run_canny_sweep_dev(images_cache: list[tuple[dict, np.ndarray]], base_cfg: dict, labels: dict) -> None:
    rows = []
    for sigma in (0.8, 1.4, 2.2):
        cfg = {**base_cfg, "sigma": sigma, "low_ratio": 0.06, "high_ratio": 0.16}
        m = _eval_real_dev(images_cache, cfg, labels)
        rows.append({"param_name": "sigma", "param_value": f"{sigma}", **m})
    for lo, hi in ((0.04, 0.12), (0.06, 0.16), (0.10, 0.25)):
        cfg = {**base_cfg, "sigma": 1.4, "low_ratio": lo, "high_ratio": hi}
        m = _eval_real_dev(images_cache, cfg, labels)
        rows.append({"param_name": "thresholds", "param_value": f"({lo},{hi})", **m})

    out_path = Path("outputs/sweep_canny_dev.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "param_name",
        "param_value",
        "pos_accepted",
        "pos_total",
        "pos_rate",
        "mean_iou",
        "succ_08",
        "succ_08_rate",
        "mean_corner_err",
        "neg_fp",
        "neg_total",
        "neg_fpr",
        "mean_runtime_ms",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r_fmt = dict(r)
            r_fmt["pos_rate"] = f"{r['pos_rate']:.4f}"
            r_fmt["mean_iou"] = f"{r['mean_iou']:.4f}"
            r_fmt["succ_08_rate"] = f"{r['succ_08_rate']:.4f}"
            r_fmt["mean_corner_err"] = f"{r['mean_corner_err']:.4f}"
            r_fmt["neg_fpr"] = f"{r['neg_fpr']:.4f}"
            r_fmt["mean_runtime_ms"] = f"{r['mean_runtime_ms']:.2f}"
            w.writerow(r_fmt)
    print(f"Saved {out_path}")


def run_hough_quad_sweep_dev(images_cache: list[tuple[dict, np.ndarray]], base_cfg: dict, labels: dict) -> None:
    rows = []
    for hr in (0.18, 0.24, 0.30):
        for use_sup in (False, True):
            cfg = {**base_cfg, "hough_thresh_ratio": hr, "use_edge_support": use_sup}
            m = _eval_real_dev(images_cache, cfg, labels)
            rows.append(
                {
                    "hough_thresh_ratio": f"{hr:.2f}",
                    "scoring_mode": "edge_support+area" if use_sup else "area_only",
                    "pos_accepted": m["pos_accepted"],
                    "pos_total": m["pos_total"],
                    "pos_rate": f"{m['pos_rate']:.4f}",
                    "mean_iou": f"{m['mean_iou']:.4f}",
                    "succ_08": m["succ_08"],
                    "succ_08_rate": f"{m['succ_08_rate']:.4f}",
                    "mean_corner_err": f"{m['mean_corner_err']:.4f}",
                    "neg_fp": m["neg_fp"],
                    "neg_total": m["neg_total"],
                    "neg_fpr": f"{m['neg_fpr']:.4f}",
                    "mean_runtime_ms": f"{m['mean_runtime_ms']:.2f}",
                }
            )
    out_path = Path("outputs/sweep_hough_quad_dev.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_path}")


def run_thresholding_sweep_dev(syn_rows: list[dict]) -> tuple[int, float]:
    dev_syn = [r for r in syn_rows if r["split"] == "development"]
    loaded = []
    for r in dev_syn:
        img = cv2.imread(os.path.join("data/synthetic", r["filename"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join("data/synthetic", r["mask_filename"]), cv2.IMREAD_GRAYSCALE)
        loaded.append((r, img, mask))

    records = []

    def _eval_method(method: str, ws: int | str, k_val: float | str):
        by_cond: dict[str, list[tuple[float, float, float]]] = {}
        all_scores = []
        for r, img, mask in loaded:
            if method == "otsu":
                res = otsu_threshold(img)
            else:
                res = sauvola_threshold(img, window_size=int(ws), k=float(k_val))
            p, rec, f1 = ink_prf1(res["binary"], mask)
            by_cond.setdefault(r["condition"], []).append((p, rec, f1))
            all_scores.append((p, rec, f1))
        for cond, vals in list(by_cond.items()) + [("all_dev", all_scores)]:
            mp = float(np.mean([v[0] for v in vals]))
            mr = float(np.mean([v[1] for v in vals]))
            mf = float(np.mean([v[2] for v in vals]))
            records.append(
                {
                    "method": method,
                    "window_size": ws,
                    "k": k_val,
                    "condition": cond,
                    "mean_precision": f"{mp:.4f}",
                    "mean_recall": f"{mr:.4f}",
                    "mean_f1": f"{mf:.4f}",
                }
            )
        return float(np.mean([v[2] for v in all_scores]))

    _eval_method("otsu", "N/A", "N/A")
    best_ws, best_k, best_f1 = 25, 0.2, -1.0
    for ws in (15, 25, 41):
        f1 = _eval_method("sauvola", ws, 0.2)
        if f1 > best_f1:
            best_f1, best_ws = f1, ws
    for k_val in (0.1, 0.2, 0.35):
        f1 = _eval_method("sauvola", best_ws, k_val)
        if f1 > best_f1:
            best_f1, best_k = f1, k_val

    out_path = Path("outputs/sweep_thresholding_dev.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "window_size",
                "k",
                "condition",
                "mean_precision",
                "mean_recall",
                "mean_f1",
            ],
        )
        w.writeheader()
        w.writerows(records)
    print(f"Saved {out_path} (best Sauvola Dev: w={best_ws}, k={best_k}, F1={best_f1:.4f})")
    return best_ws, best_k


def run_synthetic_eval(syn_rows: list[dict], cfg: dict) -> None:
    custom_out = []
    base_out = []
    for r in syn_rows:
        img = cv2.imread(os.path.join("data/synthetic", r["filename"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join("data/synthetic", r["mask_filename"]), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        res_str = f"{w}x{h}"

        for m_name, fn in (
            ("otsu", lambda im: otsu_threshold(im)["binary"]),
            (
                "sauvola",
                lambda im: sauvola_threshold(
                    im, window_size=int(cfg["sauvola_window"]), k=float(cfg["sauvola_k"])
                )["binary"],
            ),
        ):
            t0 = time.perf_counter()
            binary = fn(img)
            rt = (time.perf_counter() - t0) * 1000.0
            p, rec, f1 = ink_prf1(binary, mask)
            custom_out.append(
                {
                    "image_id": r["image_id"],
                    "split": r["split"],
                    "condition": r["condition"],
                    "method": m_name,
                    "precision": f"{p:.4f}",
                    "recall": f"{rec:.4f}",
                    "f1": f"{f1:.4f}",
                    "runtime_ms": f"{rt:.2f}",
                    "resolution": res_str,
                }
            )

        for m_name, fn in (
            (
                "cv2_otsu",
                lambda im: cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            ),
            (
                "skimage_sauvola",
                lambda im: np.where(
                    im > threshold_sauvola(im, window_size=int(cfg["sauvola_window"]), k=float(cfg["sauvola_k"])),
                    np.uint8(255),
                    np.uint8(0),
                ),
            ),
        ):
            t0 = time.perf_counter()
            binary = fn(img)
            rt = (time.perf_counter() - t0) * 1000.0
            p, rec, f1 = ink_prf1(binary, mask)
            base_out.append(
                {
                    "image_id": r["image_id"],
                    "split": r["split"],
                    "condition": r["condition"],
                    "method": m_name,
                    "precision": f"{p:.4f}",
                    "recall": f"{rec:.4f}",
                    "f1": f"{f1:.4f}",
                    "runtime_ms": f"{rt:.2f}",
                    "resolution": res_str,
                }
            )

    cols = ["image_id", "split", "condition", "method", "precision", "recall", "f1", "runtime_ms", "resolution"]
    for path, data in (
        ("outputs/results_synthetic.csv", custom_out),
        ("outputs/results_baseline_synthetic.csv", base_out),
    ):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w_csv = csv.DictWriter(f, fieldnames=cols)
            w_csv.writeheader()
            w_csv.writerows(data)
        print(f"Saved {path}")


def main() -> None:
    np.random.seed(42)
    dev_manifest = _load_manifest("data/manifest_development.csv")
    dev_cache = [
        (r, cv2.imread(os.path.join("data/development", r["filename"]))) for r in dev_manifest
    ]
    syn_rows = _load_manifest("data/synthetic/manifest_synthetic.csv")

    base_cfg = {
        **DEFAULT_CONFIG,
        "sigma": 1.4,
        "low_ratio": 0.06,
        "high_ratio": 0.16,
        "hough_thresh_ratio": 0.24,
        "min_area_ratio": 0.14,
        "min_edge_support": 0.72,
        "min_side_support": 0.50,
    }

    print("=== Step 1: Development Parameter Sweeps (No Test Leakage) ===")
    labels = load_labels("data/labels.json")
    run_canny_sweep_dev(dev_cache, base_cfg, labels)
    run_hough_quad_sweep_dev(dev_cache, base_cfg, labels)
    best_ws, best_k = run_thresholding_sweep_dev(syn_rows)

    frozen_cfg = {
        **base_cfg,
        "sauvola_window": best_ws,
        "sauvola_k": best_k,
    }
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open("outputs/frozen_config.json", "w", encoding="utf-8") as f:
        json.dump(frozen_cfg, f, indent=2)
    print("Froze configuration -> outputs/frozen_config.json")

    baseline_cfg = {
        "seed": 42,
        "work_max_dim": int(frozen_cfg["work_max_dim"]),
        "gaussian_kernel": [9, 9],
        "gaussian_sigma": float(frozen_cfg["sigma"]),
        "canny_method": "cv2.Canny",
        "canny_low": 50,
        "canny_high": 140,
        "hough_method": "cv2.HoughLines",
        "hough_rho_res": 1.0,
        "hough_theta_res_deg": 1.0,
        "hough_threshold": 60,
        "top_k_lines": int(frozen_cfg["top_k_lines"]),
        "quad_selection": "find_document_quad(use_edge_support=True)",
        "min_area_ratio": float(frozen_cfg["min_area_ratio"]),
        "max_area_ratio": float(frozen_cfg["max_area_ratio"]),
        "min_edge_support": float(frozen_cfg["min_edge_support"]),
        "min_side_support": float(frozen_cfg["min_side_support"]),
        "min_angle_deg": float(frozen_cfg["min_angle_deg"]),
        "max_angle_deg": float(frozen_cfg["max_angle_deg"]),
        "otsu_method": "cv2.threshold(THRESH_BINARY + THRESH_OTSU)",
        "sauvola_method": "skimage.filters.threshold_sauvola",
        "sauvola_window": int(frozen_cfg["sauvola_window"]),
        "sauvola_k": float(frozen_cfg["sauvola_k"]),
        "warp_max_dim": int(frozen_cfg["warp_max_dim"]),
    }
    with open("outputs/baseline_config.json", "w", encoding="utf-8") as f:
        json.dump(baseline_cfg, f, indent=2)
    print("Saved -> outputs/baseline_config.json")

    ablation_cfgs = {
        "no_gaussian": {
            "description": "Disable Gaussian smoothing before Sobel in Canny; keep top_k_lines and all other params frozen.",
            "config": {**frozen_cfg, "use_gaussian": False},
        },
        "no_hough_suppression": {
            "description": "Disable duplicate peak suppression in Hough; keep top_k_lines and all other params frozen.",
            "config": {**frozen_cfg, "hough_suppress": False},
        },
        "no_edge_support": {
            "description": "Score quad candidates by area only without edge_support; keep top_k_lines and all other params frozen.",
            "config": {**frozen_cfg, "use_edge_support": False},
        },
    }
    with open("outputs/ablation_configs.json", "w", encoding="utf-8") as f:
        json.dump(ablation_cfgs, f, indent=2)
    print("Saved -> outputs/ablation_configs.json")

    rep_ids = {"dev_01", "dev_03", "dev_05", "dev_07", "dev_16", "dev_29", "dev_32", "test_01"}
    print("=== Step 2: Run Frozen Pipeline on Dev, Test, Synthetic ===")
    run_batch(
        "data/manifest_development.csv",
        "data/development",
        "outputs/results_development.csv",
        "development",
        frozen_cfg,
        labels_path="data/labels.json",
        seed=42,
        save_intermediates_dir="outputs/intermediates",
        intermediate_ids=rep_ids,
    )
    run_batch(
        "data/manifest_test.csv",
        "data/test",
        "outputs/results_test.csv",
        "test",
        frozen_cfg,
        labels_path="data/labels.json",
        seed=42,
        save_intermediates_dir="outputs/intermediates",
        intermediate_ids=rep_ids,
    )
    run_synthetic_eval(syn_rows, frozen_cfg)

    print("=== Step 3: Run Baseline on Dev & Test ===")
    run_baseline_batch(
        "data/manifest_development.csv",
        "data/development",
        "outputs/results_baseline_dev.csv",
        "development",
        labels_path="data/labels.json",
        seed=42,
    )
    run_baseline_batch(
        "data/manifest_test.csv",
        "data/test",
        "outputs/results_baseline_test.csv",
        "test",
        labels_path="data/labels.json",
        seed=42,
    )

    print("=== Step 4: Run 3 Mandatory Ablations on Dev (identical top_k_lines) ===")
    ablations = [
        ("outputs/ablation_no_gaussian.csv", {**frozen_cfg, "use_gaussian": False}),
        ("outputs/ablation_no_hough_suppression.csv", {**frozen_cfg, "hough_suppress": False}),
        ("outputs/ablation_no_edge_support.csv", {**frozen_cfg, "use_edge_support": False}),
    ]
    for out_csv, ab_cfg in ablations:
        run_batch(
            "data/manifest_development.csv",
            "data/development",
            out_csv,
            "development",
            ab_cfg,
            labels_path="data/labels.json",
            seed=42,
        )
        print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
