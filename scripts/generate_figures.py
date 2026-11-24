import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_pipeline import process_single_image, resize_for_work
from src.core.hough import polar_to_cartesian_segment
from src.core.thresholding import otsu_threshold, sauvola_threshold
from src.core.warping import warp_document


def _load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _draw_quad_and_lines(
    img_bgr: np.ndarray, res: dict, draw_lines: bool = True
) -> np.ndarray:
    vis, _ = resize_for_work(img_bgr.copy(), 640)
    h, w = vis.shape[:2]
    if draw_lines:
        for p in res["peaks"][:12]:
            p1, p2 = polar_to_cartesian_segment(p["rho"], p["theta"], w, h)
            cv2.line(vis, p1, p2, (255, 160, 40), 1, cv2.LINE_AA)
    if res["status"] == "accepted" and res["work_corners"] is not None:
        pts = np.rint(res["work_corners"]).astype(np.int32)
        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 220, 0), 3, cv2.LINE_AA)
        for idx, (cx, cy) in enumerate(pts):
            cv2.circle(vis, (int(cx), int(cy)), 7, (0, 0, 255), -1)
            cv2.putText(
                vis,
                str(idx),
                (int(cx) + 8, int(cy) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def export_pipeline_stages_dev01(cfg: dict, out_dir: Path) -> None:
    img = cv2.imread("data/development/dev_01.jpg")
    res = process_single_image(img, cfg)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    axes[0, 0].imshow(cv2.cvtColor(resize_for_work(img, 640)[0], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Input Image (dev_01)")

    axes[0, 1].imshow(res["canny"]["edges"], cmap="gray")
    axes[0, 1].set_title("2. Canny Edge Map")

    acc_log = np.log1p(res["accumulator"].astype(np.float64))
    axes[0, 2].imshow(acc_log, aspect="auto", cmap="inferno")
    t_idx = [p["theta_idx"] for p in res["peaks"]]
    r_idx = [p["rho_idx"] for p in res["peaks"]]
    axes[0, 2].scatter(t_idx, r_idx, c="cyan", s=24, marker="x", linewidths=1.2)
    axes[0, 2].set_title(f"3. Hough Accumulator ({len(res['peaks'])} peaks)")
    axes[0, 2].set_xlabel("Theta index (deg)")
    axes[0, 2].set_ylabel("Rho index (px)")

    overlay = _draw_quad_and_lines(img, res, draw_lines=True)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(
        f"4. Quad (score={res['score']:.2f}, sup={res['edge_support']:.2f})"
    )

    axes[1, 1].imshow(cv2.cvtColor(res["warped_bgr"], cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("5. Bilinear Warped Color")

    axes[1, 2].imshow(res["sauvola"]["binary"], cmap="gray")
    axes[1, 2].set_title(
        f"6. Sauvola Binary (w={cfg['sauvola_window']}, k={cfg['sauvola_k']})"
    )

    for ax in axes.ravel():
        if ax != axes[0, 2]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "pipeline_stages_dev01.png", dpi=180)
    plt.close(fig)


def export_resampling_comparison(cfg: dict, out_dir: Path) -> None:
    img = cv2.imread("data/development/dev_01.jpg")
    res = process_single_image(img, cfg)
    corners = res["corners"]

    wb, _ = warp_document(img, corners, interpolation="bilinear")
    wn, _ = warp_document(img, corners, interpolation="nearest")

    gb = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    gn = cv2.cvtColor(wn, cv2.COLOR_BGR2GRAY)

    # Pick a 120x120 text region with strong gradient activity
    h, w = gb.shape
    ch, cw = min(140, h // 3), min(140, w // 3)
    best_var, best_y, best_x = -1.0, h // 4, w // 4
    for y in range(h // 6, h - ch - h // 8, max(20, ch // 2)):
        for x in range(w // 6, w - cw - w // 8, max(20, cw // 2)):
            patch = gb[y : y + ch, x : x + cw]
            grad_e = float(np.mean(np.abs(np.diff(patch.astype(np.float64), axis=1))))
            if grad_e > best_var:
                best_var, best_y, best_x = grad_e, y, x

    crop_b = gb[best_y : best_y + ch, best_x : best_x + cw]
    crop_n = gn[best_y : best_y + ch, best_x : best_x + cw]
    diff = np.abs(crop_b.astype(np.float64) - crop_n.astype(np.float64))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    axes[0].imshow(crop_n, cmap="gray")
    axes[0].set_title("Nearest-Neighbor Crop")
    axes[1].imshow(crop_b, cmap="gray")
    axes[1].set_title("Bilinear Crop")
    im_d = axes[2].imshow(diff, cmap="magma")
    axes[2].set_title(f"|Bilinear - Nearest| (MAE={np.mean(diff):.2f})")
    fig.colorbar(im_d, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "resampling_bilinear_vs_nearest.png", dpi=180)
    plt.close(fig)


def export_thresholding_comparison(cfg: dict, out_dir: Path) -> None:
    # Real shadow document
    img_real = cv2.imread("data/development/dev_20.jpg")
    res_real = process_single_image(img_real, cfg)
    if res_real["status"] != "accepted":
        for iid in ("dev_10", "dev_25", "dev_30"):
            cand = cv2.imread(f"data/development/{iid}.jpg")
            res_c = process_single_image(cand, cfg)
            if res_c["status"] == "accepted":
                res_real = res_c
                break

    real_gray = res_real["warped_gray"].astype(np.uint8)
    real_otsu = res_real["otsu"]
    real_sauv = res_real["sauvola"]

    # Synthetic shadow+noise document
    syn_gray = cv2.imread("data/synthetic/syn_test_03.jpg", cv2.IMREAD_GRAYSCALE)
    syn_otsu = otsu_threshold(syn_gray)
    syn_sauv = sauvola_threshold(
        syn_gray, window_size=int(cfg["sauvola_window"]), k=float(cfg["sauvola_k"])
    )

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2))
    datasets = [
        ("Real Shadow (dev_20)", real_gray, real_otsu, real_sauv),
        ("Synthetic Shadow (syn_test_03)", syn_gray, syn_otsu, syn_sauv),
    ]
    for r_idx, (label, gray, otsu_r, sauv_r) in enumerate(datasets):
        axes[r_idx, 0].imshow(gray, cmap="gray")
        axes[r_idx, 0].set_title(f"{label}\nGrayscale")
        axes[r_idx, 0].axis("off")

        axes[r_idx, 1].plot(np.arange(256), otsu_r["hist"], color="steelblue", lw=1.2)
        axes[r_idx, 1].axvline(otsu_r["threshold"], color="crimson", ls="--", lw=1.5)
        axes[r_idx, 1].set_title(f"Histogram (Otsu T={otsu_r['threshold']})")
        axes[r_idx, 1].set_xlim(0, 255)
        axes[r_idx, 1].set_yticks([])

        axes[r_idx, 2].imshow(otsu_r["binary"], cmap="gray")
        axes[r_idx, 2].set_title("Otsu Binary")
        axes[r_idx, 2].axis("off")

        im_t = axes[r_idx, 3].imshow(sauv_r["threshold_map"], cmap="viridis")
        axes[r_idx, 3].set_title("Sauvola T(x,y) Map")
        axes[r_idx, 3].axis("off")
        fig.colorbar(im_t, ax=axes[r_idx, 3], fraction=0.046, pad=0.04)

        axes[r_idx, 4].imshow(sauv_r["binary"], cmap="gray")
        axes[r_idx, 4].set_title("Sauvola Binary")
        axes[r_idx, 4].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "thresholding_otsu_vs_sauvola.png", dpi=180)
    plt.close(fig)


def export_ablation_and_baseline_chart(out_dir: Path) -> None:
    configs = [
        ("Custom (Full)", "outputs/results_development.csv"),
        ("Baseline (OpenCV)", "outputs/results_baseline_dev.csv"),
        ("No Gaussian", "outputs/ablation_no_gaussian.csv"),
        ("No Hough Supp.", "outputs/ablation_no_hough_suppression.csv"),
        ("No Edge Support", "outputs/ablation_no_edge_support.csv"),
    ]
    labels, mean_ious, succ_rates, neg_fprs, rt_acc, rt_rej = [], [], [], [], [], []
    for name, path in configs:
        rows = _load_csv(path)
        pos = [r for r in rows if r["condition"] != "khong_co_tai_lieu_hop_le"]
        neg = [r for r in rows if r["condition"] == "khong_co_tai_lieu_hop_le"]
        labels.append(name)
        ious = [float(r["iou"]) for r in pos]
        mean_ious.append(100.0 * float(np.mean(ious)))
        succ_rates.append(100.0 * sum(1 for v in ious if v >= 0.8) / max(1, len(pos)))
        neg_fprs.append(100.0 * sum(r["status"] == "accepted" for r in neg) / max(1, len(neg)))
        acc_t = [float(r["runtime_ms"]) for r in rows if r["status"] == "accepted"]
        rej_t = [float(r["runtime_ms"]) for r in rows if r["status"] == "rejected"]
        rt_acc.append(float(np.mean(acc_t)) if acc_t else 0.0)
        rt_rej.append(float(np.mean(rej_t)) if rej_t else 0.0)

    x = np.arange(len(labels))
    width = 0.26
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.9))

    b1 = ax1.bar(x - width, mean_ious, width, label="Mean Polygon IoU (%)", color="#2b6cb0")
    b2 = ax1.bar(x, succ_rates, width, label="Success @ IoU >= 0.8 (%)", color="#2f855a")
    b3 = ax1.bar(x + width, neg_fprs, width, label="Negative FPR (%)", color="#c53030")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right")
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Development: Mean IoU, Success@0.8 & Negative FPR")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", ls="--", alpha=0.4)
    for bar in list(b1) + list(b2) + list(b3):
        h_val = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h_val + 1.5, f"{h_val:.1f}", ha="center", va="bottom", fontsize=7.2)

    w2 = 0.36
    b4 = ax2.bar(x - w2 / 2, rt_acc, w2, label="Accepted Mean (ms)", color="#2f855a")
    b5 = ax2.bar(x + w2 / 2, rt_rej, w2, label="Rejected Mean (ms)", color="#d69e2e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=18, ha="right")
    ax2.set_ylim(0, max(rt_acc + rt_rej) * 1.2)
    ax2.set_ylabel("CPU Runtime (ms)")
    ax2.set_title("Development: CPU Runtime (Accepted vs Rejected)")
    ax2.legend()
    ax2.grid(axis="y", ls="--", alpha=0.4)
    for bar in list(b4) + list(b5):
        h_val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h_val + 4, f"{h_val:.0f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "chart_ablation_and_baseline.png", dpi=180)
    plt.close(fig)


def export_synthetic_f1_chart(out_dir: Path) -> None:
    rows = _load_csv("outputs/results_synthetic.csv")
    conds = ["nen_deu", "gradient_sang", "bong_kem_nhieu"]
    cond_labels = ["Uniform", "Gradient", "Shadow + Noise"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, split, title in (
        (axes[0], "development", "Synthetic Development (6 images)"),
        (axes[1], "test", "Synthetic Test (6 images)"),
    ):
        otsu_vals = [
            float(np.mean([float(r["f1"]) for r in rows if r["split"] == split and r["condition"] == c and r["method"] == "otsu"]))
            for c in conds
        ]
        sauv_vals = [
            float(np.mean([float(r["f1"]) for r in rows if r["split"] == split and r["condition"] == c and r["method"] == "sauvola"]))
            for c in conds
        ]
        x = np.arange(len(conds))
        w = 0.35
        b1 = ax.bar(x - w / 2, otsu_vals, w, label="Otsu (Global)", color="#dd6b20")
        b2 = ax.bar(x + w / 2, sauv_vals, w, label="Sauvola (Local)", color="#2b6cb0")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels)
        ax.set_ylim(0, 1.15)
        ax.set_title(title)
        ax.set_ylabel("Ink F1 Score")
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.legend(loc="lower left")
        for bar in list(b1) + list(b2):
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_dir / "chart_synthetic_f1.png", dpi=180)
    plt.close(fig)


def export_case_studies(cfg: dict, out_dir: Path) -> None:
    manifest = {r["image_id"]: r for r in _load_csv("data/manifest_development.csv")}
    csv_res = {r["image_id"]: r for r in _load_csv("outputs/results_development.csv")}
    gt_labels = json.load(open("data/labels.json", encoding="utf-8")) if os.path.exists("data/labels.json") else {}

    cases = [
        ("dev_01", "case1_dev01_good.png", "Case 1: Simple Background (Accepted)"),
        ("dev_07", "case2_dev07_oblique.png", "Case 2: Oblique Angle (Rejected)"),
        ("dev_03", "case3_dev03_clutter.png", "Case 3: Rectangular Clutter (Accepted)"),
        ("dev_29", "case4_dev29_lowcontrast.png", "Case 4: Low Contrast Border (Rejected)"),
        ("dev_05", "case5_dev05_shadow.png", "Case 5: Uneven Shadow (Rejected)"),
        ("dev_32", "case6_dev32_negative_fp.png", "Case 6a: Negative Screen (False Positive)"),
        ("dev_33", "case6b_dev33_negative_rej.png", "Case 6b: Negative Torn Paper (Rejected)"),
    ]

    metadata = {}
    for img_id, fname, title_prefix in cases:
        row = manifest[img_id]
        img_path = os.path.join("data/development", row["filename"])
        img = cv2.imread(img_path)
        h_orig, w_orig = img.shape[:2]
        res = process_single_image(img, cfg)

        assert res["status"] == csv_res[img_id]["status"], f"Mismatch for {img_id}"

        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
        work_bgr, scale_to_orig = resize_for_work(img, int(cfg["work_max_dim"]))
        wh, ww = work_bgr.shape[:2]

        axes[0].imshow(cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"{img_id}: Input ({w_orig}x{h_orig})")

        axes[1].imshow(res["canny"]["edges"], cmap="gray")
        edge_ratio = float(np.mean(res["canny"]["edges"] > 0))
        axes[1].set_title(f"Canny Edges ({edge_ratio*100:.2f}% px)")

        acc_log = np.log1p(res["accumulator"].astype(np.float64))
        axes[2].imshow(acc_log, aspect="auto", cmap="inferno")
        axes[2].scatter(
            [p["theta_idx"] for p in res["peaks"]],
            [p["rho_idx"] for p in res["peaks"]],
            c="cyan",
            s=22,
            marker="x",
        )
        max_votes = int(np.max(res["accumulator"])) if res["accumulator"].size > 0 else 0
        axes[2].set_title(f"Hough ({len(res['peaks'])} peaks, max={max_votes})")

        overlay_bgr = img.copy()
        work_vis, _ = resize_for_work(overlay_bgr, int(cfg["work_max_dim"]))
        for p in res["peaks"][:12]:
            p1, p2 = polar_to_cartesian_segment(p["rho"], p["theta"], ww, wh)
            cv2.line(work_vis, p1, p2, (255, 160, 40), 1, cv2.LINE_AA)
        if img_id in gt_labels:
            gt_pts = np.rint(np.array(gt_labels[img_id]["corners"], dtype=np.float64) / scale_to_orig).astype(np.int32)
            cv2.polylines(work_vis, [gt_pts.reshape(-1, 1, 2)], True, (255, 0, 255), 2, cv2.LINE_AA)
        if res["status"] == "accepted" and res["work_corners"] is not None:
            pts = np.rint(res["work_corners"]).astype(np.int32)
            cv2.polylines(work_vis, [pts.reshape(-1, 1, 2)], True, (0, 220, 0), 3, cv2.LINE_AA)
            for idx, (cx, cy) in enumerate(pts):
                cv2.circle(work_vis, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                cv2.putText(work_vis, str(idx), (int(cx) + 7, int(cy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        axes[3].imshow(cv2.cvtColor(work_vis, cv2.COLOR_BGR2RGB))

        iou_str = csv_res[img_id]["iou"]
        ce_str = csv_res[img_id]["corner_error_norm"]
        st_lbl = res["status"].upper()
        if res["reject_reason"]:
            st_lbl += f": {res['reject_reason']} (IoU={iou_str})"
        elif iou_str != "N/A":
            st_lbl += f" (IoU={iou_str})"
        else:
            st_lbl += f" (sup={res['edge_support']:.2f})"
        axes[3].set_title(st_lbl, fontsize=9.2)

        for idx_ax, ax in enumerate(axes):
            if idx_ax != 2:
                ax.axis("off")

        fig.suptitle(title_prefix + (" [Green=Pred, Magenta=GT]" if img_id in gt_labels else ""), fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=180)
        plt.close(fig)

        metadata[img_id] = {
            "figure": fname,
            "condition": row["condition"],
            "status": res["status"],
            "reject_reason": res["reject_reason"],
            "iou": iou_str,
            "corner_error_norm": ce_str,
            "edge_pixel_ratio": round(edge_ratio, 4),
            "max_hough_votes": max_votes,
            "num_peaks": len(res["peaks"]),
            "edge_support": round(float(res["edge_support"]), 4),
            "area_ratio": round(float(res["area_ratio"]), 4),
            "score": round(float(res["score"]), 4),
            "runtime_ms": float(csv_res[img_id]["runtime_ms"]),
            "resolution": csv_res[img_id]["resolution"],
            "work_resolution": f"{ww}x{wh}",
        }

    with open(out_dir / "cases_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported {len(cases)} diagnostic case figures + cases_metadata.json")


def main() -> None:
    np.random.seed(42)
    with open("outputs/frozen_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    export_pipeline_stages_dev01(cfg, out_dir)
    export_resampling_comparison(cfg, out_dir)
    export_thresholding_comparison(cfg, out_dir)
    export_ablation_and_baseline_chart(out_dir)
    export_synthetic_f1_chart(out_dir)
    export_case_studies(cfg, out_dir)
    print("All Phase 2 figures generated successfully.")


if __name__ == "__main__":
    main()
