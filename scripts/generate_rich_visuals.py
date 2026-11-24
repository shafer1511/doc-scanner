import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from run_pipeline import process_single_image, resize_for_work
from src.core.canny import canny_edge_detection, to_grayscale
from src.core.hough import find_hough_peaks, hough_transform, polar_to_cartesian_segment
from src.core.quadrilateral import find_document_quad
from src.core.thresholding import otsu_threshold, sauvola_threshold
from src.core.warping import warp_document


def _load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _draw_overlay(img_bgr: np.ndarray, res: dict, draw_lines: bool = True) -> np.ndarray:
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


def _select_accepted_by_group(
    manifest_path: str, csv_path: str, images_dir: str, preferred_id: str, doc_group: str
) -> tuple[str, str]:
    manifest = {r["image_id"]: r for r in _load_csv(manifest_path)}
    status_map = {r["image_id"]: r["status"] for r in _load_csv(csv_path)}
    if preferred_id in manifest and status_map.get(preferred_id) == "accepted":
        return preferred_id, os.path.join(images_dir, manifest[preferred_id]["filename"])
    for iid, row in manifest.items():
        if row["document_name"] == doc_group and status_map.get(iid) == "accepted":
            return iid, os.path.join(images_dir, row["filename"])
    return preferred_id, os.path.join(images_dir, manifest[preferred_id]["filename"])


def build_architecture_diagram(cfg: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 6.2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(
        7.75,
        5.82,
        "DOCUMENT SCANNER PIPELINE ARCHITECTURE (src/core/)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1a202c",
    )
    ax.text(
        7.75,
        5.48,
        f"CPU Pure-NumPy/Primitive Pipeline | seed=42 | work_max_dim={cfg['work_max_dim']}px | No OpenCV high-level Canny/Hough/Threshold",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#4a5568",
    )

    boxes = [
        (0.25, 2.7, 1.85, 2.25, "#edf2f7", "#4a5568", "0. INPUT", "BGR Image\n(H x W x 3)\nResize <= 640px\nfor Canny/Hough"),
        (2.45, 2.7, 2.25, 2.25, "#ebf8ff", "#2b6cb0", "1. CANNY EDGE\n(canny.py)", f"Gray + Gaussian\n(sigma={cfg['sigma']})\nSobel + NMS (4 dir)\nDouble Thresh\n8-conn BFS"),
        (5.05, 2.7, 2.25, 2.25, "#e6fffa", "#2c7a7b", "2. HOUGH LINE\n(hough.py)", f"np.bincount (rho,theta)\nmax_votes * {cfg['hough_thresh_ratio']}\ntop_k={cfg['top_k_lines']}\nCircular NMS\n({cfg['rho_supp']}px, {cfg['theta_supp_deg']}deg)"),
        (7.65, 2.7, 2.35, 2.25, "#fefcbf", "#b7791f", "3. QUAD SELECT\n(quadrilateral.py)", f"Opposite pairs\nnp.linalg.solve\nConvex & Angle\nArea >= {cfg['min_area_ratio']}\nEdgeSup >= {cfg['min_edge_support']}"),
        (10.35, 2.7, 2.25, 2.25, "#f0fff4", "#2f855a", "4. HOMOGRAPHY\n(warping.py)", "Scale corners\nto original res\nPerspective Matrix\nBilinear Warp\n+ bilinear_sample"),
        (12.95, 2.7, 2.30, 2.25, "#faf5ff", "#6b46c1", "5. BINARIZATION\n(thresholding.py)", f"Global Otsu\n(256-bin variance)\nLocal Sauvola\n(w={cfg['sauvola_window']}, k={cfg['sauvola_k']})\nOutput 0/255"),
    ]

    for x, y, w, h, fc, ec, title, body in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", facecolor=fc, edgecolor=ec, linewidth=1.8
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="top", fontsize=9.2, fontweight="bold", color=ec)
        ax.plot([x + 0.12, x + w - 0.12], [y + h - 0.72, y + h - 0.72], color=ec, lw=0.8, alpha=0.5)
        ax.text(x + w / 2, y + (h - 0.72) / 2, body, ha="center", va="center", fontsize=8.1, color="#2d3748", linespacing=1.3)

    # Arrows between main stages
    arrow_xs = [(2.12, 2.43), (4.72, 5.03), (7.32, 7.63), (10.02, 10.33), (12.62, 12.93)]
    for x1, x2 in arrow_xs:
        arr = FancyArrowPatch((x1, 3.82), (x2, 3.82), arrowstyle="-|>", mutation_scale=14, lw=1.8, color="#2d3748")
        ax.add_patch(arr)

    ax.text(
        10.18,
        4.08,
        "ACCEPTED",
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color="#2f855a",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"),
    )

    # Early-Exit Rejected branch under Stage 3
    rej_box = FancyBboxPatch(
        (7.55, 0.45), 2.55, 1.35, boxstyle="round,pad=0.06,rounding_size=0.12", facecolor="#fff5f5", edgecolor="#c53030", linewidth=1.8
    )
    ax.add_patch(rej_box)
    ax.text(8.825, 1.62, "EARLY EXIT: REJECTED", ha="center", va="top", fontsize=9.0, fontweight="bold", color="#c53030")
    ax.text(
        8.825,
        0.95,
        "Skip Warp & Thresholding\nLog reject_reason + IoU=0\n(insufficient_lines, short_side,\nout_of_bounds, low_edge_support)",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#742a2a",
        linespacing=1.25,
    )
    arr_rej = FancyArrowPatch((8.825, 2.68), (8.825, 1.84), arrowstyle="-|>", mutation_scale=14, lw=1.8, color="#c53030")
    ax.add_patch(arr_rej)
    ax.text(8.98, 2.28, "Invalid / Negative", ha="left", va="center", fontsize=7.8, fontweight="bold", color="#c53030")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_pipeline_architecture.png", dpi=200)
    plt.close(fig)


def build_dataset_overview(out_dir: Path) -> None:
    gt_labels = json.load(open("data/labels.json", encoding="utf-8")) if os.path.exists("data/labels.json") else {}
    samples = [
        ("dev_01", "data/development/dev_01.jpg", "dev_01: Sach HP Book\n(GT 4-corner magenta)"),
        ("dev_06", "data/development/dev_06.jpg", "dev_06: De thi XSTK\n(GT 4-corner magenta)"),
        ("dev_11", "data/development/dev_11.jpg", "dev_11: Vo ghi viet tay\n(GT 4-corner magenta)"),
        ("dev_16", "data/development/dev_16.jpg", "dev_16: So tay Leaf\n(GT 4-corner magenta)"),
        ("dev_21", "data/development/dev_21.jpg", "dev_21: Giay to dau do\n(GT 4-corner magenta)"),
        ("dev_26", "data/development/dev_26.jpg", "dev_26: The sinh vien\n(GT 4-corner magenta)"),
        ("test_01", "data/test/test_01.jpg", "test_01: Ve xe buyt (Test)\n(GT 4-corner magenta)"),
        ("dev_31", "data/development/dev_31.jpg", "dev_31: Gach lat san\n(Anh am / Negative)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.8))
    for ax, (iid, path, title) in zip(axes.ravel(), samples):
        im = cv2.imread(path)
        im_small, scale_to_orig = resize_for_work(im, 480)
        if iid in gt_labels:
            pts = np.rint(np.array(gt_labels[iid]["corners"], dtype=np.float64) / scale_to_orig).astype(np.int32)
            cv2.polylines(im_small, [pts.reshape(-1, 1, 2)], True, (255, 0, 255), 3, cv2.LINE_AA)
            for cx, cy in pts:
                cv2.circle(im_small, (int(cx), int(cy)), 5, (0, 255, 255), -1)
        ax.imshow(cv2.cvtColor(im_small, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9.2)
        ax.axis("off")
    fig.suptitle("Representative Dataset Samples with Ground-Truth 4-Corner Labels (data/labels.json)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_dataset_overview.png", dpi=180)
    plt.close(fig)


def build_canny_sweep_visual(cfg: dict, out_dir: Path) -> dict:
    img = cv2.imread("data/development/dev_01.jpg")
    work, _ = resize_for_work(img, int(cfg["work_max_dim"]))
    specs = [
        ("sigma=0.8 (0.06, 0.16)", 0.8, 0.06, 0.16),
        ("sigma=1.4 (0.06, 0.16) [Selected]", 1.4, 0.06, 0.16),
        ("sigma=2.2 (0.06, 0.16)", 2.2, 0.06, 0.16),
        ("sigma=1.4 (0.04, 0.12)", 1.4, 0.04, 0.12),
        ("sigma=1.4 (0.06, 0.16) [Selected]", 1.4, 0.06, 0.16),
        ("sigma=1.4 (0.10, 0.25)", 1.4, 0.10, 0.25),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6))
    stats = []
    for ax, (lbl, sig, lo, hi) in zip(axes.ravel(), specs):
        res = canny_edge_detection(work, sigma=sig, low_ratio=lo, high_ratio=hi, use_gaussian=True)
        edges = res["edges"]
        px_cnt = int(np.sum(edges > 0))
        dens = 100.0 * px_cnt / float(edges.size)
        ax.imshow(edges, cmap="gray")
        ax.set_title(f"{lbl}\n{px_cnt:,} edge px ({dens:.2f}%)", fontsize=9.0)
        ax.axis("off")
        stats.append({"label": lbl, "sigma": sig, "low": lo, "high": hi, "edge_pixels": px_cnt, "density_pct": round(dens, 2)})

    fig.suptitle("Canny Sensitivity on dev_01 (640x480): Gaussian Sigma & Double-Threshold Ratios", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_canny_sweep_visual.png", dpi=180)
    plt.close(fig)
    return {"canny_visual_stats": stats}


def build_ablation_visual(cfg: dict, out_dir: Path) -> dict:
    # Find a positive Dev image where Full accepts and No Hough Suppression rejects (or degrades),
    # and a negative Dev image where Full rejects and No Edge Support falsely accepts.
    full_rows = {r["image_id"]: r for r in _load_csv("outputs/results_development.csv")}
    no_hough_rows = {r["image_id"]: r for r in _load_csv("outputs/ablation_no_hough_suppression.csv")}
    no_edge_rows = {r["image_id"]: r for r in _load_csv("outputs/ablation_no_edge_support.csv")}
    manifest = {r["image_id"]: r for r in _load_csv("data/manifest_development.csv")}

    pos_id = "dev_03"
    for iid, r in full_rows.items():
        if r["condition"] != "khong_co_tai_lieu_hop_le" and r["status"] == "accepted" and no_hough_rows[iid]["status"] == "rejected":
            pos_id = iid
            break

    neg_id = "dev_33"
    for iid, r in full_rows.items():
        if r["condition"] == "khong_co_tai_lieu_hop_le" and r["status"] == "rejected" and no_edge_rows[iid]["status"] == "accepted":
            neg_id = iid
            break

    variants = [
        ("Full Pipeline", cfg),
        ("No Gaussian", {**cfg, "use_gaussian": False}),
        ("No Hough Supp.", {**cfg, "hough_suppress": False}),
        ("No Edge Support", {**cfg, "use_edge_support": False}),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.4))
    meta = {"pos_id": pos_id, "neg_id": neg_id, "results": {}}
    for row_idx, iid in enumerate((pos_id, neg_id)):
        img = cv2.imread(os.path.join("data/development", manifest[iid]["filename"]))
        meta["results"][iid] = {}
        for col_idx, (vname, vcfg) in enumerate(variants):
            res = process_single_image(img, vcfg)
            vis = _draw_overlay(img, res, draw_lines=True)
            ax = axes[row_idx, col_idx]
            ax.imshow(vis)
            st = res["status"].upper()
            extra = f"sup={res['edge_support']:.2f}" if res["status"] == "accepted" else res["reject_reason"]
            ax.set_title(f"{iid} | {vname}\n{st} ({extra})", fontsize=8.6)
            ax.axis("off")
            meta["results"][iid][vname] = {"status": res["status"], "reason": res["reject_reason"], "edge_support": round(float(res["edge_support"]), 4), "area_ratio": round(float(res["area_ratio"]), 4)}

    fig.suptitle(f"Ablation Visual Impact: Positive Sample ({pos_id}) & Negative Control ({neg_id})", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ablation_visual.png", dpi=180)
    plt.close(fig)
    return meta


def build_multidoc_stages(cfg: dict, out_dir: Path) -> dict:
    specs = [
        ("fig_stages_thesv_dev26.png", "data/manifest_development.csv", "outputs/results_development.csv", "data/development", "dev_26", "the_sinh_vien", "Student ID Card"),
        ("fig_stages_sign_dev21.png", "data/manifest_development.csv", "outputs/results_development.csv", "data/development", "dev_21", "giay_to_dau_do", "Stamped Document"),
        ("fig_stages_bus_test01.png", "data/manifest_test.csv", "outputs/results_test.csv", "data/test", "test_01", "ve_xe_buyt", "Bus Ticket (Test Split)"),
        ("fig_stages_leaf_dev16.png", "data/manifest_development.csv", "outputs/results_development.csv", "data/development", "dev_16", "so_leaf", "Leaf Notebook"),
    ]
    out_meta = {}
    for fname, man_p, csv_p, img_dir, pref_id, group, title_grp in specs:
        chosen_id, img_path = _select_accepted_by_group(man_p, csv_p, img_dir, pref_id, group)
        csv_map = {r["image_id"]: r for r in _load_csv(csv_p)}
        img = cv2.imread(img_path)
        res = process_single_image(img, cfg)

        fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2))
        work_bgr, _ = resize_for_work(img, int(cfg["work_max_dim"]))
        gray_w = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)

        axes[0, 0].imshow(gray_w, cmap="gray")
        axes[0, 0].set_title(f"1. Grayscale ({chosen_id})")

        axes[0, 1].imshow(res["canny"]["edges"], cmap="gray")
        axes[0, 1].set_title("2. Custom Canny Edge Map")

        vis_lines = work_bgr.copy()
        wh, ww = vis_lines.shape[:2]
        for p in res["peaks"][:12]:
            p1, p2 = polar_to_cartesian_segment(p["rho"], p["theta"], ww, wh)
            cv2.line(vis_lines, p1, p2, (0, 165, 255), 2, cv2.LINE_AA)
        axes[0, 2].imshow(cv2.cvtColor(vis_lines, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f"3. Top Hough Lines ({len(res['peaks'])})")

        iou_v = csv_map[chosen_id]["iou"]
        ce_v = csv_map[chosen_id]["corner_error_norm"]
        axes[1, 0].imshow(_draw_overlay(img, res, draw_lines=False))
        axes[1, 0].set_title(f"4. Quad (IoU={iou_v}, CE={ce_v})")

        axes[1, 1].imshow(cv2.cvtColor(res["warped_bgr"], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("5. Bilinear Warped Color")

        axes[1, 2].imshow(res["sauvola"]["binary"], cmap="gray")
        axes[1, 2].set_title(f"6. Sauvola Binary (w={cfg['sauvola_window']}, k={cfg['sauvola_k']})")

        for ax in axes.ravel():
            ax.axis("off")

        fig.suptitle(f"6-Stage Pipeline Progression: {title_grp} ({chosen_id})", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=180)
        plt.close(fig)

        out_meta[fname] = {
            "image_id": chosen_id,
            "group": group,
            "status": res["status"],
            "iou": iou_v,
            "corner_error_norm": ce_v,
            "edge_support": round(float(res["edge_support"]), 4),
            "area_ratio": round(float(res["area_ratio"]), 4),
            "score": round(float(res["score"]), 4),
        }
    return out_meta


def _find_text_crop(gray: np.ndarray, otsu_bin: np.ndarray, crop_h: int = 180, crop_w: int = 320) -> tuple[int, int]:
    h, w = gray.shape
    ch = min(crop_h, h // 2)
    cw = min(crop_w, w // 2)
    best_score, best_y, best_x = -1.0, h // 4, w // 4
    for y in range(h // 8, h - ch - h // 8, max(25, ch // 3)):
        for x in range(w // 8, w - cw - w // 8, max(25, cw // 3)):
            patch = gray[y : y + ch, x : x + cw].astype(np.float64)
            gx = np.mean(np.abs(np.diff(patch, axis=1)))
            gy = np.mean(np.abs(np.diff(patch, axis=0)))
            score = gx + gy
            if score > best_score:
                best_score, best_y, best_x = score, y, x
    return best_y, best_x


def build_crop_comparisons(cfg: dict, out_dir: Path) -> dict:
    thesv_id, thesv_p = _select_accepted_by_group("data/manifest_development.csv", "outputs/results_development.csv", "data/development", "dev_26", "the_sinh_vien")
    bus_id, bus_p = _select_accepted_by_group("data/manifest_test.csv", "outputs/results_test.csv", "data/test", "test_01", "ve_xe_buyt")
    shad_id, shad_p = _select_accepted_by_group("data/manifest_development.csv", "outputs/results_development.csv", "data/development", "dev_20", "de_thi_xstk")
    # Prefer an accepted anh_sang_khong_deu_bong_do image
    dev_rows = _load_csv("outputs/results_development.csv")
    man_dev = {r["image_id"]: r for r in _load_csv("data/manifest_development.csv")}
    for r in dev_rows:
        if r["condition"] == "anh_sang_khong_deu_bong_do" and r["status"] == "accepted":
            shad_id = r["image_id"]
            shad_p = os.path.join("data/development", man_dev[shad_id]["filename"])
            break

    row_specs = [
        (thesv_id, thesv_p, "Student Card (Watermark/Gradient)"),
        (bus_id, bus_p, "Bus Ticket (Dot-Matrix Ink)"),
        (shad_id, shad_p, "Shadow Document (Uneven Light)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 8.8))
    crop_meta = {}
    shadow_crop_gray = None

    for r_i, (iid, path, lbl) in enumerate(row_specs):
        img = cv2.imread(path)
        res = process_single_image(img, cfg)
        wg = res["warped_gray"].astype(np.uint8)
        ob = res["otsu"]["binary"]
        sb = res["sauvola"]["binary"]
        ot = res["otsu"]["threshold"]
        diff_pct = 100.0 * float(np.mean(ob != sb))

        y0, x0 = _find_text_crop(wg, ob, crop_h=min(220, wg.shape[0] // 2), crop_w=min(420, wg.shape[1] // 2))
        ch = min(220, wg.shape[0] - y0)
        cw = min(420, wg.shape[1] - x0)

        cg = wg[y0 : y0 + ch, x0 : x0 + cw]
        co = ob[y0 : y0 + ch, x0 : x0 + cw]
        cs = sb[y0 : y0 + ch, x0 : x0 + cw]
        if r_i == 2:
            shadow_crop_gray = wg

        axes[r_i, 0].imshow(cg, cmap="gray")
        axes[r_i, 0].set_title(f"{iid}: {lbl}\nWarped Grayscale Crop", fontsize=8.8)

        axes[r_i, 1].imshow(co, cmap="gray")
        axes[r_i, 1].set_title(f"Global Otsu (T*={ot})\nDiff vs Sauvola = {diff_pct:.1f}%", fontsize=8.8)

        axes[r_i, 2].imshow(cs, cmap="gray")
        axes[r_i, 2].set_title(f"Local Sauvola (w={cfg['sauvola_window']}, k={cfg['sauvola_k']})\nPreserved Ink Strokes", fontsize=8.8)

        for c_i in range(3):
            axes[r_i, c_i].axis("off")

        crop_meta[iid] = {"label": lbl, "otsu_T": int(ot), "otsu_sauvola_diff_pct": round(diff_pct, 2)}

    fig.suptitle("Zoomed Real-Document Crop Comparison: Global Otsu vs Local Sauvola", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_crop_otsu_vs_sauvola_real.png", dpi=180)
    plt.close(fig)

    # Sauvola parameter sensitivity grid on real shadow warped document
    wg = shadow_crop_gray
    o_full = otsu_threshold(wg)
    y0, x0 = _find_text_crop(wg, o_full["binary"], crop_h=min(240, wg.shape[0] // 2), crop_w=min(360, wg.shape[1] // 2))
    ch = min(240, wg.shape[0] - y0)
    cw = min(360, wg.shape[1] - x0)

    s_15_20 = sauvola_threshold(wg, window_size=15, k=0.20)
    s_25_20 = sauvola_threshold(wg, window_size=25, k=0.20)
    s_41_20 = sauvola_threshold(wg, window_size=41, k=0.20)
    s_25_10 = sauvola_threshold(wg, window_size=25, k=0.10)
    s_25_35 = sauvola_threshold(wg, window_size=25, k=0.35)

    panels = [
        ("Grayscale Crop", wg[y0 : y0 + ch, x0 : x0 + cw], "gray"),
        (f"Otsu (T*={o_full['threshold']})", o_full["binary"][y0 : y0 + ch, x0 : x0 + cw], "gray"),
        ("Sauvola w=15, k=0.20", s_15_20["binary"][y0 : y0 + ch, x0 : x0 + cw], "gray"),
        ("Sauvola w=25, k=0.20", s_25_20["binary"][y0 : y0 + ch, x0 : x0 + cw], "gray"),
        ("Sauvola w=41, k=0.20", s_41_20["binary"][y0 : y0 + ch, x0 : x0 + cw], "gray"),
        ("Sauvola w=25, k=0.10", s_25_10["binary"][y0 : y0 + ch, x0 : x0 + cw], "gray"),
        ("Sauvola w=25, k=0.35 [Frozen]", s_25_35["binary"][y0 : y0 + ch, x0 : x0 + cw], "gray"),
        ("Threshold Map T(x,y)", s_25_35["threshold_map"][y0 : y0 + ch, x0 : x0 + cw], "viridis"),
    ]
    fig2, axes2 = plt.subplots(2, 4, figsize=(14.5, 6.8))
    for ax, (ptitle, patch, cmap) in zip(axes2.ravel(), panels):
        ax.imshow(patch, cmap=cmap)
        ax.set_title(ptitle, fontsize=9.0)
        ax.axis("off")
    fig2.suptitle(f"Sauvola Window Size & k Sensitivity on Real Shadow Crop ({shad_id})", fontsize=12, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig_sauvola_param_grid.png", dpi=180)
    plt.close(fig2)

    return crop_meta


def profile_cpu_stages(cfg: dict, out_dir: Path) -> dict:
    rows = _load_csv("data/manifest_development.csv")
    stage_times = {
        "canny": [],
        "hough": [],
        "quad": [],
        "warping": [],
        "thresholding": [],
    }
    accepted_totals = []
    rejected_totals = []

    for r in rows:
        img = cv2.imread(os.path.join("data/development", r["filename"]))

        t0 = time.perf_counter()
        work_img, scale_to_orig = resize_for_work(img, int(cfg["work_max_dim"]))
        canny_res = canny_edge_detection(
            work_img,
            sigma=float(cfg["sigma"]),
            low_ratio=float(cfg["low_ratio"]),
            high_ratio=float(cfg["high_ratio"]),
            use_gaussian=bool(cfg["use_gaussian"]),
        )
        t1 = time.perf_counter()

        acc, rhos, thetas = hough_transform(
            canny_res["edges"], rho_res=float(cfg["rho_res"]), theta_res_deg=float(cfg["theta_res_deg"])
        )
        max_votes = int(np.max(acc)) if acc.size > 0 else 0
        vt = max(int(cfg["hough_min_votes"]), int(round(float(cfg["hough_thresh_ratio"]) * max_votes)))
        peaks = find_hough_peaks(
            acc,
            rhos,
            thetas,
            threshold=vt,
            top_k=int(cfg["top_k_lines"]),
            suppress=bool(cfg["hough_suppress"]),
            rho_supp=float(cfg["rho_supp"]),
            theta_supp_deg=float(cfg["theta_supp_deg"]),
        )
        t2 = time.perf_counter()

        quad_res = find_document_quad(
            peaks,
            canny_res["edges"],
            work_img.shape,
            use_edge_support=bool(cfg["use_edge_support"]),
            min_area_ratio=float(cfg["min_area_ratio"]),
            max_area_ratio=float(cfg["max_area_ratio"]),
            min_edge_support=float(cfg["min_edge_support"]),
            min_side_support=float(cfg.get("min_side_support", 0.50)),
            min_angle_deg=float(cfg["min_angle_deg"]),
            max_angle_deg=float(cfg["max_angle_deg"]),
        )
        t3 = time.perf_counter()

        dt_c = (t1 - t0) * 1000.0
        dt_h = (t2 - t1) * 1000.0
        dt_q = (t3 - t2) * 1000.0
        stage_times["canny"].append(dt_c)
        stage_times["hough"].append(dt_h)
        stage_times["quad"].append(dt_q)

        if quad_res["status"] == "accepted" and quad_res["corners"] is not None:
            orig_corners = (quad_res["corners"] * scale_to_orig).astype(np.float32)
            t4 = time.perf_counter()
            warped_bgr, _ = warp_document(img, orig_corners, interpolation="bilinear")
            warped_gray = to_grayscale(warped_bgr)
            t5 = time.perf_counter()
            _ = otsu_threshold(warped_gray)
            _ = sauvola_threshold(
                warped_gray, window_size=int(cfg["sauvola_window"]), k=float(cfg["sauvola_k"])
            )
            t6 = time.perf_counter()
            dt_w = (t5 - t4) * 1000.0
            dt_t = (t6 - t5) * 1000.0
            stage_times["warping"].append(dt_w)
            stage_times["thresholding"].append(dt_t)
            accepted_totals.append(dt_c + dt_h + dt_q + dt_w + dt_t)
        else:
            rejected_totals.append(dt_c + dt_h + dt_q)

    # Mean ms on accepted images (or overall for early stages)
    means = {k: float(np.mean(v)) if v else 0.0 for k, v in stage_times.items()}
    total_acc = sum(means.values())
    shares = {k: (100.0 * means[k] / total_acc if total_acc > 0 else 0.0) for k in means}
    profile = {
        "stage_mean_ms": {k: round(v, 2) for k, v in means.items()},
        "stage_share_pct": {k: round(v, 1) for k, v in shares.items()},
        "accepted_mean_total_ms": round(float(np.mean(accepted_totals)), 2),
        "rejected_mean_total_ms": round(float(np.mean(rejected_totals)), 2),
        "num_accepted": len(accepted_totals),
        "num_rejected": len(rejected_totals),
    }

    with open(out_dir / "runtime_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    # Plot breakdown bar chart
    names = [
        "1. Canny Edge\n(canny.py)",
        "2. Hough Transform\n(hough.py)",
        "3. Quad Selection\n(quadrilateral.py)",
        "4. Homography Warp\n(warping.py)",
        "5. Otsu & Sauvola\n(thresholding.py)",
    ]
    keys = ["canny", "hough", "quad", "warping", "thresholding"]
    vals = [means[k] for k in keys]
    pcts = [shares[k] for k in keys]
    colors = ["#2b6cb0", "#2c7a7b", "#b7791f", "#2f855a", "#6b46c1"]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, vals, color=colors, height=0.58)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean CPU Time per Stage (ms, excluding disk I/O)", fontsize=10)
    ax.set_title(
        f"Per-Module CPU Time Breakdown on Development (36 images, Accepted Mean = {profile['accepted_mean_total_ms']:.1f} ms)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(0, max(vals) * 1.28)
    ax.grid(axis="x", ls="--", alpha=0.4)
    for bar, v, p in zip(bars, vals, pcts):
        ax.text(v + 2.0, bar.get_y() + bar.get_height() / 2, f"{v:.1f} ms ({p:.1f}%)", va="center", fontsize=9.0)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_runtime_breakdown.png", dpi=180)
    plt.close(fig)
    return profile


def main() -> None:
    np.random.seed(42)
    cfg = json.load(open("outputs/frozen_config.json", encoding="utf-8"))
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {}
    build_architecture_diagram(cfg, out_dir)
    build_dataset_overview(out_dir)
    meta["canny_sweep"] = build_canny_sweep_visual(cfg, out_dir)
    meta["ablation_visual"] = build_ablation_visual(cfg, out_dir)
    meta["multidoc_stages"] = build_multidoc_stages(cfg, out_dir)
    meta["crop_comparisons"] = build_crop_comparisons(cfg, out_dir)
    meta["runtime_profile"] = profile_cpu_stages(cfg, out_dir)

    with open(out_dir / "rich_visuals_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Generated all Phase A rich visuals and profiling JSON successfully.")


if __name__ == "__main__":
    main()
