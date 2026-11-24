"""Build 12-slide widescreen (16:9) PowerPoint presentation from verified experiment data."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# Palette
NAVY = RGBColor(0x00, 0x20, 0x60)
STEEL = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x59, 0x59, 0x59)
SLATE_BG = RGBColor(0xF4, 0xF6, 0xF9)
ALT_ROW = RGBColor(0xEB, 0xF1, 0xF6)
BORDER = RGBColor(0xD0, 0xD7, 0xDE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
EMERALD = RGBColor(0x0F, 0x76, 0x6E)
EMERALD_BG = RGBColor(0xE6, 0xF4, 0xF1)
CORAL = RGBColor(0xB9, 0x1C, 0x1C)
CORAL_BG = RGBColor(0xFD, 0xEC, 0xEA)
AMBER_BG = RGBColor(0xFE, 0xF7, 0xE6)
AMBER = RGBColor(0xB4, 0x53, 0x09)

FONT_UI = "Segoe UI"
FONT_MONO = "Consolas"
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def _load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _summarize_det(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["condition"] != "khong_co_tai_lieu_hop_le"]
    neg = [r for r in rows if r["condition"] == "khong_co_tai_lieu_hop_le"]
    acc_pos = sum(r["status"] == "accepted" for r in pos)
    fp_neg = sum(r["status"] == "accepted" for r in neg)
    ious = [float(r["iou"]) for r in pos if r.get("iou", "N/A") != "N/A"]
    succ08 = sum(float(r["iou"]) >= 0.8 for r in pos if r.get("iou", "N/A") != "N/A")
    ces = [float(r["corner_error_norm"]) for r in pos if r.get("corner_error_norm", "N/A") != "N/A"]
    rt_acc = [float(r["runtime_ms"]) for r in rows if r["status"] == "accepted"]
    rt_rej = [float(r["runtime_ms"]) for r in rows if r["status"] == "rejected"]
    return {
        "pos_acc": acc_pos,
        "pos_tot": len(pos),
        "neg_fp": fp_neg,
        "neg_tot": len(neg),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "succ_08": succ08,
        "succ_rate": 100.0 * succ08 / max(1, len(pos)),
        "mean_ce": float(np.mean(ces)) if ces else 0.0,
        "rt_acc": float(np.mean(rt_acc)) if rt_acc else 0.0,
        "rt_rej": float(np.mean(rt_rej)) if rt_rej else 0.0,
    }


def _syn_f1(rows: list[dict], split: str, method: str) -> float:
    vals = [float(r["f1"]) for r in rows if r["split"] == split and r["method"] == method]
    return float(np.mean(vals)) if vals else 0.0


def add_picture_fit(slide, img_path: str, left: float, top: float, max_w: float, max_h: float):
    """Place image preserving exact natural aspect ratio, centered inside (left, top, max_w, max_h) inches."""
    w_px, h_px = Image.open(img_path).size
    aspect = w_px / max(1, h_px)
    box_aspect = max_w / max(1e-6, max_h)
    if aspect >= box_aspect:
        w = max_w
        h = w / aspect
    else:
        h = max_h
        w = h * aspect
    x = left + (max_w - w) / 2.0
    y = top + (max_h - h) / 2.0
    return slide.shapes.add_picture(img_path, Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def add_rect(slide, left: float, top: float, w: float, h: float, fill: RGBColor, border: RGBColor | None = None, radius: bool = False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, Inches(left), Inches(top), Inches(w), Inches(h))
    if radius and len(shp.adjustments) > 0:
        shp.adjustments[0] = 0.04
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if border is not None:
        shp.line.color.rgb = border
        shp.line.width = Pt(1.0)
    else:
        shp.line.fill.background()
    return shp


def add_textbox(slide, left: float, top: float, w: float, h: float, text: str = "", font_size: float = 14, bold: bool = False, color: RGBColor = DARK, align=PP_ALIGN.LEFT, font_name: str = FONT_UI):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    if text:
        p = tf.paragraphs[0]
        p.alignment = align
        r = p.add_run()
        r.text = text
        r.font.name = font_name
        r.font.size = Pt(font_size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def add_bullets(tf, items: list[tuple[str, str]], font_size: float = 13.5, space_after: float = 6.0):
    """Populate a text frame with bold-prefix + regular body bullets."""
    for idx, (prefix, body) in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = 1.18
        if prefix:
            r1 = p.add_run()
            r1.text = prefix + " "
            r1.font.name = FONT_UI
            r1.font.size = Pt(font_size)
            r1.font.bold = True
            r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_UI
        r2.font.size = Pt(font_size)
        r2.font.bold = False
        r2.font.color.rgb = DARK


def add_header_footer(slide, section_tag: str, title: str, slide_num: int, total_slides: int = 12):
    add_rect(slide, 0.0, 0.0, 13.333, 0.92, WHITE)
    add_rect(slide, 0.0, 0.0, 0.14, 0.92, NAVY)
    add_rect(slide, 0.55, 0.90, 12.23, 0.025, STEEL)

    add_textbox(slide, 0.55, 0.09, 11.8, 0.24, section_tag.upper(), font_size=10, bold=True, color=STEEL)
    add_textbox(slide, 0.55, 0.34, 12.2, 0.48, title, font_size=21, bold=True, color=NAVY)

    add_rect(slide, 0.55, 7.06, 12.23, 0.012, BORDER)
    add_textbox(slide, 0.55, 7.11, 8.5, 0.28, "Đồ án giữa kỳ: Document Scanner | Nhập môn Xử lý ảnh và Video (23MTNhung)", font_size=9.5, color=MUTED)
    add_textbox(slide, 10.5, 7.11, 2.28, 0.28, f"Slide {slide_num} / {total_slides}", font_size=9.5, bold=True, color=STEEL, align=PP_ALIGN.RIGHT)


def style_table(table, col_widths: list[float], font_size: float = 11.5):
    for i, w in enumerate(col_widths):
        table.columns[i].width = Inches(w)
    for r_i, row in enumerate(table.rows):
        for c_i, cell in enumerate(row.cells):
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.04)
            cell.fill.solid()
            if r_i == 0:
                cell.fill.fore_color.rgb = NAVY
            elif r_i % 2 == 1:
                cell.fill.fore_color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = ALT_ROW
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.name = FONT_UI
                    run.font.size = Pt(font_size if r_i > 0 else font_size)
                    run.font.bold = (r_i == 0) or (c_i == 0 and r_i > 0)
                    run.font.color.rgb = WHITE if r_i == 0 else DARK


def build_slides() -> Path:
    dev_sum = _summarize_det(_load_csv("outputs/results_development.csv"))
    test_sum = _summarize_det(_load_csv("outputs/results_test.csv"))
    b_dev = _summarize_det(_load_csv("outputs/results_baseline_dev.csv"))
    b_test = _summarize_det(_load_csv("outputs/results_baseline_test.csv"))
    ab_g = _summarize_det(_load_csv("outputs/ablation_no_gaussian.csv"))
    ab_h = _summarize_det(_load_csv("outputs/ablation_no_hough_suppression.csv"))
    ab_e = _summarize_det(_load_csv("outputs/ablation_no_edge_support.csv"))
    syn_rows = _load_csv("outputs/results_synthetic.csv")
    cases = json.loads(Path("outputs/figures/cases_metadata.json").read_text(encoding="utf-8"))
    rt_prof = json.loads(Path("outputs/figures/runtime_profile.json").read_text(encoding="utf-8"))

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]

    # ------------------------------------------------------------------
    # SLIDE 1: COVER
    # ------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank)
    add_rect(s1, 0.0, 0.0, 0.42, 7.5, NAVY)
    add_rect(s1, 0.42, 0.0, 0.08, 7.5, STEEL)
    add_rect(s1, 0.50, 0.0, 12.833, 7.5, WHITE)

    add_textbox(
        s1, 1.05, 0.65, 11.4, 0.85,
        "ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH  |  TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN\n"
        "KHOA ĐIỆN TỬ - VIỄN THÔNG  |  BỘ MÔN MÁY TÍNH - HỆ THỐNG NHÚNG",
        font_size=12, bold=True, color=STEEL,
    )
    add_rect(s1, 1.05, 1.48, 3.2, 0.04, NAVY)

    add_textbox(s1, 1.05, 1.72, 11.4, 0.42, "BÁO CÁO ĐỒ ÁN GIỮA KỲ  —  NHẬP MÔN XỬ LÝ ẢNH VÀ VIDEO", font_size=14, bold=True, color=EMERALD)
    add_textbox(
        s1, 1.05, 2.22, 11.5, 1.35,
        "HỆ THỐNG DOCUMENT SCANNER:\nPHÁT HIỆN TÀI LIỆU, HIỆU CHỈNH PHỐI CẢNH\nVÀ NHỊ PHÂN HÓA ẢNH",
        font_size=26, bold=True, color=NAVY,
    )

    card = add_rect(s1, 1.05, 4.05, 11.3, 2.25, SLATE_BG, border=BORDER, radius=True)
    tb_m = add_textbox(s1, 1.30, 4.18, 10.8, 0.35, "THÔNG TIN NHÓM THỰC HIỆN (LỚP 23MTNHUNG)", font_size=12, bold=True, color=NAVY)
    tf_m = add_textbox(s1, 1.30, 4.58, 10.8, 1.55).text_frame
    add_bullets(
        tf_m,
        [
            ("• Nguyễn Vũ Tuấn Khoa (23200096) — 34%:", "Kiến trúc pipeline, Canny/Hough/Quadrilateral, thực nghiệm & tổng hợp báo cáo"),
            ("• Lê Ngọc Long (23200107) — 33%:", "Warping/Bilinear, Otsu/Sauvola, bộ sinh 12 ảnh synthetic & 17 unit tests"),
            ("• Nguyễn Thanh Tuấn (23200162) — 33%:", "Phân tách 60 ảnh, gắn nhãn 50 ground-truth góc, Baseline/Ablation & trực quan"),
        ],
        font_size=13.5, space_after=6.0,
    )

    add_textbox(s1, 1.05, 6.62, 7.0, 0.35, "Môi trường: Python 3.11.9 | OpenCV 5.0.0 | NumPy 2.3.2 | CPU-only | Seed 42", font_size=11, color=MUTED, font_name=FONT_MONO)
    add_textbox(s1, 8.8, 6.62, 3.55, 0.35, "TP. Hồ Chí Minh — Tháng 11/2026", font_size=12, bold=True, color=STEEL, align=PP_ALIGN.RIGHT)

    # ------------------------------------------------------------------
    # SLIDE 2: MỤC TIÊU & RÀNG BUỘC
    # ------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank)
    add_header_footer(s2, "1. Giới thiệu bài toán", "Mục tiêu Đề tài & Ràng buộc Kỹ thuật Cốt lõi", 2)

    add_rect(s2, 0.55, 1.18, 5.95, 5.55, SLATE_BG, border=BORDER, radius=True)
    add_textbox(s2, 0.80, 1.35, 5.4, 0.38, "Mục tiêu & Phạm vi Xử lý", font_size=16, bold=True, color=NAVY)
    tf2l = add_textbox(s2, 0.80, 1.85, 5.45, 4.65).text_frame
    add_bullets(
        tf2l,
        [
            ("• Đầu vào:", "Ảnh chụp tài liệu bằng điện thoại dưới 5 điều kiện khó + nhóm ảnh âm không chứa giấy."),
            ("• Điều kiện chụp:", "Nền đơn giản, góc xiên mạnh, nền bàn phím chữ nhật nhiễu, biên tương phản thấp, bóng đổ."),
            ("• Nhiệm vụ 1 — Định vị:", "Xác định chính xác 4 góc trang giấy hoặc từ chối an toàn (rejected) ảnh không hợp lệ."),
            ("• Nhiệm vụ 2 — Nắn phẳng:", "Biến đổi phối cảnh Homography 3×3 giữ nguyên độ nét chữ ở độ phân giải gốc."),
            ("• Nhiệm vụ 3 — Nhị phân hóa:", "Tách nền/chữ rõ nét dưới ánh sáng không đều, phục vụ đọc và lưu trữ."),
        ],
        font_size=13.5, space_after=10.0,
    )

    cards_r = [
        (1.18, EMERALD_BG, EMERALD, "1. Tự cài đặt 100% lõi thuật toán (src/core/)",
         "Cài tường minh Canny, Hough, Quad Selection, Bilinear Sampling và Otsu/Sauvola từ mảng NumPy. "
         "Kiểm tra tĩnh AST xác nhận 0 vi phạm (không gọi cv2.Canny, cv2.HoughLines, cv2.threshold, skimage Sauvola)."),
        (3.10, SLATE_BG, STEEL, "2. Tuân thủ nghiêm ngặt No Data Leakage",
         "Chia tách 72 ảnh theo loại tài liệu ngay từ đầu (36 Dev + 24 Test + 12 Synthetic). "
         "Mọi khảo sát tham số chỉ chạy trên Development rồi đóng băng frozen_config.json trước khi chạy Test."),
        (5.02, AMBER_BG, AMBER, "3. Chạy CPU-only & Tái lập 100% với Seed 42",
         "Thời gian CPU đo thuần trên RAM (bỏ I/O ổ đĩa và vẽ hình). "
         "Cố định np.random.seed(42) và bộ kiểm thử biên 17 unit tests (17/17 PASSED)."),
    ]
    for top_y, bg_c, acc_c, head, desc in cards_r:
        add_rect(s2, 6.78, top_y, 6.00, 1.71, bg_c, border=BORDER, radius=True)
        add_rect(s2, 6.78, top_y, 0.12, 1.71, acc_c)
        add_textbox(s2, 7.05, top_y + 0.14, 5.55, 0.36, head, font_size=14.5, bold=True, color=acc_c)
        add_textbox(s2, 7.05, top_y + 0.52, 5.55, 1.08, desc, font_size=12.5, color=DARK)

    # ------------------------------------------------------------------
    # SLIDE 3: PIPELINE ARCHITECTURE
    # ------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank)
    add_header_footer(s3, "2. Kiến trúc hệ thống", "Các bước xử lý: Pipeline 5 Giai đoạn & Nhánh Early Exit", 3)

    add_rect(s3, 0.55, 1.12, 12.23, 4.18, WHITE, border=BORDER, radius=True)
    add_picture_fit(s3, "outputs/figures/fig_pipeline_architecture.png", 0.70, 1.20, 11.93, 4.02)

    pillars = [
        (0.55, "Xử lý 2 thang độ phân giải",
         "Canny & Hough chạy ở 640 px để tăng tốc CPU (~193 ms), sau đó nhân tỷ lệ 4 góc về độ phân giải gốc khi Warping giữ nguyên độ nét chữ."),
        (4.72, "Chấm điểm Edge Support + Area",
         "Kết hợp 0.65×S_edge + 0.35×A_norm và yêu cầu từng cạnh đạt min_side_support >= 0.50 để loại tứ giác giả trên nền nhiễu."),
        (8.89, "Cơ chế từ chối Early Exit",
         "Khi ảnh âm hoặc đứt biên không đủ tứ giác đạt chuẩn, pipeline dừng tại Bước 3 kèm mã lỗi cụ thể, bỏ qua Warping và Thresholding."),
    ]
    for x0, p_head, p_desc in pillars:
        add_rect(s3, x0, 5.48, 3.89, 1.38, SLATE_BG, border=BORDER, radius=True)
        add_textbox(s3, x0 + 0.15, 5.58, 3.60, 0.32, p_head, font_size=13.5, bold=True, color=NAVY)
        add_textbox(s3, x0 + 0.15, 5.93, 3.60, 0.85, p_desc, font_size=11.8, color=DARK)

    # ------------------------------------------------------------------
    # SLIDE 4: CORE MODULES & VERIFICATION
    # ------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank)
    add_header_footer(s4, "3. Phần tự cài & Kiểm thử", "Chi tiết 5 Module Tự cài (src/core/) và Cổng Kiểm định", 4)

    rows_mod = [
        ["Module", "Thuật toán & Kỹ thuật tự cài", "Chi tiết lõi & Tối ưu", "Unit Tests (ca biên)"],
        ["src/core/canny.py", "Gaussian 1D khả tách -> Sobel 3×3 -> NMS 4 hướng -> Hysteresis 8-láng giềng", "Vector hóa NMS (0°, 45°, 90°, 135°); duyệt BFS hàng đợi tách biên yếu nối biên mạnh.", "5/5 PASSED (ảnh hằng, step edge, NMS, ngưỡng 0, BFS)"],
        ["src/core/hough.py", "Hough Polar rho = x·cos(θ) + y·sin(θ), θ ∈ [0, π) + Peak NMS", "Cộng dồn phiếu vector hóa bằng np.bincount; triệt tiêu đỉnh trùng tuần hoàn theo Δθ.", "2/2 PASSED (edge map rỗng, đường 0° và 90°)"],
        ["src/core/quadrilateral.py", "Ghép 4 đường trực giao -> np.linalg.solve -> Convexity -> Scoring", "Lọc khung ảnh, tính lồi, góc [45°,135°], cạnh tối thiểu, độ bám biên chu vi & từng cạnh.", "5/5 PASSED (CW, 3 điểm thẳng hàng, song song, thiếu đường)"],
        ["src/core/warping.py", "Ước lượng kích thước A4/trang -> Homography DLT 3×3 + Bilinear", "Tự cài bilinear_sample_gray với 4 lân cận chuẩn; giải phối cảnh trên độ phân giải gốc.", "3/3 PASSED (bilinear tính tay, identity, phối cảnh thật)"],
        ["src/core/thresholding.py", "Otsu toàn cục (histogram 256 mức) & Sauvola cục bộ (w×w)", "Otsu tối đa phương sai liên lớp σ²_b(t); Sauvola dùng boxFilter tính tổng cục bộ I và I².", "2/2 PASSED (ảnh hằng không NaN, synthetic mask F1 = 1.0)"],
    ]
    tbl4 = s4.shapes.add_table(len(rows_mod), 4, Inches(0.55), Inches(1.12), Inches(12.23), Inches(4.75)).table
    for r_i, row_vals in enumerate(rows_mod):
        for c_i, val in enumerate(row_vals):
            tbl4.cell(r_i, c_i).text = val
    style_table(tbl4, [2.15, 3.65, 3.63, 2.80], font_size=11.0)

    add_rect(s4, 0.55, 6.02, 5.95, 0.86, EMERALD_BG, border=EMERALD, radius=True)
    add_textbox(s4, 0.75, 6.10, 5.55, 0.70, "AST Guard (tests/verify_ast.py): PASSED (6 tệp src/core/ — 0 vi phạm)\nXác nhận không import/gọi hàm xử lý ảnh cấm từ OpenCV hoặc scikit-image.", font_size=12, bold=True, color=EMERALD)

    add_rect(s4, 6.83, 6.02, 5.95, 0.86, SLATE_BG, border=STEEL, radius=True)
    add_textbox(s4, 7.03, 6.10, 5.55, 0.70, "Edge-Case Suite (pytest tests/ -v): 17 / 17 PASSED (0.28 s)\nPhủ toàn bộ ca nhỏ: ảnh hằng, step edge, đường ngang/đứng, bilinear tính tay, mask.", font_size=12, bold=True, color=NAVY)

    # ------------------------------------------------------------------
    # SLIDE 5: DATASET & GROUND TRUTH
    # ------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank)
    add_header_footer(s5, "4. Dữ liệu & Nhãn chuẩn", "Bộ dữ liệu 72 Ảnh (60 Thực tế + 12 Tổng hợp) & Nhãn 4 góc GT", 5)

    add_rect(s5, 0.55, 1.15, 7.65, 5.65, WHITE, border=BORDER, radius=True)
    add_picture_fit(s5, "outputs/figures/fig_dataset_overview.png", 0.65, 1.25, 7.45, 5.15)
    add_textbox(s5, 0.75, 6.42, 7.25, 0.30, "Khung hồng cánh sen (magenta): Ground-truth 4 góc trích xuất từ data/labels.json (0 ảnh dương bị N/A).", font_size=10.5, color=MUTED, align=PP_ALIGN.CENTER)

    add_rect(s5, 8.40, 1.15, 4.38, 5.65, SLATE_BG, border=BORDER, radius=True)
    add_textbox(s5, 8.62, 1.32, 3.95, 0.38, "Phân tách Development / Test", font_size=15.5, bold=True, color=NAVY)
    tf5 = add_textbox(s5, 8.62, 1.80, 3.95, 4.85).text_frame
    add_bullets(
        tf5,
        [
            ("• Development (36 thực + 6 synth):", "6 loại tài liệu × 5 ĐK (30 dương) + 6 ảnh âm + 6 ảnh chữ tổng hợp (Layout 1–2)."),
            ("• Test (24 thực + 6 synth):", "4 loại tài liệu × 5 ĐK (20 dương) + 4 ảnh âm + 6 ảnh chữ tổng hợp (Layout 3–4)."),
            ("• Nhãn 4 góc (data/labels.json):", "Gắn nhãn thủ công + kiểm tra trực quan đủ 50/50 ảnh dương (TL, TR, BR, BL)."),
            ("• Thước đo chuẩn:", "Polygon IoU (rejected = 0), Success@0.8, Corner Error / đường chéo, FPR ảnh âm và Ink F1."),
        ],
        font_size=12.5, space_after=10.0,
    )

    # ------------------------------------------------------------------
    # SLIDE 6: PARAMETER SWEEPS
    # ------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank)
    add_header_footer(s6, "5. Khảo sát tham số (Dev only)", "Ảnh hưởng tham số Canny, Hough/Quad và Sauvola trên Development", 6)

    add_rect(s6, 0.55, 1.12, 6.00, 4.62, WHITE, border=BORDER, radius=True)
    add_textbox(s6, 0.70, 1.18, 5.70, 0.30, "Canny: σ ∈ {0.8, 1.4, 2.2} và 3 cặp ngưỡng kép", font_size=12.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    add_picture_fit(s6, "outputs/figures/fig_canny_sweep_visual.png", 0.65, 1.50, 5.80, 4.15)

    add_rect(s6, 6.78, 1.12, 6.00, 4.62, WHITE, border=BORDER, radius=True)
    add_textbox(s6, 6.93, 1.18, 5.70, 0.30, "Sauvola: Window {15, 25, 41} × k {0.10, 0.20, 0.35}", font_size=12.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    add_picture_fit(s6, "outputs/figures/fig_sauvola_param_grid.png", 6.88, 1.50, 5.80, 4.15)

    add_rect(s6, 0.55, 5.88, 12.23, 1.00, NAVY, radius=True)
    add_textbox(
        s6, 0.78, 5.98, 11.78, 0.80,
        "Cấu hình đóng băng trên Dev (outputs/frozen_config.json):\n"
        "• Canny: σ = 1.4, (low, high) = (0.06, 0.16)   |   Hough & Quad: thresh_ratio = 0.24, edge_support+area (giảm FPR từ 5/6 -> 3/6 & CE còn 0.0824)\n"
        "• Sauvola: window_size = 25, k = 0.35 (F1 synthetic = 0.9999; làm sạch nền bóng đổ tốt hơn k = 0.10/0.20)",
        font_size=11.8, color=WHITE,
    )

    # ------------------------------------------------------------------
    # SLIDE 7: CUSTOM VS BASELINE QUANTITATIVE
    # ------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank)
    add_header_footer(s7, "6. Kết quả định lượng", "So sánh Custom Pipeline và OpenCV Baseline trên Dev & Test", 7)

    rows_cmp = [
        ["Chỉ số đánh giá", f"Custom (Dev, n={dev_sum['pos_tot']+dev_sum['neg_tot']})", f"Baseline (Dev, n={b_dev['pos_tot']+b_dev['neg_tot']})", f"Custom (Test, n={test_sum['pos_tot']+test_sum['neg_tot']})", f"Baseline (Test, n={b_test['pos_tot']+b_test['neg_tot']})"],
        ["Ảnh dương hợp lệ", f"{dev_sum['pos_acc']}/{dev_sum['pos_tot']} ({100*dev_sum['pos_acc']/dev_sum['pos_tot']:.1f}%)", f"{b_dev['pos_acc']}/{b_dev['pos_tot']} ({100*b_dev['pos_acc']/b_dev['pos_tot']:.1f}%)", f"{test_sum['pos_acc']}/{test_sum['pos_tot']} ({100*test_sum['pos_acc']/test_sum['pos_tot']:.1f}%)", f"{b_test['pos_acc']}/{b_test['pos_tot']} ({100*b_test['pos_acc']/b_test['pos_tot']:.1f}%)"],
        ["Mean Polygon IoU", f"{dev_sum['mean_iou']:.4f}", f"{b_dev['mean_iou']:.4f}", f"{test_sum['mean_iou']:.4f}", f"{b_test['mean_iou']:.4f}"],
        ["Success @ IoU >= 0.8", f"{dev_sum['succ_08']}/{dev_sum['pos_tot']} ({dev_sum['succ_rate']:.1f}%)", f"{b_dev['succ_08']}/{b_dev['pos_tot']} ({b_dev['succ_rate']:.1f}%)", f"{test_sum['succ_08']}/{test_sum['pos_tot']} ({test_sum['succ_rate']:.1f}%)", f"{b_test['succ_08']}/{b_test['pos_tot']} ({b_test['succ_rate']:.1f}%)"],
        ["Normalized Corner Err", f"{dev_sum['mean_ce']:.4f} ({dev_sum['pos_acc']}/{dev_sum['pos_tot']})", f"{b_dev['mean_ce']:.4f} ({b_dev['pos_acc']}/{b_dev['pos_tot']})", f"{test_sum['mean_ce']:.4f} ({test_sum['pos_acc']}/{test_sum['pos_tot']})", f"{b_test['mean_ce']:.4f} ({b_test['pos_acc']}/{b_test['pos_tot']})"],
        ["FPR Ảnh âm", f"{dev_sum['neg_fp']}/{dev_sum['neg_tot']} ({100*dev_sum['neg_fp']/dev_sum['neg_tot']:.1f}%)", f"{b_dev['neg_fp']}/{b_dev['neg_tot']} ({100*b_dev['neg_fp']/b_dev['neg_tot']:.1f}%)", f"{test_sum['neg_fp']}/{test_sum['neg_tot']} ({100*test_sum['neg_fp']/test_sum['neg_tot']:.1f}%)", f"{b_test['neg_fp']}/{b_test['neg_tot']} (0.0%)"],
        ["CPU Acc / Rej (ms)", f"{dev_sum['rt_acc']:.1f} / {dev_sum['rt_rej']:.1f}", f"{b_dev['rt_acc']:.1f} / {b_dev['rt_rej']:.1f}", f"{test_sum['rt_acc']:.1f} / {test_sum['rt_rej']:.1f}", f"{b_test['rt_acc']:.1f} / {b_test['rt_rej']:.1f}"],
    ]
    tbl7 = s7.shapes.add_table(len(rows_cmp), 5, Inches(0.55), Inches(1.18), Inches(8.55), Inches(5.60)).table
    for r_i, row_vals in enumerate(rows_cmp):
        for c_i, val in enumerate(row_vals):
            tbl7.cell(r_i, c_i).text = val
    style_table(tbl7, [2.15, 1.60, 1.60, 1.60, 1.60], font_size=11.5)

    kpis = [
        (1.18, EMERALD_BG, EMERALD, "+0.2618 Mean IoU trên Test", f"Custom đạt {test_sum['mean_iou']:.4f} so với {b_test['mean_iou']:.4f} của OpenCV Baseline (tăng hơn gấp 2.2 lần)."),
        (3.08, SLATE_BG, NAVY, "Vượt trội ảnh phức tạp", "Nhận đúng 17/20 ảnh dương Test (85%) nhờ tích lũy Hough toàn cục khi biên bị đứt do chữ/hoa văn."),
        (4.98, AMBER_BG, AMBER, "Điểm nghẽn khung lồng", "Một số ca sách có khung tranh nội tại làm tứ giác co vào trong, kéo Success@0.8 xuống 30% (Dev) & 15% (Test)."),
    ]
    for y0, bg, fg, k_head, k_txt in kpis:
        add_rect(s7, 9.35, y0, 3.43, 1.80, bg, border=BORDER, radius=True)
        add_rect(s7, 9.35, y0, 0.11, 1.80, fg)
        add_textbox(s7, 9.58, y0 + 0.15, 3.05, 0.36, k_head, font_size=14, bold=True, color=fg)
        add_textbox(s7, 9.58, y0 + 0.56, 3.05, 1.15, k_txt, font_size=12, color=DARK)

    # ------------------------------------------------------------------
    # SLIDE 8: ABLATION STUDY
    # ------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank)
    add_header_footer(s8, "7. Nghiên cứu tách biệt thành phần", "Ablation Study trên Development (cố định top_k_lines = 20)", 8)

    add_rect(s8, 0.55, 1.12, 12.23, 3.85, WHITE, border=BORDER, radius=True)
    add_picture_fit(s8, "outputs/figures/chart_ablation_and_baseline.png", 0.70, 1.20, 11.93, 3.68)

    ab_cards = [
        (0.55, CORAL_BG, CORAL, f"1. Bỏ Gaussian (IoU {ab_g['mean_iou']:.4f})",
         f"Succ@0.8 giảm từ {dev_sum['succ_08']}/30 xuống {ab_g['succ_08']}/30; nhiễu cạnh tăng làm CPU Acc lên {ab_g['rt_acc']:.1f} ms (+58.6 ms)."),
        (4.72, CORAL_BG, CORAL, f"2. Bỏ Hough Supp. (IoU {ab_h['mean_iou']:.4f})",
         f"Hợp lệ tụt mạnh còn {ab_h['pos_acc']}/30 (IoU giảm 0.1579) vì các đường trùng ở cạnh mạnh chiếm hết top-20 ứng viên."),
        (8.89, AMBER_BG, AMBER, f"3. Bỏ Edge Support (FPR {ab_e['neg_fp']}/6)",
         f"Mean IoU tăng lên {ab_e['mean_iou']:.4f} do chọn khung lớn, nhưng FPR ảnh âm vọt lên {ab_e['neg_fp']}/6 (83.3%) vì nhận cả viền nền."),
    ]
    for x0, bg, fg, a_head, a_txt in ab_cards:
        add_rect(s8, x0, 5.15, 3.89, 1.72, bg, border=BORDER, radius=True)
        add_rect(s8, x0, 5.15, 3.89, 0.08, fg)
        add_textbox(s8, x0 + 0.14, 5.30, 3.62, 0.34, a_head, font_size=13.5, bold=True, color=fg)
        add_textbox(s8, x0 + 0.14, 5.68, 3.62, 1.10, a_txt, font_size=12.0, color=DARK)

    # ------------------------------------------------------------------
    # SLIDE 9: BINARIZATION (SYNTHETIC + REAL CROP)
    # ------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank)
    add_header_footer(s9, "8. Nhị phân hóa Otsu & Sauvola", "Đánh giá định lượng trên 12 Ảnh tổng hợp & Crop Thực tế", 9)

    add_rect(s9, 0.55, 1.15, 5.65, 3.70, WHITE, border=BORDER, radius=True)
    add_picture_fit(s9, "outputs/figures/chart_synthetic_f1.png", 0.68, 1.24, 5.39, 3.52)

    o_dev = _syn_f1(syn_rows, "development", "otsu")
    s_dev = _syn_f1(syn_rows, "development", "sauvola")
    o_tst = _syn_f1(syn_rows, "test", "otsu")
    s_tst = _syn_f1(syn_rows, "test", "sauvola")
    add_rect(s9, 0.55, 5.00, 5.65, 1.85, EMERALD_BG, border=EMERALD, radius=True)
    add_textbox(
        s9, 0.75, 5.12, 5.25, 1.62,
        "Tổng kết 12 ảnh Synthetic (Precision / Recall / F1):\n"
        f"• Otsu toàn cục:   Dev Ink F1 = {o_dev:.4f}  |  Test Ink F1 = {o_tst:.4f}\n"
        f"• Sauvola cục bộ:  Dev Ink F1 = {s_dev:.4f}  |  Test Ink F1 = {s_tst:.4f}\n"
        "• Trên ảnh nền gradient/bóng đổ, Otsu bôi đen cả vùng râm (F1 ~ 0.16–0.20), "
        "trong khi Sauvola bảo tồn nét chữ gần như tuyệt đối.",
        font_size=12.0, color=DARK,
    )

    add_rect(s9, 6.40, 1.15, 6.38, 5.70, WHITE, border=BORDER, radius=True)
    add_picture_fit(s9, "outputs/figures/fig_crop_otsu_vs_sauvola_real.png", 6.50, 1.25, 6.18, 5.05)
    add_textbox(s9, 6.52, 6.36, 6.14, 0.42, "Crop thực tế: Thẻ sinh viên (dev_26), Vé xe buýt (test_01, Otsu sai 32.2% pixel) và Tài liệu bóng đổ (dev_10).", font_size=10.8, color=MUTED, align=PP_ALIGN.CENTER)

    # ------------------------------------------------------------------
    # SLIDE 10: REPRESENTATIVE RESULTS (6-STAGE PROGRESSION)
    # ------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank)
    add_header_footer(s10, "9. Kết quả tiêu biểu", "Chuỗi 6 bước hoàn chỉnh trên các loại tài liệu thực tế", 10)

    add_rect(s10, 0.55, 1.15, 5.95, 5.70, WHITE, border=BORDER, radius=True)
    add_textbox(s10, 0.72, 1.24, 5.60, 0.34, "Thẻ sinh viên (dev_26) — IoU = 0.9828 | Corner Error = 0.0021", font_size=13, bold=True, color=EMERALD, align=PP_ALIGN.CENTER)
    add_picture_fit(s10, "outputs/figures/fig_stages_thesv_dev26.png", 0.68, 1.62, 5.69, 4.70)
    add_textbox(s10, 0.72, 6.38, 5.60, 0.40, "Khớp sát 4 góc bất chấp mặt bàn tối; Sauvola tách sạch chữ trên nền gradient hoa văn.", font_size=11, color=MUTED, align=PP_ALIGN.CENTER)

    add_rect(s10, 6.83, 1.15, 5.95, 5.70, WHITE, border=BORDER, radius=True)
    add_textbox(s10, 7.00, 1.24, 5.60, 0.34, "Sổ tay bìa Leaf (dev_16) — IoU = 0.9830 | Corner Error = 0.0046", font_size=13, bold=True, color=EMERALD, align=PP_ALIGN.CENTER)
    add_picture_fit(s10, "outputs/figures/fig_stages_leaf_dev16.png", 6.96, 1.62, 5.69, 4.70)
    add_textbox(s10, 7.00, 6.38, 5.60, 0.40, "6 đường Hough sạch tách đúng khung trang vở lớn; Homography giữ nét chữ và logo rõ nét.", font_size=11, color=MUTED, align=PP_ALIGN.CENTER)

    # ------------------------------------------------------------------
    # SLIDE 11: 6 DIAGNOSTIC CASE STUDIES
    # ------------------------------------------------------------------
    s11 = prs.slides.add_slide(blank)
    add_header_footer(s11, "10. Phân tích 6 trường hợp tiêu biểu", "Truy vết nguyên nhân thành công, ca lỗi & ảnh âm bằng ảnh trung gian", 11)

    strips = [
        ("outputs/figures/case3_dev03_clutter.png", "dev_03 (Nền phím nhiễu): Accepted, IoU=0.8615", 1.15),
        ("outputs/figures/case5_dev05_shadow.png", "dev_05 (Bóng đổ gắt): Rejected out_of_bounds", 3.07),
        ("outputs/figures/case6_dev32_negative_fp.png", "dev_32 (Ảnh âm màn hình): False-Positive IoU=0", 4.99),
    ]
    for img_p, cap, y_p in strips:
        add_rect(s11, 0.55, y_p, 7.25, 1.80, WHITE, border=BORDER, radius=True)
        add_picture_fit(s11, img_p, 0.63, y_p + 0.06, 7.09, 1.44)
        add_textbox(s11, 0.65, y_p + 1.50, 7.05, 0.26, cap, font_size=10.2, bold=True, color=STEEL, align=PP_ALIGN.CENTER)

    add_rect(s11, 7.98, 1.15, 4.80, 5.64, SLATE_BG, border=BORDER, radius=True)
    add_textbox(s11, 8.18, 1.28, 4.40, 0.34, "Tổng hợp 6 ca phân tích (Xanh=Pred, Hồng=GT)", font_size=13.5, bold=True, color=NAVY)
    tf11 = add_textbox(s11, 8.18, 1.70, 4.40, 4.95).text_frame
    add_bullets(
        tf11,
        [
            ("• Ca 1 (dev_01 — Nền sạch):", f"Accepted (IoU={cases['dev_01']['iou']}); co nhẹ vào khung tranh in trên bìa do hỗ trợ biên cao."),
            ("• Ca 2 (dev_07 — Góc xiên):", "Rejected (low_edge_support < 0.72); góc nghiêng làm đứt đoạn cạnh ngắn."),
            ("• Ca 3 (dev_03 — Nền nhiễu):", f"Accepted (IoU={cases['dev_03']['iou']}); loại hoàn toàn bàn phím nhờ edge_support = {cases['dev_03']['edge_support']:.2f}."),
            ("• Ca 4 (dev_29 — Tương phản thấp):", "Rejected (short_side); top-20 Hough bị chữ bên trong thẻ chiếm chỗ."),
            ("• Ca 5 (dev_05 — Bóng đổ):", "Rejected (out_of_bounds); bóng làm tối cạnh dưới, ghép nhầm đường gân chữ."),
            ("• Ca 6 (dev_32 & dev_33 — Ảnh âm):", "dev_32 (màn hình) gây FP do viền chữ nhật nét; dev_33 (giấy xé) bị loại đúng (short_side)."),
        ],
        font_size=11.5, space_after=7.0,
    )

    # ------------------------------------------------------------------
    # SLIDE 12: RUNTIME, LIMITATIONS & CONCLUSION
    # ------------------------------------------------------------------
    s12 = prs.slides.add_slide(blank)
    add_header_footer(s12, "11. Thời gian CPU & Kết luận", "Phân bổ Thời gian xử lý, Hạn chế và Hướng Phát triển", 12)

    add_rect(s12, 0.55, 1.15, 6.10, 3.65, WHITE, border=BORDER, radius=True)
    add_picture_fit(s12, "outputs/figures/fig_runtime_breakdown.png", 0.65, 1.25, 5.90, 3.45)

    add_rect(s12, 0.55, 4.95, 6.10, 1.90, SLATE_BG, border=BORDER, radius=True)
    add_textbox(
        s12, 0.75, 5.06, 5.70, 1.68,
        "Hiệu năng CPU thuần (loại trừ đọc/ghi đĩa & vẽ hình):\n"
        f"• 3 bước định vị (Canny {rt_prof['stage_share_pct']['canny']:.1f}% + Hough {rt_prof['stage_share_pct']['hough']:.1f}% + Quad {rt_prof['stage_share_pct']['quad']:.1f}%) chiếm 84.8% tổng thời gian.\n"
        f"• Nhờ Early Exit, ca bị từ chối tiết kiệm hoàn toàn bước Warping ({rt_prof['stage_mean_ms']['warping']:.1f} ms) và Thresholding ({rt_prof['stage_mean_ms']['thresholding']:.1f} ms).",
        font_size=12.0, color=DARK,
    )

    add_rect(s12, 6.88, 1.15, 5.90, 4.40, SLATE_BG, border=BORDER, radius=True)
    add_textbox(s12, 7.10, 1.32, 5.45, 0.36, "Hạn chế hiện tại & Hướng nâng cấp", font_size=15, bold=True, color=NAVY)
    tf12 = add_textbox(s12, 7.10, 1.78, 5.45, 3.65).text_frame
    add_bullets(
        tf12,
        [
            ("• Ưu điểm đạt được:", "Tự cài 100% lõi thuật toán (0 vi phạm AST), vượt xa Baseline OpenCV về Mean IoU trên cả Dev và Test; Sauvola đạt Ink F1 = 0.9999."),
            ("• Hạn chế 1 — Khung lồng nhau:", "Sách/bìa có khung ảnh in bên trong khiến điểm ưu tiên khung nhỏ rõ nét hơn mép ngoài cùng."),
            ("• Hạn chế 2 — Báo giả ảnh âm:", "Vật thể hình chữ nhật phẳng (màn hình, gạch lát) có viền sắc nét vẫn vượt qua bộ lọc hình học."),
            ("• Hướng cải tiến:", "Ưu tiên tứ giác bao ngoài cùng khi các ứng viên lồng nhau và thêm hậu kiểm mật độ nét chữ / phương sai vùng sau warp."),
        ],
        font_size=12.3, space_after=8.0,
    )

    add_rect(s12, 6.88, 5.72, 5.90, 1.13, NAVY, radius=True)
    add_textbox(s12, 7.08, 5.86, 5.50, 0.85, "CẢM ƠN THẦY VÀ CÁC BẠN ĐÃ LẮNG NGHE!\nSẵn sàng trình bày & giải đáp câu hỏi (Q&A)", font_size=14.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    out_pptx = Path("outputs/Slides_Giua_Ky_Document_Scanner.pptx")
    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_pptx))
    print(f"Saved {out_pptx}")
    return out_pptx


def export_pdf(pptx_path: Path) -> Path:
    abs_pptx = str(pptx_path.resolve())
    pdf_path = pptx_path.with_suffix(".pdf")
    abs_pdf = str(pdf_path.resolve())
    ps_script = f"""
$pp = New-Object -ComObject PowerPoint.Application
$pres = $null
try {{
    $pres = $pp.Presentations.Open('{abs_pptx}', $true, $false, $false)
    $pres.SaveAs('{abs_pdf}', 32)
}} finally {{
    if ($pres -ne $null) {{ $pres.Close() }}
    $pp.Quit()
}}
"""
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], capture_output=True, text=True)
    if res.returncode != 0 or not pdf_path.exists():
        raise RuntimeError(f"PowerPoint PDF export failed: {res.stderr or res.stdout}")
    print(f"Exported {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    p = build_slides()
    if "--export-pdf" in sys.argv:
        export_pdf(p)
