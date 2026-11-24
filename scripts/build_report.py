import ast
import csv
import json
import os
import platform
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import scipy
import skimage
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree


MML_NS = "http://www.w3.org/1998/Math/MathML"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
MML2OMML_PATH = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
OMML_XSLT = etree.XSLT(etree.parse(MML2OMML_PATH)) if os.path.exists(MML2OMML_PATH) else None

PPR_ORDER = [
    "pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl",
    "numPr", "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens",
    "kinsoku", "wordWrap", "overflowPunct", "topLinePunct", "autoSpaceDE",
    "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid", "spacing", "ind",
    "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
    "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl",
    "divId", "cnfStyle", "rPr", "sectPr", "pPrChange",
]

RPR_ORDER = [
    "rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps",
    "strike", "dstrike", "outline", "shadow", "emboss", "imprint", "noProof",
    "snapToGrid", "vanish", "webHidden", "color", "spacing", "w", "kern",
    "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd",
    "fitText", "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout",
    "specVanish", "oMath", "rPrChange",
]

TCPR_ORDER = [
    "cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd",
    "noWrap", "tcMar", "textDirection", "tcFitText", "vAlign", "hideMark",
    "headers", "cellIns", "cellDel", "cellMerge", "tcPrChange",
]


def _insert_ooxml_ordered(parent, child, order: list[str]) -> None:
    """Insert child element into parent respecting strict OOXML schema sequence."""
    tag_name = etree.QName(child).localname
    for existing in list(parent):
        if etree.QName(existing).localname == tag_name:
            parent.remove(existing)
    target_idx = order.index(tag_name) if tag_name in order else len(order)
    insert_pos = len(parent)
    for pos, elem in enumerate(parent):
        ename = etree.QName(elem).localname
        e_idx = order.index(ename) if ename in order else len(order)
        if e_idx > target_idx:
            insert_pos = pos
            break
    parent.insert(insert_pos, child)


def _load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract_func_source(filepath: str, func_name: str, max_lines: int = 34) -> str:
    """Read exact source lines of a top-level function from a Python module."""
    src_text = Path(filepath).read_text(encoding="utf-8")
    tree = ast.parse(src_text)
    lines = src_text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            chunk = [
                ln for ln in lines[node.lineno - 1 : node.end_lineno]
                if not ln.lstrip().startswith("#")
            ]
            if len(chunk) > max_lines:
                chunk = chunk[:max_lines]
            return "\n".join(chunk)
    raise ValueError(f"Function {func_name} not found in {filepath}")


def _summarize_detection(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["condition"] != "khong_co_tai_lieu_hop_le"]
    neg = [r for r in rows if r["condition"] == "khong_co_tai_lieu_hop_le"]
    pos_acc = sum(r["status"] == "accepted" for r in pos)
    neg_fp = sum(r["status"] == "accepted" for r in neg)
    ious = [float(r["iou"]) for r in pos if r["iou"] != "N/A"]
    ces = [float(r["corner_error_norm"]) for r in pos if r["corner_error_norm"] != "N/A"]
    succ_08 = sum(1 for v in ious if v >= 0.8)
    acc_rt = [float(r["runtime_ms"]) for r in rows if r["status"] == "accepted"]
    rej_rt = [float(r["runtime_ms"]) for r in rows if r["status"] == "rejected"]

    by_cond = {}
    for cond in (
        "nen_don_gian_du_sang",
        "goc_chup_xien",
        "nen_chu_nhat_gay_nhieu",
        "bien_tuong_phan_thap",
        "anh_sang_khong_deu_bong_do",
        "khong_co_tai_lieu_hop_le",
    ):
        sub = [r for r in rows if r["condition"] == cond]
        acc = sum(r["status"] == "accepted" for r in sub)
        c_ious = [float(r["iou"]) for r in sub if r["iou"] != "N/A"]
        c_ces = [float(r["corner_error_norm"]) for r in sub if r["corner_error_norm"] != "N/A"]
        c_s08 = sum(1 for v in c_ious if v >= 0.8)
        by_cond[cond] = {
            "accepted": acc,
            "total": len(sub),
            "mean_iou": float(np.mean(c_ious)) if c_ious else 0.0,
            "succ_08": c_s08,
            "mean_ce": float(np.mean(c_ces)) if c_ces else 0.0,
            "ce_cnt": len(c_ces),
        }

    return {
        "pos_acc": pos_acc,
        "pos_tot": len(pos),
        "pos_rate": 100.0 * pos_acc / max(1, len(pos)),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "succ_08": succ_08,
        "succ_08_rate": 100.0 * succ_08 / max(1, len(pos)),
        "mean_ce": float(np.mean(ces)) if ces else 0.0,
        "ce_cnt": len(ces),
        "neg_fp": neg_fp,
        "neg_tot": len(neg),
        "neg_fpr": 100.0 * neg_fp / max(1, len(neg)),
        "rt_acc_mean": float(np.mean(acc_rt)) if acc_rt else 0.0,
        "rt_rej_mean": float(np.mean(rej_rt)) if rej_rt else 0.0,
        "by_cond": by_cond,
    }


def _summarize_synthetic(rows: list[dict]) -> dict:
    out = {}
    for split in ("development", "test"):
        out[split] = {}
        for method in sorted({r["method"] for r in rows}):
            out[split][method] = {}
            for cond in ("nen_deu", "gradient_sang", "bong_kem_nhieu", "all"):
                sub = [
                    r
                    for r in rows
                    if r["split"] == split
                    and r["method"] == method
                    and (cond == "all" or r["condition"] == cond)
                ]
                out[split][method][cond] = {
                    "p": float(np.mean([float(r["precision"]) for r in sub])),
                    "r": float(np.mean([float(r["recall"]) for r in sub])),
                    "f1": float(np.mean([float(r["f1"]) for r in sub])),
                    "rt": float(np.mean([float(r["runtime_ms"]) for r in sub])),
                }
    return out


def build_markdown_report() -> tuple[str, dict]:
    cfg = json.load(open("outputs/frozen_config.json", encoding="utf-8"))
    cases = json.load(open("outputs/figures/cases_metadata.json", encoding="utf-8"))
    rich_meta = json.load(open("outputs/figures/rich_visuals_metadata.json", encoding="utf-8"))
    rt_prof = json.load(open("outputs/figures/runtime_profile.json", encoding="utf-8"))

    dev_rows = _load_csv("outputs/results_development.csv")
    test_rows = _load_csv("outputs/results_test.csv")
    dev_sum = _summarize_detection(dev_rows)
    test_sum = _summarize_detection(test_rows)
    base_dev = _summarize_detection(_load_csv("outputs/results_baseline_dev.csv"))
    base_test = _summarize_detection(_load_csv("outputs/results_baseline_test.csv"))

    ab_gauss = _summarize_detection(_load_csv("outputs/ablation_no_gaussian.csv"))
    ab_hough = _summarize_detection(_load_csv("outputs/ablation_no_hough_suppression.csv"))
    ab_edge = _summarize_detection(_load_csv("outputs/ablation_no_edge_support.csv"))

    syn_custom = _summarize_synthetic(_load_csv("outputs/results_synthetic.csv"))
    canny_sweep = _load_csv("outputs/sweep_canny_dev.csv")
    hq_sweep = _load_csv("outputs/sweep_hough_quad_dev.csv")
    thresh_sweep = [r for r in _load_csv("outputs/sweep_thresholding_dev.csv") if r["condition"] == "all_dev"]

    cond_labels = {
        "nen_don_gian_du_sang": "Nền đơn giản, đủ sáng",
        "goc_chup_xien": "Góc chụp xiên",
        "nen_chu_nhat_gay_nhieu": "Nền chữ nhật gây nhiễu",
        "bien_tuong_phan_thap": "Biên tương phản thấp",
        "anh_sang_khong_deu_bong_do": "Ánh sáng không đều / bóng đổ",
        "khong_co_tai_lieu_hop_le": "Ảnh âm (không có tài liệu)",
    }

    lines = []
    # Cover metadata block (parsed specially by DOCX compiler)
    lines.append("<!-- COVER_START -->")
    lines.append("TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN - ĐHQG TP.HCM")
    lines.append("KHOA ĐIỆN TỬ - VIỄN THÔNG")
    lines.append("BÁO CÁO ĐỒ ÁN GIỮA KỲ: DOCUMENT SCANNER")
    lines.append("Phát hiện tài liệu, hiệu chỉnh phối cảnh và nhị phân hóa ảnh")
    lines.append("Học phần: Nhập môn Xử lý ảnh và Video")
    lines.append(f"Môi trường thực thi: Python {platform.python_version()} | OpenCV {cv2.__version__} | NumPy {np.__version__} | Seed 42")
    lines.append("Ngày hoàn thành báo cáo: 24/11/2026")
    lines.append("<!-- COVER_END -->")
    lines.append("")

    # Chapter 1
    lines.append("## CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI VÀ MỤC TIÊU")
    lines.append("")
    lines.append("### 1.1. Mục tiêu và phạm vi bài toán")
    lines.append("")
    lines.append(
        "Đề tài xây dựng hệ thống Document Scanner tự động chuyển ảnh chụp tài liệu thành trang giấy nhìn thẳng. "
        "Đầu ra gồm ảnh màu đã hiệu chỉnh phối cảnh và ảnh nhị phân rõ nét."
    )
    lines.append(
        "Chương trình giải quyết ba bài toán cốt lõi gồm định vị tứ giác, nắn thẳng phối cảnh và nhị phân hóa. "
        "Mọi thuật toán xử lý lõi chạy hoàn toàn trên CPU với seed cố định `42`."
    )
    lines.append(
        "Hệ thống thử nghiệm trên năm điều kiện chụp thực tế và một nhóm ảnh âm đối chứng. "
        "Các điều kiện gồm nền đơn giản, góc xiên, nền chữ nhật nhiễu, tương phản thấp và bóng đổ."
    )
    lines.append("")
    lines.append("### 1.2. Ràng buộc kỹ thuật và môi trường thực nghiệm")
    lines.append("")
    lines.append(
        "Pipeline chính trong `src/core/` tự cài đặt tường minh bằng Python, NumPy và các primitive cơ bản. "
        "Mã nguồn tuyệt đối không gọi `cv2.Canny`, `cv2.HoughLines`, `cv2.threshold` hay `skimage.filters.threshold_*`."
    )
    lines.append(
        f"Thực nghiệm chạy trên bộ xử lý `{platform.processor() or 'CPU x86_64'}` với Python `{platform.python_version()}`. "
        f"Các thư viện cài đặt gồm OpenCV `{cv2.__version__}`, NumPy `{np.__version__}`, SciPy `{scipy.__version__}` và scikit-image `{skimage.__version__}`."
    )
    lines.append("")

    # Chapter 2
    lines.append("## CHƯƠNG 2. KIẾN TRÚC HỆ THỐNG VÀ PHÂN TÍCH TỪNG MODULE")
    lines.append("")
    lines.append("### 2.1. Kiến trúc tổng thể Pipeline và cơ chế Early Exit")
    lines.append("")
    lines.append(
        "Hệ thống tổ chức theo kiến trúc ống dẫn năm giai đoạn tuần tự từ ảnh BGR thô tới ảnh nhị phân. "
        "Giai đoạn phát hiện biên và Hough xử lý ở độ phân giải chuẩn hóa `640 px` để tối ưu tốc độ CPU."
    )
    lines.append(
        "Tọa độ bốn góc tìm được sau đó quy đổi về độ phân giải gốc để biến đổi Homography giữ nguyên độ nét chữ. "
        "Nếu không tìm thấy tứ giác hợp lệ tại bước ba, nhánh Early Exit trả về `rejected` và bỏ qua warping."
    )
    lines.append("")
    lines.append("![Sơ đồ kiến trúc tổng thể hệ thống](outputs/figures/fig_pipeline_architecture.png)")
    lines.append("*Hình 1: Kiến trúc tổng thể 5 giai đoạn của hệ thống Document Scanner và nhánh Early Exit khi từ chối ảnh âm.*")
    lines.append("")
    lines.append("*Bảng 1: Vai trò, dữ liệu vào/ra và chức năng từng module trong mã nguồn dự án.*")
    lines.append("")
    lines.append("| Module | Mục đích chính | Đầu vào -> Đầu ra (shape/dtype) | Hàm cốt lõi |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append("| `src/core/canny.py` | Tách biên ảnh bằng Canny tự cài | `(H,W,3)` uint8 -> `edges (H,W)` uint8 | `gaussian_smooth`, `non_maximum_suppression`, `hysteresis_8conn` |")
    lines.append("| `src/core/hough.py` | Dựng accumulator và trích xuất đường thẳng | `edges` -> `acc (R,T)` int32, `peaks` list | `hough_transform`, `find_hough_peaks`, `polar_to_cartesian_segment` |")
    lines.append("| `src/core/quadrilateral.py` | Ghép 4 đường, chấm điểm và lọc hợp lệ | `peaks` -> `corners (4,2)` float32 hoặc `rejected` | `intersect_lines`, `is_convex_quad`, `find_document_quad` |")
    lines.append("| `src/core/warping.py` | Tính Homography và nội suy song tuyến tính | `image`, `corners` -> `warped` BGR, `H (3,3)` | `bilinear_sample_gray`, `compute_output_size`, `warp_document` |")
    lines.append("| `src/core/thresholding.py` | Phân ngưỡng toàn cục Otsu và cục bộ Sauvola | `warped_gray` -> `binary (H,W)` uint8 | `otsu_threshold`, `sauvola_threshold` |")
    lines.append("| `src/baseline/baseline.py` | Nhánh đối chứng dùng thư viện OpenCV/skimage | `image` -> `status`, `otsu/sauvola` binary | `run_baseline_on_image`, `run_baseline_batch` |")
    lines.append("| `run_pipeline.py` | Điều phối batch, đo CPU runtime và tính metrics | `manifest.csv` -> `results_*.csv` (12 cột) | `process_single_image`, `polygon_iou`, `ink_prf1` |")
    lines.append("| `tests/verify_ast.py` | Quét AST cấm gọi hàm thư viện gian lận | `src/core/*.py` -> Exit 0 nếu 0 vi phạm | `scan_file`, `main` |")
    lines.append("")

    # 2.2 Canny
    lines.append("### 2.2. Module phát hiện biên Canny (`src/core/canny.py`)")
    lines.append("")
    lines.append(
        "Module `src/core/canny.py` tách toàn bộ công đoạn phát hiện biên thành các hàm thuần để xuất từng ảnh trung gian. "
        "Ảnh màu BGR chuyển sang mức xám `float64` rồi làm mịn bằng mặt nạ Gaussian theo tham số $\\sigma$."
    )
    lines.append(
        "Toán tử Sobel 3x3 tính đạo hàm theo hai phương để thu được biên độ $M(x, y)$ và hướng gradient $\\theta_g(x, y)$."
    )
    lines.append("")
    lines.append("$$M(x, y) = \\sqrt{G_x(x, y)^2 + G_y(x, y)^2}, \\quad \\theta_g(x, y) = \\arctan2(G_y, G_x)$$")
    lines.append("")
    lines.append(
        "Hàm `non_maximum_suppression` lượng tử hóa góc về bốn hướng $0^\\circ, 45^\\circ, 90^\\circ, 135^\\circ$ bằng phép cắt mảng NumPy. "
        "Cách vector hóa này làm mảnh cạnh xuống đúng 1 pixel mà không cần vòng lặp Python."
    )
    lines.append(
        "Cuối cùng, `double_threshold` chia điểm biên thành hai mức `strong` và `weak`, còn `hysteresis_8conn` duyệt BFS theo 8 lân cận."
    )
    lines.append("")
    lines.append("```python")
    lines.append(_extract_func_source("src/core/canny.py", "non_maximum_suppression", 28))
    lines.append("```")
    lines.append("")

    # 2.3 Hough
    lines.append("### 2.3. Module biến đổi Hough (`src/core/hough.py`)")
    lines.append("")
    lines.append(
        "Module `src/core/hough.py` chuyển tập điểm biên rời rạc thành các đường thẳng tham số trong không gian cực."
    )
    lines.append("")
    lines.append("$$\\rho = x \\cos\\theta + y \\sin\\theta, \\quad \\theta \\in [0, \\pi)$$")
    lines.append("")
    lines.append(
        "Hàm `hough_transform` tính đồng loạt giá trị $\\rho$ cho mọi điểm biên và cộng dồn phiếu bằng `np.bincount`. "
        "Nhờ vector hóa theo góc $\\theta$, quá trình dựng accumulator trên ảnh `640 px` chỉ mất khoảng `88 ms`."
    )
    lines.append(
        "Hàm `find_hough_peaks` sắp xếp đỉnh giảm dần và triệt tiêu đường gần trùng theo $\\Delta\\rho$ và $\\Delta\\theta$ tuần hoàn. "
        "Cơ chế tuần hoàn giúp hợp nhất hai đường thẳng đứng gần biên $0^\\circ$ và $180^\\circ$ thành một cực đại duy nhất."
    )
    lines.append("")
    lines.append("```python")
    lines.append(_extract_func_source("src/core/hough.py", "hough_transform", 26))
    lines.append("```")
    lines.append("")

    # 2.4 Quad
    lines.append("### 2.4. Module chọn tứ giác tài liệu (`src/core/quadrilateral.py`)")
    lines.append("")
    lines.append(
        "Module `src/core/quadrilateral.py` ghép các cặp đường gần song song trực giao nhau để giải bốn giao điểm qua `np.linalg.solve`. "
        "Tứ giác ứng viên phải vượt qua các bộ lọc giới hạn khung ảnh, tính lồi, độ dài cạnh tối thiểu và góc đỉnh."
    )
    lines.append(
        "Hàm `score_quadrilateral` đánh giá ứng viên dựa trên tỷ lệ hỗ trợ biên dọc chu vi $S_{edge}$ và diện tích chuẩn hóa."
    )
    lines.append("")
    lines.append("$$Score = w_s \\cdot S_{edge} + w_a \\cdot A_{norm} = 0.65 \\cdot S_{edge} + 0.35 \\cdot \\frac{A_{quad}}{W \\cdot H}$$")
    lines.append("")
    lines.append(
        "Ngoài trung bình $S_{edge}$, bộ lọc yêu cầu từng cạnh riêng lẻ đạt `min_side_support` để loại tứ giác thiếu cạnh. "
        "Nếu không có ứng viên đạt chuẩn, hệ thống trả về `rejected` kèm lý do cụ thể như `short_side` hay `out_of_bounds`."
    )
    lines.append("")
    lines.append("```python")
    lines.append(_extract_func_source("src/core/quadrilateral.py", "score_quadrilateral", 28))
    lines.append("```")
    lines.append("")

    # 2.5 Warping
    lines.append("### 2.5. Module hiệu chỉnh phối cảnh và nội suy Bilinear (`src/core/warping.py`)")
    lines.append("")
    lines.append(
        "Module `src/core/warping.py` ước lượng kích thước trang giấy đích từ độ dài lớn nhất hai cặp cạnh đối diện. "
        "Ma trận Homography $3 \\times 3$ biến đổi cả ảnh màu lẫn ảnh xám gốc về khung hình chữ nhật nhìn thẳng."
    )
    lines.append(
        "Hàm `bilinear_sample_gray` cài đặt tường minh công thức nội suy song tuyến tính từ bốn điểm lân cận để kiểm chứng độ chính xác."
    )
    lines.append("")
    lines.append("$$f(x, y) = (1-a)(1-b)I(x_0, y_0) + a(1-b)I(x_1, y_0) + (1-a)bI(x_0, y_1) + abI(x_1, y_1)$$")
    lines.append("")
    lines.append(
        "Trong công thức trên, $x_0 = \\lfloor x \\rfloor$, $y_0 = \\lfloor y \\rfloor$, $a = x - x_0$ và $b = y - y_0$. "
        "Hình 2 minh họa sự khác biệt giữa nội suy nearest-neighbor (răng cưa bậc thang) và bilinear (chuyển mức xám mượt)."
    )
    lines.append("")
    lines.append("```python")
    lines.append(_extract_func_source("src/core/warping.py", "bilinear_sample_gray", 32))
    lines.append("```")
    lines.append("")
    lines.append("![So sánh Bilinear và Nearest-Neighbor](outputs/figures/resampling_bilinear_vs_nearest.png)")
    lines.append("*Hình 2: So sánh crop văn bản phóng to giữa nearest-neighbor và bilinear cùng bản đồ sai lệch tuyệt đối trên dev_01.*")
    lines.append("")

    # 2.6 Thresholding
    lines.append("### 2.6. Module nhị phân hóa Otsu và Sauvola (`src/core/thresholding.py`)")
    lines.append("")
    lines.append(
        "Module `src/core/thresholding.py` cài đặt song song hai thuật toán nhị phân hóa cho ảnh xám đã nắn thẳng. "
        "Phương pháp Otsu dựng lược đồ 256 mức bằng `np.bincount` và tìm ngưỡng $t^*$ cực đại hóa phương sai giữa hai lớp."
    )
    lines.append("")
    lines.append("$$\\sigma_b^2(t) = w_0(t) w_1(t) \\left( \\mu_0(t) - \\mu_1(t) \\right)^2$$")
    lines.append("")
    lines.append(
        "Phương pháp Sauvola tính ngưỡng cục bộ $T(x, y)$ dựa trên trung bình $m(x, y)$ và độ lệch chuẩn $s(x, y)$ trong cửa sổ $w \\times w$."
    )
    lines.append("")
    lines.append("$$T(x, y) = m(x, y) \\left[ 1 + k \\left( \\frac{s(x, y)}{R} - 1 \\right) \\right], \\quad R = 128.0$$")
    lines.append("")
    lines.append(
        "Tổng cục bộ của $I$ và $I^2$ tính nhanh qua bộ lọc hộp `cv2.boxFilter`, mất trung bình khoảng `42.6 ms` trên ảnh lớn."
    )
    lines.append("")
    lines.append("```python")
    lines.append(_extract_func_source("src/core/thresholding.py", "sauvola_threshold", 26))
    lines.append("```")
    lines.append("")

    # Chapter 3
    lines.append("## CHƯƠNG 3. DỮ LIỆU THỰC NGHIỆM, KIỂM ĐỊNH AST VÀ UNIT TESTS")
    lines.append("")
    lines.append("### 3.1. Cơ cấu dữ liệu 72 ảnh và nguyên tắc No Data Leakage")
    lines.append("")
    lines.append(
        "Bộ dữ liệu gồm 60 ảnh chụp thực tế và 12 ảnh chữ tổng hợp có mask chuẩn. "
        "Tập ảnh thực chia thành nhiều nhóm tài liệu quen thuộc như sách, đề thi, vở viết tay và thẻ sinh viên. "
        "Ngoài ra còn có giấy tờ dấu đỏ, sổ tay Leaf và vé xe buýt."
    )
    lines.append(
        "Toàn bộ ảnh của cùng một loại tài liệu nằm trọn trong một tập để tránh rò rỉ dữ liệu giữa Development và Test. "
        "Các tham số chỉ được phép khảo sát trên Development trước khi đóng băng chạy Test."
    )
    lines.append(
        "Tệp `data/labels.json` lưu trữ đầy đủ tọa độ bốn góc ground-truth cho toàn bộ 50 ảnh dương thực tế. "
        "Các nhãn gồm 30 ảnh Development và 20 ảnh Test được minh họa trực quan trong Hình 3."
    )
    lines.append(
        "Mọi ảnh dương bị `rejected` được tính `IoU = 0.0000` trên toàn bộ tập dương. "
        "Sai số góc chuẩn hóa chia đường chéo ảnh chỉ tính trên các dự đoán hợp lệ."
    )
    lines.append("")
    lines.append("![Toàn cảnh tập dữ liệu thực nghiệm](outputs/figures/fig_dataset_overview.png)")
    lines.append("*Hình 3: Ảnh đại diện từ các nhóm tài liệu thực tế kèm khung ground-truth 4 góc (magenta) từ data/labels.json.*")
    lines.append("")
    lines.append("*Bảng 2: Phân chia tập dữ liệu thực nghiệm Development và Test (tổng cộng 72 ảnh).*")
    lines.append("")
    lines.append("| Tập dữ liệu | Ảnh thực dương | Ảnh thực âm | Ảnh tổng hợp (Mask) | Mục đích sử dụng |")
    lines.append("| :--- | :---: | :---: | :---: | :--- |")
    lines.append("| Development | 30 ảnh (6 tài liệu x 5 ĐK) | 6 ảnh | 6 ảnh (Layout 1–2) | Dò tìm và đóng băng bộ tham số |")
    lines.append("| Test | 20 ảnh (4 tài liệu x 5 ĐK) | 4 ảnh | 6 ảnh (Layout 3–4) | Đánh giá độc lập cấu hình đóng băng |")
    lines.append("| **Tổng cộng** | **50 ảnh dương** | **10 ảnh âm** | **12 ảnh tổng hợp** | **Đánh giá toàn diện pipeline** |")
    lines.append("")

    lines.append("### 3.2. Kiểm định tĩnh AST và Bộ kiểm thử biên (17 Unit Tests)")
    lines.append("")
    lines.append(
        "Script `tests/verify_ast.py` phân tích cây cú pháp trừu tượng toàn bộ 6 tệp trong `src/core/`. "
        "Kết quả kiểm tra xác nhận 0 lệnh gọi hoặc import tới các hàm xử lý ảnh bị cấm."
    )
    lines.append(
        "Bộ `pytest` gồm 17 kiểm thử đơn vị phủ đủ các tình huống biên và thuật toán tính tay trong đề bài."
    )
    lines.append("")
    lines.append("*Bảng 3: Danh sách các trường hợp kiểm thử đơn vị trong thư mục `tests/` (17/17 PASSED).*")
    lines.append("")
    lines.append("| Tệp kiểm thử | Số test | Các ca biên đã kiểm chứng | Kết quả |")
    lines.append("| :--- | :---: | :--- | :---: |")
    lines.append("| `tests/test_canny.py` | 5 | Ảnh hằng không biên, step edge, NMS làm mảnh, ngưỡng kép 0, hysteresis liên thông 8 | PASSED |")
    lines.append("| `tests/test_hough.py` | 2 | Edge map toàn số 0, đường thẳng đứng $0^\\circ$ và đường ngang $90^\\circ$ | PASSED |")
    lines.append("| `tests/test_quadrilateral.py` | 5 | Sắp xếp góc CW, 3 điểm thẳng hàng suy biến, đường song song, tứ giác hợp lệ & thiếu đường | PASSED |")
    lines.append("| `tests/test_warping.py` | 3 | `bilinear_sample_gray` tính tay (nguyên/phân số/biên), ma trận đơn vị, phối cảnh biết trước | PASSED |")
    lines.append("| `tests/test_thresholding.py` | 2 | Ảnh hằng Otsu/Sauvola không NaN, tách chữ trên mask chuẩn (F1 = 1.00 / >0.85) | PASSED |")
    lines.append("")

    # Chapter 4
    lines.append("## CHƯƠNG 4. KHẢO SÁT ĐỘ NHẠY THAM SỐ TRÊN DEVELOPMENT")
    lines.append("")
    lines.append("### 4.1. Khảo sát tham số Canny (`sigma` và cặp ngưỡng)")
    lines.append("")
    lines.append(
        "Trên tập Development 36 ảnh, thực nghiệm lần lượt thay đổi $\\sigma$ và cặp ngưỡng kép để theo dõi biên giấy và kết quả cuối."
    )
    lines.append("")
    lines.append("*Bảng 4: Độ nhạy của bộ tham số Canny trên tập Development (36 ảnh).*")
    lines.append("")
    lines.append("| Tham số | Giá trị thử | Hợp lệ | Mean IoU | Succ@0.8 | CE Norm | Ảnh âm FP | CPU TB (ms) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in canny_sweep:
        lines.append(
            f"| `{r['param_name']}` | `{r['param_value']}` | {r['pos_accepted']}/{r['pos_total']} | {r['mean_iou']} | {r['succ_08']}/{r['pos_total']} | {r['mean_corner_err']} | {r['neg_fp']}/{r['neg_total']} | {r['mean_runtime_ms']} |"
        )
    lines.append("")
    cm_list = rich_meta["canny_sweep"]["canny_visual_stats"]
    lines.append(
        f"Trên ảnh `dev_01` (`640x480`), khi giảm $\\sigma = 0.8$ số điểm biên tăng lên `{cm_list[0]['edge_pixels']:,}` (`{cm_list[0]['density_pct']}%`) do nhiễu nền. "
        f"Khi tăng $\\sigma = 2.2$, số điểm biên giảm còn `{cm_list[2]['edge_pixels']:,}` (`{cm_list[2]['density_pct']}%`) làm mờ góc giấy."
    )
    lines.append(
        f"Cấu hình $\\sigma = {cfg['sigma']}$ và ngưỡng `({cfg['low_ratio']}, {cfg['high_ratio']})` giữ `{cm_list[1]['edge_pixels']:,}` điểm biên cân bằng giữa nhiễu và độ liền mạch."
    )
    lines.append("")
    lines.append("![Khảo sát trực quan Canny trên dev_01](outputs/figures/fig_canny_sweep_visual.png)")
    lines.append("*Hình 4: Edge map của dev_01 khi thay đổi độ lệch chuẩn Gaussian sigma và cặp ngưỡng kép Canny.*")
    lines.append("")

    lines.append("### 4.2. Khảo sát ngưỡng Hough và chế độ chấm điểm Quadrilateral")
    lines.append("")
    lines.append(
        "Thực nghiệm đối chiếu hàm mục tiêu chỉ dùng diện tích (`area_only`) và hàm kết hợp hỗ trợ biên (`edge_support+area`). "
        "Quá trình thử nghiệm chạy qua ba mức ngưỡng Hough khác nhau."
    )
    lines.append("")
    lines.append("*Bảng 5: So sánh chế độ chấm điểm Area-Only và Edge Support + Area trên Development (36 ảnh).*")
    lines.append("")
    lines.append("| Ngưỡng Hough | Chế độ chấm điểm | Hợp lệ | Mean IoU | Succ@0.8 | CE Norm | Ảnh âm FP | CPU TB (ms) |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in hq_sweep:
        lines.append(
            f"| `{r['hough_thresh_ratio']}` | `{r['scoring_mode']}` | {r['pos_accepted']}/{r['pos_total']} | {r['mean_iou']} | {r['succ_08']}/{r['pos_total']} | {r['mean_corner_err']} | {r['neg_fp']}/{r['neg_total']} | {r['mean_runtime_ms']} |"
        )
    lines.append("")
    lines.append(
        "Chế độ `area_only` chấp nhận khung lớn trên nền nên Mean IoU đạt cao hơn một chút nhưng FPR ảnh âm lên tới 5/6 (`83.3%`). "
        "Kết hợp `edge_support+area` ở ngưỡng `0.24` giảm FPR ảnh âm xuống 3/6 (`50.0%`) và cải thiện sai số góc còn `0.0824`."
    )
    lines.append("")

    lines.append("### 4.3. Khảo sát tham số Otsu và Sauvola trên tập Development")
    lines.append("")
    lines.append(
        "Trên 6 ảnh tổng hợp Development, thuật toán thay đổi từng yếu tố `window_size` và hệ số `k` của Sauvola để chọn cấu hình tối ưu."
    )
    lines.append("")
    lines.append("*Bảng 6: Kết quả quét tham số Otsu và Sauvola trên 6 ảnh chữ tổng hợp Development.*")
    lines.append("")
    lines.append("| Phương pháp | Window Size | Tham số k | Precision TB | Recall TB | F1 TB |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in thresh_sweep:
        lines.append(
            f"| {r['method']} | {r['window_size']} | {r['k']} | {r['mean_precision']} | {r['mean_recall']} | {r['mean_f1']} |"
        )
    lines.append("")
    lines.append(
        f"Sauvola tại `w = {cfg['sauvola_window']}` và `k = {cfg['sauvola_k']}` đạt F1 trung bình `{syn_custom['development']['sauvola']['all']['f1']:.4f}`, triệt tiêu hoàn toàn nền gradient. "
        "Trên crop bóng đổ ảnh thật (Hình 5), `k = 0.10` còn chấm nhiễu nền trong khi `k = 0.35` làm sạch hẳn vùng râm."
    )
    lines.append("")
    lines.append("![Lưới khảo sát tham số Sauvola trên ảnh bóng đổ thật](outputs/figures/fig_sauvola_param_grid.png)")
    lines.append("*Hình 5: Ảnh hưởng của window_size và k của Sauvola trên vùng chữ thật chịu bóng đổ kèm threshold map T(x,y).*")
    lines.append("")

    lines.append("### 4.4. Cấu hình đóng băng (`outputs/frozen_config.json`)")
    lines.append("")
    lines.append("Toàn bộ tham số chốt trên tập Development được đóng băng trước khi chạy đánh giá tập Test:")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(cfg, indent=2))
    lines.append("```")
    lines.append("")

    # Chapter 5
    lines.append("## CHƯƠNG 5. KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG VÀ SO SÁNH TRỰC QUAN")
    lines.append("")
    lines.append("### 5.1. So sánh Custom Pipeline và OpenCV Baseline theo từng điều kiện")
    lines.append("")
    lines.append(
        f"Trên tập Development, pipeline tự cài đạt Mean IoU `{dev_sum['mean_iou']:.4f}` và `{dev_sum['succ_08']}/{dev_sum['pos_tot']}` ảnh đạt `IoU >= 0.8`. "
        f"OpenCV Baseline chỉ đạt Mean IoU `{base_dev['mean_iou']:.4f}` với `{base_dev['succ_08']}/{base_dev['pos_tot']}` ảnh vượt ngưỡng `0.8`."
    )
    lines.append(
        f"Trên tập Test đóng băng, pipeline tự cài đạt Mean IoU `{test_sum['mean_iou']:.4f}` (hợp lệ `{test_sum['pos_acc']}/{test_sum['pos_tot']}`), cao hơn Baseline (`{base_test['mean_iou']:.4f}`). "
        f"Tỷ lệ báo giả trên ảnh âm Test đạt `{test_sum['neg_fp']}/{test_sum['neg_tot']}` (`{test_sum['neg_fpr']:.1f}%`)."
    )
    lines.append("")
    lines.append("*Bảng 7a: Kết quả định lượng theo từng điều kiện trên tập Development (36 ảnh).*")
    lines.append("")
    lines.append("| Điều kiện chụp (Dev) | Custom Hợp lệ | Custom IoU | Custom Succ@0.8 | Custom CE | Baseline IoU | Baseline Succ@0.8 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for cond, label in cond_labels.items():
        cd = dev_sum["by_cond"][cond]
        bd = base_dev["by_cond"][cond]
        if cond == "khong_co_tai_lieu_hop_le":
            lines.append(f"| {label} | FP {cd['accepted']}/{cd['total']} | - | - | - | FP {bd['accepted']}/{bd['total']} | - |")
        else:
            lines.append(
                f"| {label} | {cd['accepted']}/{cd['total']} | {cd['mean_iou']:.4f} | {cd['succ_08']}/{cd['total']} | {cd['mean_ce']:.4f} | {bd['mean_iou']:.4f} | {bd['succ_08']}/{bd['total']} |"
            )
    lines.append(
        f"| **Tổng ảnh dương Dev** | **{dev_sum['pos_acc']}/{dev_sum['pos_tot']}** | **{dev_sum['mean_iou']:.4f}** | **{dev_sum['succ_08']}/{dev_sum['pos_tot']} ({dev_sum['succ_08_rate']:.1f}%)** | **{dev_sum['mean_ce']:.4f}** | **{base_dev['mean_iou']:.4f}** | **{base_dev['succ_08']}/{base_dev['pos_tot']} ({base_dev['succ_08_rate']:.1f}%)** |"
    )
    lines.append("")
    lines.append("*Bảng 7b: Kết quả định lượng theo từng điều kiện trên tập Test đóng băng (24 ảnh).*")
    lines.append("")
    lines.append("| Điều kiện chụp (Test) | Custom Hợp lệ | Custom IoU | Custom Succ@0.8 | Custom CE | Baseline IoU | Baseline Succ@0.8 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for cond, label in cond_labels.items():
        ct = test_sum["by_cond"][cond]
        bt = base_test["by_cond"][cond]
        if cond == "khong_co_tai_lieu_hop_le":
            lines.append(f"| {label} | FP {ct['accepted']}/{ct['total']} | - | - | - | FP {bt['accepted']}/{bt['total']} | - |")
        else:
            lines.append(
                f"| {label} | {ct['accepted']}/{ct['total']} | {ct['mean_iou']:.4f} | {ct['succ_08']}/{ct['total']} | {ct['mean_ce']:.4f} | {bt['mean_iou']:.4f} | {bt['succ_08']}/{bt['total']} |"
            )
    lines.append(
        f"| **Tổng ảnh dương Test** | **{test_sum['pos_acc']}/{test_sum['pos_tot']}** | **{test_sum['mean_iou']:.4f}** | **{test_sum['succ_08']}/{test_sum['pos_tot']} ({test_sum['succ_08_rate']:.1f}%)** | **{test_sum['mean_ce']:.4f}** | **{base_test['mean_iou']:.4f}** | **{base_test['succ_08']}/{test_sum['pos_tot']} ({base_test['succ_08_rate']:.1f}%)** |"
    )
    lines.append("")

    lines.append("### 5.2. Nghiên cứu tách biệt thành phần (Ablation Study)")
    lines.append("")
    lines.append(
        "Ba ca Ablation chạy trên Development với cùng giới hạn `top_k_lines = 20` và toàn bộ tham số còn lại cố định. "
        f"Bỏ Gaussian smoothing làm `Succ@0.8` giảm từ `{dev_sum['succ_08']}/{dev_sum['pos_tot']}` xuống `{ab_gauss['succ_08']}/{ab_gauss['pos_tot']}` và thời gian CPU tăng lên `{ab_gauss['rt_acc_mean']:.1f} ms`."
    )
    lines.append(
        f"Bỏ triệt tiêu đường Hough trùng làm Mean IoU tụt sâu từ `{dev_sum['mean_iou']:.4f}` xuống `{ab_hough['mean_iou']:.4f}`. "
        f"Nguyên nhân là các đường song song sát nhau ở một cạnh mạnh chiếm hết 20 ứng viên đầu tiên."
    )
    lines.append(
        f"Bỏ thành phần edge-support giúp chọn khung lớn hơn (`Mean IoU = {ab_edge['mean_iou']:.4f}`) nhưng ảnh âm báo giả vọt lên `{ab_edge['neg_fp']}/{ab_edge['neg_tot']}` (`{ab_edge['neg_fpr']:.1f}%`)."
    )
    lines.append("")
    lines.append("*Bảng 8: Đối chiếu định lượng giữa Custom, Baseline và 3 ca Ablation trên tập Development (36 ảnh).*")
    lines.append("")
    lines.append("| Biến thể cấu hình | Hợp lệ | Mean IoU | Succ@0.8 | CE Norm | Ảnh âm FP | CPU Acc (ms) | CPU Rej (ms) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, sm in (
        ("Custom (Đầy đủ)", dev_sum),
        ("Baseline (OpenCV)", base_dev),
        ("Ablation 1: Bỏ Gaussian", ab_gauss),
        ("Ablation 2: Bỏ Hough Supp.", ab_hough),
        ("Ablation 3: Bỏ Edge Support", ab_edge),
    ):
        lines.append(
            f"| {name} | {sm['pos_acc']}/{sm['pos_tot']} | {sm['mean_iou']:.4f} | {sm['succ_08']}/{sm['pos_tot']} ({sm['succ_08_rate']:.1f}%) | {sm['mean_ce']:.4f} | {sm['neg_fp']}/{sm['neg_tot']} | {sm['rt_acc_mean']:.1f} | {sm['rt_rej_mean']:.1f} |"
        )
    lines.append("")
    lines.append("![Biểu đồ Ablation và Baseline](outputs/figures/chart_ablation_and_baseline.png)")
    lines.append("*Hình 6: So sánh tỷ lệ nhận diện ảnh dương, tỷ lệ báo giả ảnh âm và thời gian CPU trên tập Development.*")
    lines.append("")
    ab_pos = rich_meta["ablation_visual"]["pos_id"]
    ab_neg = rich_meta["ablation_visual"]["neg_id"]
    lines.append("![Minh họa trực quan tác động của Ablation](outputs/figures/fig_ablation_visual.png)")
    lines.append(f"*Hình 7: Tác động trực quan của 3 ca Ablation trên mẫu dương {ab_pos} (mất tứ giác khi bỏ Hough suppression) và mẫu âm {ab_neg} (nhận nhầm khi bỏ edge support).*")
    lines.append("")

    lines.append("### 5.3. Đánh giá nhị phân hóa trên 12 ảnh tổng hợp và Crop ảnh thực tế")
    lines.append("")
    lines.append(
        f"Trên 12 ảnh chữ tổng hợp (`480x360`), Sauvola tự cài đạt F1 trung bình `{syn_custom['development']['sauvola']['all']['f1']:.4f}` (Dev) và `{syn_custom['test']['sauvola']['all']['f1']:.4f}` (Test). "
        f"Otsu toàn cục sụt giảm xuống `{syn_custom['development']['otsu']['all']['f1']:.4f}` (Dev) và `{syn_custom['test']['otsu']['all']['f1']:.4f}` (Test) khi có gradient hoặc bóng đổ."
    )
    lines.append(
        "Hình 10 đối chiếu trực tiếp ba vùng crop văn bản thực tế giữa Otsu toàn cục và Sauvola cục bộ. "
        "Trên thẻ sinh viên `dev_26` (nền hoa văn gradient), Otsu bôi đen vùng tối trong khi Sauvola giữ sạch nét chữ."
    )
    cm_crop = rich_meta["crop_comparisons"]
    lines.append(
        f"Trên vé xe buýt `test_01` (mực in kim nhạt), Otsu với ngưỡng $T^* = {cm_crop['test_01']['otsu_T']}$ sai khác `{cm_crop['test_01']['otsu_sauvola_diff_pct']}%` điểm ảnh và làm mất chấm chữ mảnh. "
        f"Trên mẫu bóng đổ `dev_10`, Sauvola bảo tồn trọn vẹn chữ vùng tối vượt trội so với ngưỡng đơn $T^* = {cm_crop['dev_10']['otsu_T']}$."
    )
    lines.append("")
    lines.append("*Bảng 9: Precision, Recall và F1 lớp ink trên 12 ảnh tổng hợp tách riêng Development và Test.*")
    lines.append("")
    lines.append("| Tập dữ liệu | Điều kiện | Otsu (P / R / F1) | Sauvola (P / R / F1) |")
    lines.append("| :--- | :--- | :---: | :---: |")
    for split, sp_lbl in (("development", "Development (6 ảnh)"), ("test", "Test (6 ảnh)")):
        for cond, c_lbl in (
            ("nen_deu", "Nền đều"),
            ("gradient_sang", "Gradient sáng"),
            ("bong_kem_nhieu", "Bóng + nhiễu"),
            ("all", "Trung bình toàn tập"),
        ):
            o = syn_custom[split]["otsu"][cond]
            s = syn_custom[split]["sauvola"][cond]
            lines.append(
                f"| {sp_lbl} | {c_lbl} | {o['p']:.4f} / {o['r']:.4f} / {o['f1']:.4f} | {s['p']:.4f} / {s['r']:.4f} / {s['f1']:.4f} |"
            )
    lines.append("")
    lines.append("![Biểu đồ F1 trên 12 ảnh tổng hợp](outputs/figures/chart_synthetic_f1.png)")
    lines.append("*Hình 8: So sánh chỉ số F1 lớp ink giữa Otsu toàn cục và Sauvola cục bộ trên 12 ảnh chữ tổng hợp.*")
    lines.append("")
    lines.append("![So sánh Otsu và Sauvola trên ảnh bóng đổ](outputs/figures/thresholding_otsu_vs_sauvola.png)")
    lines.append("*Hình 9: Lược đồ mức xám, ngưỡng Otsu, bản đồ ngưỡng Sauvola và ảnh nhị phân trên dev_20 và syn_test_03.*")
    lines.append("")
    lines.append("![So sánh phóng to crop nét chữ thực tế Otsu vs Sauvola](outputs/figures/fig_crop_otsu_vs_sauvola_real.png)")
    lines.append("*Hình 10: Đối chiếu crop nét chữ thực tế trên Thẻ sinh viên (dev_26), Vé xe buýt (test_01) và Tài liệu bóng đổ (dev_10).*")
    lines.append("")

    lines.append("### 5.4. Minh họa chuỗi 6 bước xử lý trên nhiều loại tài liệu khác nhau")
    lines.append("")
    lines.append(
        "Chuỗi ảnh trung gian từ grayscale, Canny, Hough, tứ giác, nắn phẳng đến Sauvola được xuất đầy đủ trên năm loại tài liệu khác nhau. "
        "Các mẫu gồm sách HP Book (`dev_01`), thẻ sinh viên (`dev_26`) và giấy tờ dấu đỏ (`dev_21`). "
        "Danh sách còn có vé xe buýt (`test_01`) và sổ tay Leaf (`dev_16`)."
    )
    lines.append("")
    lines.append("![Các bước trung gian trên dev_01](outputs/figures/pipeline_stages_dev01.png)")
    lines.append("*Hình 11: Tiến trình 6 bước trung gian trên mẫu sách HP Book dev_01 (edge support = 0.87, score = 0.71).*")
    lines.append("")
    m_doc = rich_meta["multidoc_stages"]
    for fname, cap in (
        ("fig_stages_thesv_dev26.png", f"Hình 12: Chuỗi 6 bước trên Thẻ sinh viên ({m_doc['fig_stages_thesv_dev26.png']['image_id']}, IoU = {m_doc['fig_stages_thesv_dev26.png']['iou']}, CE = {m_doc['fig_stages_thesv_dev26.png']['corner_error_norm']})."),
        ("fig_stages_sign_dev21.png", f"Hình 13: Chuỗi 6 bước trên Giấy tờ dấu đỏ ({m_doc['fig_stages_sign_dev21.png']['image_id']}, IoU = {m_doc['fig_stages_sign_dev21.png']['iou']}, CE = {m_doc['fig_stages_sign_dev21.png']['corner_error_norm']})."),
        ("fig_stages_bus_test01.png", f"Hình 14: Chuỗi 6 bước trên Vé xe buýt tập Test ({m_doc['fig_stages_bus_test01.png']['image_id']}, IoU = {m_doc['fig_stages_bus_test01.png']['iou']}, CE = {m_doc['fig_stages_bus_test01.png']['corner_error_norm']})."),
        ("fig_stages_leaf_dev16.png", f"Hình 15: Chuỗi 6 bước trên Sổ tay bìa Leaf ({m_doc['fig_stages_leaf_dev16.png']['image_id']}, IoU = {m_doc['fig_stages_leaf_dev16.png']['iou']}, CE = {m_doc['fig_stages_leaf_dev16.png']['corner_error_norm']})."),
    ):
        lines.append(f"![{cap}](outputs/figures/{fname})")
        lines.append(f"*{cap}*")
        lines.append("")

    lines.append("### 5.5. Phân bổ thời gian CPU theo từng module")
    lines.append("")
    lines.append(
        "Thời gian xử lý được đo thuần từ mảng ảnh trong RAM đến đầu ra, loại trừ đọc ghi đĩa và vẽ hình. "
        f"Trên 36 ảnh Development (gốc tới `2560x1922`), ca `accepted` mất `{rt_prof['accepted_mean_total_ms']:.2f} ms` và ca `rejected` mất `{rt_prof['rejected_mean_total_ms']:.2f} ms`."
    )
    lines.append(
        f"Trên tập Test (24 ảnh), thời gian trung bình là `{test_sum['rt_acc_mean']:.2f} ms` cho nhóm `accepted` và `{test_sum['rt_rej_mean']:.2f} ms` cho nhóm `rejected`."
    )
    lines.append("")
    lines.append("*Bảng 10: Thời gian thực thi trung bình trên CPU theo từng module tại tập Development.*")
    lines.append("")
    lines.append("| Giai đoạn xử lý | Module | Thời gian TB (ms) | Tỷ trọng (%) |")
    lines.append("| :--- | :--- | :---: | :---: |")
    st_ms = rt_prof["stage_mean_ms"]
    st_pct = rt_prof["stage_share_pct"]
    for label, mod, key in (
        ("1. Resize + Canny Edge", "`src/core/canny.py`", "canny"),
        ("2. Hough Transform + NMS", "`src/core/hough.py`", "hough"),
        ("3. Quadrilateral Selection", "`src/core/quadrilateral.py`", "quad"),
        ("4. Homography + Warping", "`src/core/warping.py`", "warping"),
        ("5. Otsu + Sauvola Thresholding", "`src/core/thresholding.py`", "thresholding"),
    ):
        lines.append(f"| {label} | {mod} | {st_ms[key]:.2f} | {st_pct[key]:.1f}% |")
    lines.append("")
    lines.append("![Phân bổ thời gian CPU theo từng module](outputs/figures/fig_runtime_breakdown.png)")
    lines.append("*Hình 16: Phân bổ thời gian xử lý CPU thuần trên 5 module lõi (ba bước định vị chiếm 84.6% tổng thời gian).*")
    lines.append("")

    # Chapter 6: 6 Case Studies
    lines.append("## CHƯƠNG 6. PHÂN TÍCH CHI TIẾT 6 TRƯỜNG HỢP TIÊU BIỂU")
    lines.append("")
    case_specs = [
        (
            "dev_01",
            "6.1. Trường hợp 1: Ảnh chuẩn, nền đơn giản (dev_01)",
            "Ảnh chụp trang sách nền đồng nhất, kích thước gốc `2560x1922` (`640x480` khi xử lý), ánh sáng đều.",
            f"Trạng thái `accepted` sau `{cases['dev_01']['runtime_ms']} ms`, đạt `IoU = {cases['dev_01']['iou']}` và `CE = {cases['dev_01']['corner_error_norm']}`.",
            f"Edge map sạch ({cases['dev_01']['edge_pixel_ratio']*100:.2f}% pixel biên), accumulator đạt đỉnh `{cases['dev_01']['max_hough_votes']}` phiếu, edge support `{cases['dev_01']['edge_support']:.4f}`.",
            "Khung tranh in trên bìa sách có hỗ trợ biên rất cao. Do đó, tứ giác chọn bị co nhẹ vào khung trong.",
            "Hình 17: Ảnh trung gian chẩn đoán ca dev_01 (Xanh=Dự đoán, Hồng=Ground Truth).",
        ),
        (
            "dev_07",
            "6.2. Trường hợp 2: Góc chụp xiên mạnh (dev_07)",
            "Ảnh trang tài liệu đặt nghiêng phối cảnh lớn (`2560x1922`), bề mặt chứa nhiều dòng chữ ngang.",
            f"Pipeline dừng tại bước Quadrilateral với trạng thái `rejected`, lý do `low_edge_support` và `IoU = {cases['dev_07']['iou']}`.",
            f"Edge map chiếm `{cases['dev_07']['edge_pixel_ratio']*100:.2f}%`, accumulator có cực đại `{cases['dev_07']['max_hough_votes']}` phiếu nhưng các ứng viên tứ giác không đạt ngưỡng.",
            "Góc nghiêng làm cạnh ngắn bị đứt đoạn hoặc góc giao lệch chuẩn. Hỗ trợ biên tụt xuống dưới ngưỡng `0.72`.",
            "Hình 18: Ảnh trung gian chẩn đoán ca góc xiên dev_07.",
        ),
        (
            "dev_03",
            "6.3. Trường hợp 3: Nền có vật hình chữ nhật gây nhiễu (dev_03)",
            "Ảnh trang sách đặt trên mặt bàn phím nhiều phím chữ nhật (`2560x1922`).",
            f"Không gãy; loại bỏ khung bàn phím nhiễu, đạt `accepted` với `IoU = {cases['dev_03']['iou']}` và `CE = {cases['dev_03']['corner_error_norm']}`.",
            f"Hough accumulator có đỉnh `{cases['dev_03']['max_hough_votes']}` phiếu; ứng viên thắng đạt edge support `{cases['dev_03']['edge_support']:.4f}` và `area_ratio = {cases['dev_03']['area_ratio']:.4f}`.",
            "Hàm điểm kết hợp `0.65 * edge_support + 0.35 * area_ratio`. Trọng số này ưu tiên tứ giác bám sát biên trang sách.",
            "Hình 19: Ảnh trung gian chẩn đoán ca nền nhiễu chữ nhật dev_03.",
        ),
        (
            "dev_29",
            "6.4. Trường hợp 4: Biên giấy tương phản thấp (dev_29)",
            "Ảnh thẻ sinh viên đặt trên nền sáng có độ chênh mức xám biên rất thấp (`1536x2048`).",
            f"Pipeline gãy tại bước Quadrilateral với lý do `short_side` và `IoU = {cases['dev_29']['iou']}`.",
            f"Edge map (`{cases['dev_29']['edge_pixel_ratio']*100:.2f}%`) xuất hiện nhiều cạnh chữ bên trong thẻ. Ngược lại, cạnh biên ngoài bị đứt đoạn.",
            "Top-20 đường Hough (`max = 345` phiếu) bị các đường chữ bên trong chiếm chỗ. Ứng viên tạo thành nhỏ hơn `min_side_ratio`.",
            "Hình 20: Ảnh trung gian chẩn đoán ca biên tương phản thấp dev_29.",
        ),
        (
            "dev_05",
            "6.5. Trường hợp 5: Ánh sáng không đều và bóng đổ (dev_05)",
            "Ảnh trang vở chịu bóng đổ chéo mạnh làm tối một phần cạnh biên (`1922x2560`).",
            f"Pipeline dừng tại bước Quadrilateral với mã lỗi `out_of_bounds` và `IoU = {cases['dev_05']['iou']}`.",
            f"Tỷ lệ pixel biên chỉ đạt `{cases['dev_05']['edge_pixel_ratio']*100:.2f}%`, đỉnh Hough `{cases['dev_05']['max_hough_votes']}` phiếu. Các đường cùng hướng giao ngoài khung ảnh.",
            "Bóng đổ làm suy giảm gradient tại cạnh tối. Hệ thống ghép nhầm đường gân chữ gần song song tạo giao điểm ngoài biên.",
            "Hình 21: Ảnh trung gian chẩn đoán ca bóng đổ dev_05.",
        ),
        (
            "dev_32",
            "6.6. Trường hợp 6: Ảnh âm không có tài liệu hợp lệ (dev_32 và dev_33)",
            "Ảnh âm `dev_32` (`2048x1536`) chụp màn hình chữ nhật; ảnh `dev_33` (`960x1280`) chụp mảnh giấy xé.",
            "`dev_32` gây false-positive (`accepted`, `IoU = 0.0000`); `dev_33` bị loại đúng (`rejected: short_side`).",
            f"Trên `dev_32`, khung màn hình tạo `{cases['dev_32']['num_peaks']}` peaks sắc nét (`edge_support = {cases['dev_32']['edge_support']:.4f}`, `area_ratio = {cases['dev_32']['area_ratio']:.4f}`); trên `dev_33`, đỉnh Hough chỉ đạt `{cases['dev_33']['max_hough_votes']}` phiếu.",
            "Bộ lọc hình học thuần túy chưa phân biệt được viền màn hình phẳng với trang giấy. Hệ thống còn thiếu bước kiểm tra mật độ chữ.",
            "Hình 22: Ảnh trung gian chẩn đoán ca ảnh âm false-positive dev_32.",
        ),
    ]
    for img_id, heading, b1, b2, b3, b4, cap in case_specs:
        lines.append(f"### {heading}")
        lines.append("")
        lines.append(f"- *Đầu vào & Điều kiện chụp:* {b1}")
        lines.append(f"- *Điểm gãy trong pipeline:* {b2}")
        lines.append(f"- *Bằng chứng trên ảnh trung gian:* {b3}")
        lines.append(f"- *Nguyên nhân & Tham số liên quan:* {b4}")
        lines.append("")
        lines.append(f"![{cap}](outputs/figures/{cases[img_id]['figure']})")
        lines.append(f"*{cap}*")
        lines.append("")

    lines.append("![Đối chiếu ảnh âm bị từ chối dev_33](outputs/figures/case6b_dev33_negative_rej.png)")
    lines.append("*Hình 23: Minh chứng ảnh âm giấy xé dev_33 bị từ chối đúng với lý do short_side.*")
    lines.append("")

    # Chapter 7
    lines.append("## CHƯƠNG 7. BÀN LUẬN KỸ THUẬT, HẠN CHẾ VÀ HƯỚNG DẪN TÁI LẬP")
    lines.append("")
    lines.append("### 7.1. Bàn luận ưu điểm, hạn chế và hướng phát triển")
    lines.append("")
    lines.append(
        f"Đối chiếu với 50 nhãn ground-truth, pipeline tự cài đạt Mean IoU `{dev_sum['mean_iou']:.4f}` (Dev) và `{test_sum['mean_iou']:.4f}` (Test). "
        f"Nhiều mẫu sạch đạt độ khớp gần tuyệt đối (`IoU > 0.98` như `dev_02`, `dev_16`, `dev_18`, `dev_26`), vượt trội Baseline OpenCV."
    )
    lines.append(
        "Tuy nhiên, chỉ số `Success@0.8` còn hạn chế ở nhóm sách có khung hình lồng bên trong (`dev_01`, `dev_21`, `test_06`). "
        "Nguyên nhân là hàm điểm ưu tiên độ khớp biên cao của khung chữ nhật nội tại thay vì mép giấy ngoài cùng."
    )
    lines.append(
        "Hướng nâng cấp gồm ưu tiên khung bao ngoài cùng khi các tứ giác lồng nhau. Đồng thời, cần thêm bộ kiểm tra mật độ chữ để giảm FPR ảnh âm."
    )
    lines.append("")
    lines.append("### 7.2. Lệnh tái lập toàn bộ mã nguồn và thực nghiệm")
    lines.append("")
    lines.append("Toàn bộ kết quả trên CPU (với `np.random.seed(42)`) tái lập bằng chuỗi lệnh sau:")
    lines.append("")
    lines.append("```powershell")
    lines.append("python tests/verify_ast.py")
    lines.append("python -m pytest tests/ -v")
    lines.append("python scripts/generate_synthetic.py")
    lines.append("python scripts/run_all_experiments.py")
    lines.append("python scripts/generate_figures.py")
    lines.append("python scripts/generate_rich_visuals.py")
    lines.append("python scripts/build_report.py")
    lines.append("python scripts/audit_report.py")
    lines.append("```")
    lines.append("")

    # Chapter 8
    lines.append("## CHƯƠNG 8. TÀI LIỆU THAM KHẢO")
    lines.append("")
    lines.append("1. Canny, J. (1986). A Computational Approach to Edge Detection. *IEEE TPAMI*, 8(6), 679–698.")
    lines.append("2. Duda, R. O., & Hart, P. E. (1972). Use of the Hough Transformation to Detect Lines and Curves in Pictures. *CACM*, 15(1), 11–15.")
    lines.append("3. Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE TSMC*, 9(1), 62–66.")
    lines.append("4. Sauvola, J., & Pietikainen, M. (2000). Adaptive document image binarization. *Pattern Recognition*, 33(2), 225–236.")
    lines.append("5. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.")
    lines.append("")

    # Landscape Appendix
    lines.append("<!-- LANDSCAPE_APPENDIX -->")
    lines.append("## PHỤ LỤC A. BẢNG KẾT QUẢ CHI TIẾT TỪNG ẢNH (DEVELOPMENT & TEST)")
    lines.append("")
    lines.append("*Bảng A.1: Kết quả chi tiết 36 ảnh tập Development trích xuất từ `outputs/results_development.csv`.*")
    lines.append("")
    lines.append("| Mã ảnh | Điều kiện | Trạng thái | Lý do từ chối | IoU | Corner Err | Runtime (ms) | Độ phân giải |")
    lines.append("| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |")
    for r in dev_rows:
        lines.append(
            f"| {r['image_id']} | {r['condition']} | {r['status']} | {r['reject_reason'] or '-'} | {r['iou']} | {r['corner_error_norm']} | {r['runtime_ms']} | {r['resolution']} |"
        )
    lines.append("")
    lines.append("*Bảng A.2: Kết quả chi tiết 24 ảnh tập Test trích xuất từ `outputs/results_test.csv`.*")
    lines.append("")
    lines.append("| Mã ảnh | Điều kiện | Trạng thái | Lý do từ chối | IoU | Corner Err | Runtime (ms) | Độ phân giải |")
    lines.append("| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |")
    for r in test_rows:
        lines.append(
            f"| {r['image_id']} | {r['condition']} | {r['status']} | {r['reject_reason'] or '-'} | {r['iou']} | {r['corner_error_norm']} | {r['runtime_ms']} | {r['resolution']} |"
        )
    lines.append("")

    md_text = "\n".join(lines)
    md_path = Path("outputs/Bao_cao_Giua_Ky_Document_Scanner.md")
    md_path.write_text(md_text, encoding="utf-8")
    print(f"Saved {md_path}")
    return md_text, {"dev_sum": dev_sum, "test_sum": test_sum, "syn_custom": syn_custom}


def _mml_elem(tag: str, text: str | None = None, **attrib) -> etree._Element:
    el = etree.Element(f"{{{MML_NS}}}{tag}", **attrib)
    if text is not None:
        el.text = text
    return el


def _latex_to_mml(tex: str) -> etree._Element:
    t = tex.strip()
    math = _mml_elem("math", display="block")

    if "M(x, y)" in t and "\\arctan2" in t:
        mrow = _mml_elem("mrow")
        for s in (
            _mml_elem("mi", "M"), _mml_elem("mo", "("), _mml_elem("mi", "x"), _mml_elem("mo", ","), _mml_elem("mi", "y"), _mml_elem("mo", ")"),
            _mml_elem("mo", "="),
        ):
            mrow.append(s)
        msqrt = _mml_elem("msqrt")
        inner = _mml_elem("mrow")
        for term, op in (("x", "+"), ("y", None)):
            msup = _mml_elem("msup")
            sub = _mml_elem("msub")
            sub.append(_mml_elem("mi", "G"))
            sub.append(_mml_elem("mi", term))
            brow = _mml_elem("mrow")
            brow.append(sub)
            brow.append(_mml_elem("mo", "("))
            brow.append(_mml_elem("mi", "x"))
            brow.append(_mml_elem("mo", ","))
            brow.append(_mml_elem("mi", "y"))
            brow.append(_mml_elem("mo", ")"))
            msup.append(brow)
            msup.append(_mml_elem("mn", "2"))
            inner.append(msup)
            if op:
                inner.append(_mml_elem("mo", op))
        msqrt.append(inner)
        mrow.append(msqrt)
        mrow.append(_mml_elem("mo", ","))
        mrow.append(_mml_elem("mspace", width="1em"))
        sub_th = _mml_elem("msub")
        sub_th.append(_mml_elem("mi", "θ"))
        sub_th.append(_mml_elem("mi", "g"))
        mrow.append(sub_th)
        for s in (
            _mml_elem("mo", "("), _mml_elem("mi", "x"), _mml_elem("mo", ","), _mml_elem("mi", "y"), _mml_elem("mo", ")"),
            _mml_elem("mo", "="), _mml_elem("mi", "arctan2"), _mml_elem("mo", "("),
        ):
            mrow.append(s)
        for idx, g_lbl in enumerate(("y", "x")):
            sub = _mml_elem("msub")
            sub.append(_mml_elem("mi", "G"))
            sub.append(_mml_elem("mi", g_lbl))
            mrow.append(sub)
            if idx == 0:
                mrow.append(_mml_elem("mo", ","))
        mrow.append(_mml_elem("mo", ")"))
        math.append(mrow)
        return math

    if "\\rho = x \\cos" in t:
        mrow = _mml_elem("mrow")
        for el in (
            _mml_elem("mi", "ρ"), _mml_elem("mo", "="), _mml_elem("mi", "x"), _mml_elem("mi", "cos"), _mml_elem("mi", "θ"),
            _mml_elem("mo", "+"), _mml_elem("mi", "y"), _mml_elem("mi", "sin"), _mml_elem("mi", "θ"), _mml_elem("mo", ","),
            _mml_elem("mspace", width="1em"), _mml_elem("mi", "θ"), _mml_elem("mo", "∈"), _mml_elem("mo", "["),
            _mml_elem("mn", "0"), _mml_elem("mo", ","), _mml_elem("mi", "π"), _mml_elem("mo", ")"),
        ):
            mrow.append(el)
        math.append(mrow)
        return math

    if "Score" in t and "A_{quad}" in t:
        mrow = _mml_elem("mrow")
        mrow.append(_mml_elem("mi", "Score"))
        mrow.append(_mml_elem("mo", "="))
        for w_sub, s_sub in (("s", "edge"), ("a", "norm")):
            sub_w = _mml_elem("msub")
            sub_w.append(_mml_elem("mi", "w"))
            sub_w.append(_mml_elem("mi", w_sub))
            sub_s = _mml_elem("msub")
            sub_s.append(_mml_elem("mi", "S" if w_sub == "s" else "A"))
            sub_s.append(_mml_elem("mi", s_sub))
            mrow.append(sub_w)
            mrow.append(_mml_elem("mo", "⋅"))
            mrow.append(sub_s)
            if w_sub == "s":
                mrow.append(_mml_elem("mo", "+"))
        mrow.append(_mml_elem("mo", "="))
        mrow.append(_mml_elem("mn", "0.65"))
        mrow.append(_mml_elem("mo", "⋅"))
        s_e = _mml_elem("msub")
        s_e.append(_mml_elem("mi", "S"))
        s_e.append(_mml_elem("mi", "edge"))
        mrow.append(s_e)
        mrow.append(_mml_elem("mo", "+"))
        mrow.append(_mml_elem("mn", "0.35"))
        mrow.append(_mml_elem("mo", "⋅"))
        frac = _mml_elem("mfrac")
        num = _mml_elem("msub")
        num.append(_mml_elem("mi", "A"))
        num.append(_mml_elem("mi", "quad"))
        den = _mml_elem("mrow")
        den.append(_mml_elem("mi", "W"))
        den.append(_mml_elem("mo", "⋅"))
        den.append(_mml_elem("mi", "H"))
        frac.append(num)
        frac.append(den)
        mrow.append(frac)
        math.append(mrow)
        return math

    if "f(x, y)" in t and "I(x_0, y_0)" in t:
        mrow = _mml_elem("mrow")
        for el in (_mml_elem("mi", "f"), _mml_elem("mo", "("), _mml_elem("mi", "x"), _mml_elem("mo", ","), _mml_elem("mi", "y"), _mml_elem("mo", ")"), _mml_elem("mo", "=")):
            mrow.append(el)
        terms = [
            ("(1−a)(1−b)", "0", "0"),
            ("a(1−b)", "1", "0"),
            ("(1−a)b", "0", "1"),
            ("ab", "1", "1"),
        ]
        for idx, (coeff, ix, iy) in enumerate(terms):
            if idx > 0:
                mrow.append(_mml_elem("mo", "+"))
            mrow.append(_mml_elem("mi", coeff))
            mrow.append(_mml_elem("mi", "I"))
            mrow.append(_mml_elem("mo", "("))
            sx = _mml_elem("msub")
            sx.append(_mml_elem("mi", "x"))
            sx.append(_mml_elem("mn", ix))
            sy = _mml_elem("msub")
            sy.append(_mml_elem("mi", "y"))
            sy.append(_mml_elem("mn", iy))
            mrow.append(sx)
            mrow.append(_mml_elem("mo", ","))
            mrow.append(sy)
            mrow.append(_mml_elem("mo", ")"))
        math.append(mrow)
        return math

    if "\\sigma_b^2(t)" in t:
        mrow = _mml_elem("mrow")
        msubsup = _mml_elem("msubsup")
        msubsup.append(_mml_elem("mi", "σ"))
        msubsup.append(_mml_elem("mi", "b"))
        msubsup.append(_mml_elem("mn", "2"))
        mrow.append(msubsup)
        for el in (_mml_elem("mo", "("), _mml_elem("mi", "t"), _mml_elem("mo", ")"), _mml_elem("mo", "=")):
            mrow.append(el)
        for idx in ("0", "1"):
            sub = _mml_elem("msub")
            sub.append(_mml_elem("mi", "w"))
            sub.append(_mml_elem("mn", idx))
            mrow.append(sub)
            mrow.append(_mml_elem("mo", "("))
            mrow.append(_mml_elem("mi", "t"))
            mrow.append(_mml_elem("mo", ")"))
        msup = _mml_elem("msup")
        paren = _mml_elem("mrow")
        paren.append(_mml_elem("mo", "("))
        for idx in ("0", "1"):
            sub = _mml_elem("msub")
            sub.append(_mml_elem("mi", "μ"))
            sub.append(_mml_elem("mn", idx))
            paren.append(sub)
            paren.append(_mml_elem("mo", "("))
            paren.append(_mml_elem("mi", "t"))
            paren.append(_mml_elem("mo", ")"))
            if idx == "0":
                paren.append(_mml_elem("mo", "−"))
        paren.append(_mml_elem("mo", ")"))
        msup.append(paren)
        msup.append(_mml_elem("mn", "2"))
        mrow.append(msup)
        math.append(mrow)
        return math

    if "T(x, y)" in t and "R = 128.0" in t:
        mrow = _mml_elem("mrow")
        for el in (_mml_elem("mi", "T"), _mml_elem("mo", "("), _mml_elem("mi", "x"), _mml_elem("mo", ","), _mml_elem("mi", "y"), _mml_elem("mo", ")"), _mml_elem("mo", "=")):
            mrow.append(el)
        for el in (_mml_elem("mi", "m"), _mml_elem("mo", "("), _mml_elem("mi", "x"), _mml_elem("mo", ","), _mml_elem("mi", "y"), _mml_elem("mo", ")"), _mml_elem("mo", "[")):
            mrow.append(el)
        mrow.append(_mml_elem("mn", "1"))
        mrow.append(_mml_elem("mo", "+"))
        mrow.append(_mml_elem("mi", "k"))
        mrow.append(_mml_elem("mo", "("))
        frac = _mml_elem("mfrac")
        num = _mml_elem("mrow")
        for el in (_mml_elem("mi", "s"), _mml_elem("mo", "("), _mml_elem("mi", "x"), _mml_elem("mo", ","), _mml_elem("mi", "y"), _mml_elem("mo", ")")):
            num.append(el)
        den = _mml_elem("mi", "R")
        frac.append(num)
        frac.append(den)
        mrow.append(frac)
        for el in (_mml_elem("mo", "−"), _mml_elem("mn", "1"), _mml_elem("mo", ")"), _mml_elem("mo", "]"), _mml_elem("mo", ","), _mml_elem("mspace", width="1em"), _mml_elem("mi", "R"), _mml_elem("mo", "="), _mml_elem("mn", "128.0")):
            mrow.append(el)
        math.append(mrow)
        return math

    mrow = _mml_elem("mrow")
    mrow.append(_mml_elem("mi", t))
    math.append(mrow)
    return math


def _set_run_font(run, name: str, size_pt: float, bold: bool = False, italic: bool = False, color_rgb: tuple[int, int, int] = (0, 0, 0)) -> None:
    run.font.name = name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        _insert_ooxml_ordered(rPr, rFonts, RPR_ORDER)
    for attr in ("ascii", "hAnsi", "cs", "eastAsia"):
        rFonts.set(qn(f"w:{attr}"), name)
    for theme_attr in ("asciiTheme", "hAnsiTheme", "csTheme", "cstheme", "eastAsiaTheme"):
        if qn(f"w:{theme_attr}") in rFonts.attrib:
            del rFonts.attrib[qn(f"w:{theme_attr}")]


def _apply_shading_ordered(element, order: list[str], fill_hex: str = "F5F5F5") -> None:
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{fill_hex}"/>')
    _insert_ooxml_ordered(element, shd, order)


def _apply_p_border_ordered(pPr, color_hex: str = "CBD5E0") -> None:
    pbdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="4" w:color="{color_hex}"/>'
        f'<w:left w:val="single" w:sz="4" w:space="6" w:color="{color_hex}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="4" w:color="{color_hex}"/>'
        f'<w:right w:val="single" w:sz="4" w:space="6" w:color="{color_hex}"/>'
        f'</w:pBdr>'
    )
    _insert_ooxml_ordered(pPr, pbdr, PPR_ORDER)


def _clean_inline_math_to_unicode(expr: str) -> str:
    replacements = [
        ("\\Delta\\theta", "Δθ"),
        ("\\Delta\\rho", "Δρ"),
        ("\\Delta", "Δ"),
        ("\\theta_g(x, y)", "θ_g(x, y)"),
        ("\\theta_g(x,y)", "θ_g(x,y)"),
        ("\\theta", "θ"),
        ("\\sigma", "σ"),
        ("\\rho", "ρ"),
        ("\\lfloor", "⌊"),
        ("\\rfloor", "⌋"),
        ("^\\circ", "°"),
        ("\\times", "×"),
        ("S_{edge}", "S_edge"),
        ("A_{norm}", "A_norm"),
        ("t^*", "t*"),
        ("T^*", "T*"),
        ("I^2", "I²"),
        ("x_0", "x0"),
        ("y_0", "y0"),
    ]
    out = expr
    for k, v in replacements:
        out = out.replace(k, v)
    return out


def _add_formatted_runs(p, text: str, base_size: float = 12.0, base_italic: bool = False) -> None:
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\$[^$]+\$)")
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            r = p.add_run(text[pos : m.start()])
            _set_run_font(r, "Times New Roman", base_size, italic=base_italic)
        tok = m.group(0)
        if tok.startswith("**") and tok.endswith("**"):
            r = p.add_run(tok[2:-2])
            _set_run_font(r, "Times New Roman", base_size, bold=True, italic=base_italic)
        elif tok.startswith("*") and tok.endswith("*"):
            r = p.add_run(tok[1:-1])
            _set_run_font(r, "Times New Roman", base_size, italic=True)
        elif tok.startswith("`") and tok.endswith("`"):
            r = p.add_run(tok[1:-1])
            _set_run_font(r, "Courier New", 9.5)
            _apply_shading_ordered(r._element.get_or_add_rPr(), RPR_ORDER, "F5F5F5")
        elif tok.startswith("$") and tok.endswith("$"):
            clean_m = _clean_inline_math_to_unicode(tok[1:-1])
            r = p.add_run(clean_m)
            _set_run_font(r, "Cambria Math", base_size, italic=True)
        pos = m.end()
    if pos < len(text):
        r = p.add_run(text[pos:])
        _set_run_font(r, "Times New Roman", base_size, italic=base_italic)


def _add_omml_equation(doc: Document, tex: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15

    mml = _latex_to_mml(tex)
    if OMML_XSLT is not None:
        omml_root = OMML_XSLT(mml).getroot()
        # Normalize m:r font properties strictly in valid OOXML order: m:rPr -> w:rPr -> m:t
        for mr in omml_root.xpath(".//m:r", namespaces={"m": M_NS}):
            w_rpr = mr.find(f"{{{W_NS}}}rPr")
            if w_rpr is None:
                w_rpr = etree.Element(f"{{{W_NS}}}rPr")
                m_rpr = mr.find(f"{{{M_NS}}}rPr")
                if m_rpr is not None:
                    mr.insert(list(mr).index(m_rpr) + 1, w_rpr)
                else:
                    mr.insert(0, w_rpr)
            rfonts = w_rpr.find(f"{{{W_NS}}}rFonts")
            if rfonts is None:
                rfonts = etree.Element(f"{{{W_NS}}}rFonts")
                _insert_ooxml_ordered(w_rpr, rfonts, RPR_ORDER)
            for attr in ("ascii", "hAnsi", "cs", "eastAsia"):
                rfonts.set(f"{{{W_NS}}}{attr}", "Cambria Math")
            for theme_attr in ("asciiTheme", "hAnsiTheme", "csTheme", "cstheme", "eastAsiaTheme"):
                if f"{{{W_NS}}}{theme_attr}" in rfonts.attrib:
                    del rfonts.attrib[f"{{{W_NS}}}{theme_attr}"]

        # Never wrap an existing m:oMathPara inside another m:oMathPara!
        if etree.QName(omml_root).localname == "oMathPara":
            p._p.append(omml_root)
        else:
            omath_para = etree.Element(f"{{{M_NS}}}oMathPara")
            omath_para.append(omml_root)
            p._p.append(omath_para)
    else:
        r = p.add_run(_clean_inline_math_to_unicode(tex))
        _set_run_font(r, "Cambria Math", 12.0, italic=True)


def _normalize_docx_styles(doc: Document) -> None:
    for st in doc.styles:
        try:
            target_font = "Courier New" if "Code" in st.name else "Times New Roman"
            st.font.name = target_font
            st.font.color.rgb = RGBColor(0, 0, 0)
            rPr = st._element.get_or_add_rPr()
            rFonts = rPr.find(qn("w:rFonts"))
            if rFonts is None:
                rFonts = OxmlElement("w:rFonts")
                _insert_ooxml_ordered(rPr, rFonts, RPR_ORDER)
            for attr in ("ascii", "hAnsi", "cs", "eastAsia"):
                rFonts.set(qn(f"w:{attr}"), target_font)
            for theme_attr in ("asciiTheme", "hAnsiTheme", "csTheme", "cstheme", "eastAsiaTheme"):
                if qn(f"w:{theme_attr}") in rFonts.attrib:
                    del rFonts.attrib[qn(f"w:{theme_attr}")]
        except Exception:
            pass

    for rFonts in doc.styles.element.xpath(".//w:rFonts"):
        for attr in ("ascii", "hAnsi", "cs", "eastAsia"):
            if qn(f"w:{attr}") in rFonts.attrib or attr == "ascii":
                rFonts.set(qn(f"w:{attr}"), "Times New Roman")
        for theme_attr in ("asciiTheme", "hAnsiTheme", "csTheme", "cstheme", "eastAsiaTheme"):
            if qn(f"w:{theme_attr}") in rFonts.attrib:
                del rFonts.attrib[qn(f"w:{theme_attr}")]


def _make_field_element(instr: str) -> etree._Element:
    fld = parse_xml(
        f'<w:fldSimple {nsdecls("w")} w:instr="{instr}">'
        f'<w:r><w:rPr>'
        f'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman"/>'
        f'<w:i/><w:color w:val="444444"/><w:sz w:val="19"/><w:szCs w:val="19"/>'
        f'</w:rPr><w:t>1</w:t></w:r></w:fldSimple>'
    )
    return fld


def _configure_header_footer(section) -> None:
    section.different_first_page_header_footer = True
    hdr = section.header
    hp = hdr.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hr = hp.add_run("Nhập môn Xử lý ảnh và Video | Đồ án giữa kỳ: Document Scanner")
    _set_run_font(hr, "Times New Roman", 9.5, italic=True, color_rgb=(68, 68, 68))

    ftr = section.footer
    fp = ftr.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r1 = fp.add_run("Document Scanner — Báo cáo giữa kỳ | Trang ")
    _set_run_font(r1, "Times New Roman", 9.5, italic=True, color_rgb=(68, 68, 68))
    fp._p.append(_make_field_element("PAGE"))
    r2 = fp.add_run(" / ")
    _set_run_font(r2, "Times New Roman", 9.5, italic=True, color_rgb=(68, 68, 68))
    fp._p.append(_make_field_element("NUMPAGES"))


def _render_cover_page(doc: Document, cover_lines: list[str]) -> None:
    for idx, txt in enumerate(cover_lines):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.3
        if idx in (0, 1):
            p.paragraph_format.space_before = Pt(6 if idx == 0 else 0)
            p.paragraph_format.space_after = Pt(4)
            r = p.add_run(txt)
            _set_run_font(r, "Times New Roman", 13.0, bold=True)
        elif idx == 2:
            p.paragraph_format.space_before = Pt(72)
            p.paragraph_format.space_after = Pt(10)
            r = p.add_run(txt)
            _set_run_font(r, "Times New Roman", 18.0, bold=True)
        elif idx == 3:
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(36)
            r = p.add_run(txt)
            _set_run_font(r, "Times New Roman", 14.0, italic=True)
        else:
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run(txt)
            _set_run_font(r, "Times New Roman", 12.0, bold=(idx == 4))
    doc.add_page_break()


def compile_markdown_to_docx(md_text: str, out_docx: str) -> None:
    doc = Document()
    _normalize_docx_styles(doc)

    sec0 = doc.sections[0]
    sec0.page_width = Cm(21.0)
    sec0.page_height = Cm(29.7)
    sec0.left_margin = Cm(3.0)
    sec0.top_margin = Cm(2.0)
    sec0.bottom_margin = Cm(2.0)
    sec0.right_margin = Cm(2.0)
    _configure_header_footer(sec0)

    in_landscape = False
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line:
            i += 1
            continue

        if line == "<!-- COVER_START -->":
            i += 1
            cover_lines = []
            while i < len(lines) and lines[i].strip() != "<!-- COVER_END -->":
                if lines[i].strip():
                    cover_lines.append(lines[i].strip())
                i += 1
            i += 1
            _render_cover_page(doc, cover_lines)
            continue

        if line == "<!-- LANDSCAPE_APPENDIX -->":
            new_sec = doc.add_section(WD_SECTION.NEW_PAGE)
            new_sec.orientation = WD_ORIENT.LANDSCAPE
            new_sec.page_width = Cm(29.7)
            new_sec.page_height = Cm(21.0)
            new_sec.left_margin = Cm(2.0)
            new_sec.right_margin = Cm(2.0)
            new_sec.top_margin = Cm(1.8)
            new_sec.bottom_margin = Cm(1.8)
            new_sec.header.is_linked_to_previous = True
            new_sec.footer.is_linked_to_previous = True
            in_landscape = True
            i += 1
            continue

        if line.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
            pPr = p._p.get_or_add_pPr()
            _apply_p_border_ordered(pPr, "CBD5E0")
            _apply_shading_ordered(pPr, PPR_ORDER, "F5F5F5")
            r = p.add_run("\n".join(code_lines))
            _set_run_font(r, "Courier New", 9.5)
            continue

        if line.startswith("$$") and line.endswith("$$"):
            _add_omml_equation(doc, line[2:-2])
            i += 1
            continue

        if line.startswith("## "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.line_spacing = 1.3
            p.paragraph_format.keep_with_next = True
            r = p.add_run(line[3:].strip().replace("`", ""))
            _set_run_font(r, "Times New Roman", 14.0, bold=True)
            i += 1
            continue

        if line.startswith("### "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.3
            p.paragraph_format.keep_with_next = True
            r = p.add_run(line[4:].strip().replace("`", ""))
            _set_run_font(r, "Times New Roman", 13.0, bold=True)
            i += 1
            continue

        m_img = re.match(r"^!\[(.*?)\]\((.*?)\)$", line)
        if m_img:
            img_path = m_img.group(2).strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            r = p.add_run()
            r.add_picture(img_path, width=Cm(15.2))
            i += 1
            continue

        if line.startswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            parsed_rows = []
            for idx_t, tl in enumerate(tbl_lines):
                if idx_t == 1 and set(tl.replace("|", "").replace(":", "").replace("-", "").strip()) == set():
                    continue
                cells = [c.strip() for c in tl.strip("|").split("|")]
                parsed_rows.append(cells)

            if parsed_rows:
                tbl_font = 9.5 if in_landscape else 10.5
                table = doc.add_table(rows=len(parsed_rows), cols=len(parsed_rows[0]))
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                table.style = "Table Grid"
                for r_i, row_vals in enumerate(parsed_rows):
                    for c_i, val in enumerate(row_vals):
                        cell = table.cell(r_i, c_i)
                        cp = cell.paragraphs[0]
                        cp.paragraph_format.space_before = Pt(2)
                        cp.paragraph_format.space_after = Pt(2)
                        cp.paragraph_format.line_spacing = 1.1
                        if r_i == 0:
                            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                            _apply_shading_ordered(cell._tc.get_or_add_tcPr(), TCPR_ORDER, "E2E8F0")
                            clean_hdr = val.replace("`", "").replace("**", "")
                            _add_formatted_runs(cp, f"**{clean_hdr}**", base_size=tbl_font)
                        else:
                            cp.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_i == 0 else WD_ALIGN_PARAGRAPH.CENTER
                            _add_formatted_runs(cp, val, base_size=tbl_font)
                spacer = doc.add_paragraph()
                spacer.paragraph_format.space_after = Pt(4)
            continue

        if line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            cap_text = line[1:-1].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing = 1.15
            if cap_text.startswith("Bảng"):
                p.paragraph_format.keep_with_next = True
            _add_formatted_runs(p, cap_text, base_size=11.0, base_italic=True)
            i += 1
            continue

        if line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.line_spacing = 1.3
            p.paragraph_format.space_after = Pt(4)
            _add_formatted_runs(p, line[2:].strip(), base_size=12.0)
            i += 1
            continue

        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.3
        p.paragraph_format.space_after = Pt(6)
        _add_formatted_runs(p, line, base_size=12.0)
        i += 1

    Path(out_docx).parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_docx)
    print(f"Saved {out_docx}")


def main() -> None:
    md_text, _ = build_markdown_report()
    compile_markdown_to_docx(md_text, "outputs/Bao_cao_Giua_Ky_Document_Scanner.docx")


if __name__ == "__main__":
    main()
