"""Generate 12 synthetic document images with ground-truth corners and ink masks.
6 images for Development split, 6 images for Test split.
"""
import csv
import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SYNTHETIC_DIR = "data/synthetic"
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

CONDITIONS = [
    ("nen_don_gian_du_sang", "Flat lighting, clean desk background"),
    ("goc_chup_xien", "Steep 3D perspective viewpoint"),
    ("nen_chu_nhat_gay_nhieu", "Rectangular books/mousepad around document"),
    ("bien_tuong_phan_thap", "Document boundary blends into table color"),
    ("anh_sang_khong_deu_bong_do", "Harsh diagonal shadow cast across text"),
    ("bien_dang_phuc_tap_nhiễu", "Blur and salt-and-pepper noise degradation"),
]

SAMPLE_TEXTS = [
    [
        "TRUONG DAI HOC KHOA HOC TU NHIEN",
        "KHOA DIEN TU - VIEN THONG",
        "NHAP MON XU LY ANH VA VIDEO",
        "DO AN GIUA KY: DOCUMENT SCANNER",
        "",
        "Muc tieu: Xay dung chuong trinh tu dong bien anh chup",
        "tai lieu thanh anh trang giay nhin thang va nhi phan hoa.",
        "Cac module bat buoc gom co: Canny edge detector,",
        "Hough transform line detector, quadrilateral selection,",
        "perspective transformation va binarization Otsu Sauvola.",
        "Toan bo pipeline phai duoc cai dat truc tiep bang NumPy.",
        "Kiem tra danh gia tren tap Development va Test.",
    ],
    [
        "ALGORITHM SPECIFICATION & COMPLEXITY",
        "Module 1: Canny Edge Detection",
        "Gradient computation using 3x3 Sobel kernels.",
        "Non-Maximum Suppression (NMS) thins edges to single-pixel ridges.",
        "Double thresholding classifies pixels into strong, weak, non-edge.",
        "8-connected hysteresis tracks weak edges connected to strong ones.",
        "",
        "Module 2: Standard Hough Line Transform",
        "Accumulator parameterization: rho = x*cos(theta) + y*sin(theta).",
        "Local maximum peak search with vote thresholding.",
        "Suppression of duplicate nearby lines in rho and theta space.",
    ],
    [
        "BINARIZATION AND EVALUATION METRICS",
        "Otsu threshold maximizes inter-class variance between foreground",
        "and background pixels from the 256-level intensity histogram.",
        "",
        "Sauvola adaptive thresholding handles complex illumination gradients:",
        "T(x,y) = m(x,y) * [ 1 + k * ( s(x,y) / R - 1 ) ]",
        "where m is the local mean and s is the local standard deviation.",
        "Ink layer precision, recall, and F1 scores are computed",
        "against pixel-level ground truth binary masks.",
    ],
]


def render_document_page(text_lines, width=400, height=560, bg_val=245):
    """Render a text document page and return (image, ink_mask)."""
    img = np.full((height, width), bg_val, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Use OpenCV putText for reproducible cross-platform font rendering
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_start = 45
    line_height = 28
    
    for i, line in enumerate(text_lines):
        y = y_start + i * line_height
        if y >= height - 20:
            break
        if not line:
            continue
        scale = 0.52 if i < 4 else 0.45
        thick = 2 if i < 4 else 1
        cv2.putText(img, line, (25, y), font, scale, 25, thick, cv2.LINE_AA)
        cv2.putText(mask, line, (25, y), font, scale, 255, thick, cv2.LINE_AA)
        
    return img, mask


def create_synthetic_scene(doc_img, doc_mask, condition_name, index, canvas_w=800, canvas_h=600):
    """Warp document onto canvas with condition-specific background and lighting."""
    H_d, W_d = doc_img.shape[:2]
    src_pts = np.array([[0, 0], [W_d - 1, 0], [W_d - 1, H_d - 1], [0, H_d - 1]], dtype=np.float32)
    
    # Background texture
    np.random.seed(1000 + index)
    if condition_name == "bien_tuong_phan_thap":
        # Table color very close to paper color (220 vs 240)
        canvas = np.full((canvas_h, canvas_w), 215, dtype=np.uint8)
        noise = np.random.normal(0, 4, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif condition_name == "nen_chu_nhat_gay_nhieu":
        # Wood desk + rectangular distracting books/mousepad
        canvas = np.full((canvas_h, canvas_w), 110, dtype=np.uint8)
        # Draw rectangular distractors
        cv2.rectangle(canvas, (40, 50), (280, 240), 60, -1)   # Dark book
        cv2.rectangle(canvas, (550, 80), (760, 480), 160, -1) # Light notebook
        cv2.rectangle(canvas, (100, 420), (340, 560), 85, -1) # Mousepad
    else:
        # Standard wooden desk tone (~120 with wood-grain pattern)
        canvas = np.full((canvas_h, canvas_w), 125, dtype=np.uint8)
        # Add subtle wood grain
        grain = np.linspace(-15, 15, canvas_w, dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + grain[None, :], 0, 255).astype(np.uint8)
    
    # Define destination quad corners [TL, TR, BR, BL]
    if condition_name == "goc_chup_xien":
        # Steep 3D perspective
        dst_pts = np.array([
            [260.0, 90.0],   # TL
            [610.0, 130.0],  # TR
            [550.0, 540.0],  # BR
            [130.0, 490.0],  # BL
        ], dtype=np.float32)
    elif condition_name == "nen_chu_nhat_gay_nhieu":
        # Centered among distractors
        dst_pts = np.array([
            [210.0, 110.0],
            [570.0, 95.0],
            [590.0, 520.0],
            [225.0, 535.0],
        ], dtype=np.float32)
    else:
        # Mild realistic perspective
        dst_pts = np.array([
            [180.0 + index * 4, 80.0 + index * 3],
            [620.0 - index * 3, 100.0 + index * 5],
            [590.0 - index * 5, 530.0 - index * 2],
            [160.0 + index * 2, 510.0 - index * 4],
        ], dtype=np.float32)
        
    H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Warp document and mask onto scene
    warped_doc = cv2.warpPerspective(doc_img, H_mat, (canvas_w, canvas_h), borderValue=0)
    doc_roi_mask = cv2.warpPerspective(np.full((H_d, W_d), 255, dtype=np.uint8), H_mat, (canvas_w, canvas_h), borderValue=0)
    
    # Composite onto canvas
    canvas_out = np.where(doc_roi_mask > 128, warped_doc, canvas)
    
    # Apply conditions
    if condition_name == "anh_sang_khong_deu_bong_do":
        # Diagonal shadow ramp: drops light by up to 130 levels across the middle
        y_idx, x_idx = np.indices((canvas_h, canvas_w))
        diag_dist = (x_idx + y_idx * 1.2) / (canvas_w + canvas_h * 1.2)
        shadow = np.clip(1.0 - 0.65 * np.exp(-((diag_dist - 0.45) ** 2) / 0.04), 0.35, 1.0)
        canvas_out = np.clip(canvas_out.astype(np.float64) * shadow, 0, 255).astype(np.uint8)
    elif condition_name == "bien_dang_phuc_tap_nhiễu":
        # Blur + Gaussian noise
        canvas_out = cv2.GaussianBlur(canvas_out, (5, 5), 1.2)
        noise = np.random.normal(0, 12, canvas_out.shape).astype(np.int16)
        canvas_out = np.clip(canvas_out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    return canvas_out, doc_mask, dst_pts


def generate_all():
    records = []
    
    for i, (cond_name, cond_desc) in enumerate(CONDITIONS):
        text_lines = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        
        # 1. Dev synthetic
        dev_id = f"syn_dev_{i+1:02d}"
        doc_img, doc_mask = render_document_page(text_lines, width=420, height=580)
        scene_dev, mask_dev, corners_dev = create_synthetic_scene(doc_img, doc_mask, cond_name, i)
        
        dev_img_name = f"{dev_id}.jpg"
        dev_mask_name = f"{dev_id}_mask.png"
        cv2.imwrite(os.path.join(SYNTHETIC_DIR, dev_img_name), scene_dev)
        cv2.imwrite(os.path.join(SYNTHETIC_DIR, dev_mask_name), mask_dev)
        
        records.append({
            "image_id": dev_id,
            "split": "development",
            "document_name": f"syn_doc_{i+1:02d}",
            "condition": cond_name,
            "is_positive": 1,
            "filename": dev_img_name,
            "mask_filename": dev_mask_name,
            "x0": f"{corners_dev[0, 0]:.1f}", "y0": f"{corners_dev[0, 1]:.1f}",
            "x1": f"{corners_dev[1, 0]:.1f}", "y1": f"{corners_dev[1, 1]:.1f}",
            "x2": f"{corners_dev[2, 0]:.1f}", "y2": f"{corners_dev[2, 1]:.1f}",
            "x3": f"{corners_dev[3, 0]:.1f}", "y3": f"{corners_dev[3, 1]:.1f}",
        })
        
        # 2. Test synthetic
        test_id = f"syn_test_{i+1:02d}"
        doc_img_t, doc_mask_t = render_document_page(text_lines, width=420, height=580)
        scene_test, mask_test, corners_test = create_synthetic_scene(doc_img_t, doc_mask_t, cond_name, i + 10)
        
        test_img_name = f"{test_id}.jpg"
        test_mask_name = f"{test_id}_mask.png"
        cv2.imwrite(os.path.join(SYNTHETIC_DIR, test_img_name), scene_test)
        cv2.imwrite(os.path.join(SYNTHETIC_DIR, test_mask_name), mask_test)
        
        records.append({
            "image_id": test_id,
            "split": "test",
            "document_name": f"syn_doc_{i+1:02d}",
            "condition": cond_name,
            "is_positive": 1,
            "filename": test_img_name,
            "mask_filename": test_mask_name,
            "x0": f"{corners_test[0, 0]:.1f}", "y0": f"{corners_test[0, 1]:.1f}",
            "x1": f"{corners_test[1, 0]:.1f}", "y1": f"{corners_test[1, 1]:.1f}",
            "x2": f"{corners_test[2, 0]:.1f}", "y2": f"{corners_test[2, 1]:.1f}",
            "x3": f"{corners_test[3, 0]:.1f}", "y3": f"{corners_test[3, 1]:.1f}",
        })
        
    csv_path = os.path.join(SYNTHETIC_DIR, "manifest_synthetic.csv")
    fieldnames = ["image_id", "split", "document_name", "condition", "is_positive", "filename", "mask_filename", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Generated 12 synthetic documents in {SYNTHETIC_DIR}:")
    print(f"  6 development, 6 test.")
    print(f"  Manifest written to {csv_path}")


if __name__ == "__main__":
    generate_all()
