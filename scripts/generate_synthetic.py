import csv
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/timesbd.ttf"]
        if bold
        else ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/times.ttf"]
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _render_layout(layout_id: int, width: int = 480, height: int = 360) -> np.ndarray:
    """Render clean binary mask (0=ink, 255=background) for one of 4 document layouts."""
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)
    f_title = _load_font(20, bold=True)
    f_body = _load_font(14, bold=False)
    f_small = _load_font(11, bold=False)

    if layout_id == 1:
        draw.text((24, 20), "BAO CAO XU LY ANH SO - TAI LIEU 01", fill=0, font=f_title)
        draw.line([(24, 48), (456, 48)], fill=0, width=2)
        lines = [
            "1. Muc tieu: Tach nen van ban va phan doan net chu ro rang.",
            "2. Phuong phap Canny su dung Gaussian blur giam nhieu nhieu hat.",
            "3. Bien doi Hough anh xa diem bien sang khong gian (rho, theta).",
            "4. Phep bien doi phoi canh Homography dua trang giay ve mat phang.",
            "5. Nguong thich nghi Sauvola giup bao toan net chu vung bong do.",
            "6. Ket qua do luong bang Precision, Recall va chi so F1 lop ink.",
            "7. Moi tham so duoc khoa tren tap Development truoc khi danh gia.",
        ]
        y = 66
        for line in lines:
            draw.text((24, y), line, fill=0, font=f_body)
            y += 34
    elif layout_id == 2:
        draw.text((24, 18), "BANG TONG HOP SO LIEU THUC NGHIEM", fill=0, font=f_title)
        draw.rectangle([24, 54, 456, 326], outline=0, width=2)
        draw.line([(230, 54), (230, 326)], fill=0, width=2)
        rows = [
            ("Chi tieu danh gia", "Gia tri do duoc"),
            ("Polygon IoU >= 0.8", "83.33 % (25/30)"),
            ("Corner Error Norm", "0.0184 (px/diag)"),
            ("Otsu F1 (Nen deu)", "0.9712"),
            ("Sauvola F1 (Bong do)", "0.9145"),
            ("Thoi gian CPU TB", "142.50 ms"),
            ("Trang thai kiem dinh", "Dat chuan AST"),
        ]
        y = 64
        for idx, (c1, c2) in enumerate(rows):
            font = _load_font(14, bold=(idx == 0))
            draw.text((34, y), c1, fill=0, font=font)
            draw.text((242, y), c2, fill=0, font=font)
            if idx < len(rows) - 1:
                draw.line([(24, y + 28), (456, y + 28)], fill=0, width=1)
            y += 38
    elif layout_id == 3:
        draw.text((24, 18), "QUY TRINH KIEM CHUNG HE THONG", fill=0, font=f_title)
        bullets = [
            "- Kiem tra AST loai bo ham cam trong thu muc src/core.",
            "- Kiem tra ca bien: anh hang, step edge, duong thang 0/90 do.",
            "- Chay dong loat tren 36 anh Development va 24 anh Test.",
            "- Ghi nhan ca cac truong hop tu choi (rejected) vao tep CSV.",
            "- Xuat anh trung gian: edge map, accumulator, tu giac ung vien.",
        ]
        y = 60
        for b in bullets:
            draw.text((28, y), b, fill=0, font=f_body)
            y += 36
        draw.line([(24, 255), (456, 255)], fill=0, width=1)
        draw.text(
            (24, 270),
            "Ghi chu: Moi phep thu nghiem su dung seed=42 de dam bao tinh tai lap.",
            fill=0,
            font=f_small,
        )
        draw.text(
            (24, 294),
            "Tai lieu tham khao: Gonzalez & Woods, Digital Image Processing, 4th ed.",
            fill=0,
            font=f_small,
        )
    else:
        draw.text((24, 18), "PHIEU GHI NHAN THONG TIN MAU", fill=0, font=f_title)
        fields = [
            ("Ma ho so:", "DOC-2026-004"),
            ("Ngay kiem tra:", "24/11/2026"),
            ("Do phan giai:", "2560 x 1922 px"),
            ("Kich thuoc cua so:", "w = 25, k = 0.20"),
            ("He so R:", "128.0 (8-bit gray)"),
        ]
        y = 62
        for label, val in fields:
            draw.text((28, y), label, fill=0, font=_load_font(14, bold=True))
            draw.rectangle([185, y - 4, 452, y + 24], outline=0, width=1)
            draw.text((195, y), val, fill=0, font=f_body)
            y += 46
        draw.text((28, 305), "Xac nhan ket qua: HOP LE [X]   TU CHOI [ ]", fill=0, font=f_body)

    arr = np.array(canvas, dtype=np.uint8)
    return np.where(arr < 128, np.uint8(0), np.uint8(255))


def _apply_lighting(mask: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    """Apply uniform, gradient, or shadow+noise illumination to binary ink mask."""
    h, w = mask.shape
    ink_mask = mask < 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    if condition == "nen_deu":
        bg = np.full((h, w), 232.0, dtype=np.float64)
        ink_val = 32.0
        noise_std = 2.0
    elif condition == "gradient_sang":
        # Diagonal illumination drop from 245 (top-left) down to 105 (bottom-right)
        norm_diag = (0.65 * xx / max(1, w - 1)) + (0.35 * yy / max(1, h - 1))
        bg = 245.0 - 140.0 * norm_diag
        ink_val = 38.0 - 15.0 * norm_diag
        noise_std = 3.0
    elif condition == "bong_kem_nhieu":
        bg = np.full((h, w), 228.0, dtype=np.float64)
        # Smooth cast shadow covering a curved region of the page
        dist = ((xx - 0.58 * w) / (0.42 * w)) ** 2 + ((yy - 0.55 * h) / (0.48 * h)) ** 2
        shadow = 1.0 / (1.0 + np.exp(-6.0 * (1.0 - dist)))
        bg = bg - 115.0 * shadow
        ink_val = 36.0 - 18.0 * shadow
        noise_std = 10.0
    else:
        raise ValueError(f"Unknown condition: {condition}")

    img = np.where(ink_mask, ink_val, bg)
    img = img + rng.normal(0.0, noise_std, size=(h, w))
    img_u8 = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def generate_all(out_dir: str = "data/synthetic", seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    layouts = [
        (1, "development", "layout_1"),
        (2, "development", "layout_2"),
        (3, "test", "layout_3"),
        (4, "test", "layout_4"),
    ]
    conditions = ["nen_deu", "gradient_sang", "bong_kem_nhieu"]

    rows: list[dict] = []
    counters = {"development": 0, "test": 0}
    for layout_id, split, layout_name in layouts:
        gt_mask = _render_layout(layout_id)
        for cond in conditions:
            counters[split] += 1
            idx = counters[split]
            prefix = "syn_dev" if split == "development" else "syn_test"
            img_id = f"{prefix}_{idx:02d}"
            img_name = f"{img_id}.jpg"
            mask_name = f"{img_id}_mask.png"

            syn_bgr = _apply_lighting(gt_mask, cond, rng)
            cv2.imwrite(str(out_path / img_name), syn_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out_path / mask_name), gt_mask)

            rows.append(
                {
                    "image_id": img_id,
                    "split": split,
                    "layout": layout_name,
                    "condition": cond,
                    "filename": img_name,
                    "mask_filename": mask_name,
                }
            )

    manifest_path = out_path / "manifest_synthetic.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "split", "layout", "condition", "filename", "mask_filename"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} synthetic images + masks in {out_dir}")
    return rows


if __name__ == "__main__":
    generate_all()
