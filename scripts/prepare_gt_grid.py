import csv
import json
import os
from pathlib import Path
import cv2
import numpy as np


def render_gridded_image(img_bgr: np.ndarray, img_id: str, doc_name: str, cond: str) -> tuple[np.ndarray, int, int]:
    h_orig, w_orig = img_bgr.shape[:2]
    scale = 640.0 / float(max(h_orig, w_orig))
    new_w = int(round(w_orig * scale))
    new_h = int(round(h_orig * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    overlay = resized.copy()
    for val in range(50, 1000, 50):
        px = int(round(val / 1000.0 * (new_w - 1)))
        py = int(round(val / 1000.0 * (new_h - 1)))
        if val % 100 == 0:
            cv2.line(overlay, (px, 0), (px, new_h - 1), (255, 255, 0), 1)
            cv2.line(overlay, (0, py), (new_w - 1, py), (255, 255, 0), 1)
        else:
            cv2.line(overlay, (px, 0), (px, new_h - 1), (0, 220, 220), 1)
            cv2.line(overlay, (0, py), (new_w - 1, py), (0, 220, 220), 1)
    blended = cv2.addWeighted(overlay, 0.55, resized, 0.45, 0)

    pad = 48
    canvas = np.full((new_h + 2 * pad, new_w + 2 * pad, 3), 255, dtype=np.uint8)
    canvas[pad : pad + new_h, pad : pad + new_w] = blended
    cv2.rectangle(canvas, (pad, pad), (pad + new_w - 1, pad + new_h - 1), (40, 40, 40), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for val in range(0, 1001, 100):
        px = pad + int(round(val / 1000.0 * (new_w - 1)))
        py = pad + int(round(val / 1000.0 * (new_h - 1)))
        lbl = str(val)
        cv2.putText(canvas, lbl, (px - 12, pad - 8), font, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (px - 12, pad + new_h + 18), font, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (4, py + 4), font, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (pad + new_w + 6, py + 4), font, 0.38, (20, 20, 20), 1, cv2.LINE_AA)

    title = f"{img_id} | {doc_name} | {cond} | {w_orig}x{h_orig}"
    cv2.putText(canvas, title, (pad, 16), font, 0.48, (0, 0, 160), 1, cv2.LINE_AA)
    return canvas, w_orig, h_orig


def main() -> None:
    out_dir = Path("scratch/gt_grid")
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}

    sources = [
        ("data/manifest_development.csv", "data/development", "development"),
        ("data/manifest_test.csv", "data/test", "test"),
    ]
    for man_path, img_dir, split in sources:
        with open(man_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["is_positive"] != "1":
                    continue
                iid = r["image_id"]
                fpath = os.path.join(img_dir, r["filename"])
                im = cv2.imread(fpath)
                if im is None:
                    raise FileNotFoundError(fpath)
                grid_img, w_orig, h_orig = render_gridded_image(
                    im, iid, r["document_name"], r["condition"]
                )
                cv2.imwrite(str(out_dir / f"{iid}.jpg"), grid_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                sizes[iid] = {
                    "width": w_orig,
                    "height": h_orig,
                    "split": split,
                    "filename": r["filename"],
                    "path": fpath,
                    "document_name": r["document_name"],
                    "condition": r["condition"],
                }

    with open(out_dir / "sizes.json", "w", encoding="utf-8") as f:
        json.dump(sizes, f, indent=2)
    print(f"Prepared {len(sizes)} gridded images and sizes.json in {out_dir}")


if __name__ == "__main__":
    main()
