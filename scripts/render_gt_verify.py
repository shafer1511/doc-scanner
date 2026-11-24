import json
from pathlib import Path
import cv2
import numpy as np


def _draw_one(img: np.ndarray, corners: list[list[float]], label_id: str) -> np.ndarray:
    h, w = img.shape[:2]
    scale = 420.0 / float(max(h, w))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    vis = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    pts = np.rint(np.array(corners, dtype=np.float64) * scale).astype(np.int32)
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (255, 0, 255), 3, cv2.LINE_AA)
    for idx, (cx, cy) in enumerate(pts):
        cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 255), -1)
        cv2.circle(vis, (int(cx), int(cy)), 6, (0, 0, 0), 1)
        cv2.putText(
            vis,
            str(idx),
            (max(4, int(cx) - 4), max(14, int(cy) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    tile = np.full((460, 440, 3), 245, dtype=np.uint8)
    y_off = 32 + (420 - nh) // 2
    x_off = (440 - nw) // 2
    tile[y_off : y_off + nh, x_off : x_off + nw] = vis
    cv2.putText(tile, label_id, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
    return tile


def _make_sheet(ids: list[str], labels: dict, sizes: dict, out_path: Path, cols: int = 5) -> None:
    tiles = []
    for iid in ids:
        info = sizes[iid]
        im = cv2.imread(info["path"])
        corners = labels[iid]["corners"]
        tiles.append(_draw_one(im, corners, f"{iid} ({info['condition'][:10]})"))

    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * 460, cols * 440, 3), 255, dtype=np.uint8)
    for idx, t in enumerate(tiles):
        r, c = divmod(idx, cols)
        sheet[r * 460 : (r + 1) * 460, c * 440 : (c + 1) * 440] = t
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved verification sheet {out_path}")


def main() -> None:
    labels = json.load(open("data/labels.json", encoding="utf-8"))
    sizes = json.load(open("scratch/gt_grid/sizes.json", encoding="utf-8"))
    dev_ids = [f"dev_{i:02d}" for i in range(1, 31)]
    test_ids = [f"test_{i:02d}" for i in range(1, 21)]

    _make_sheet(dev_ids[:15], labels, sizes, Path("scratch/gt_verify_dev_1.jpg"), cols=5)
    _make_sheet(dev_ids[15:], labels, sizes, Path("scratch/gt_verify_dev_2.jpg"), cols=5)
    _make_sheet(test_ids[:10], labels, sizes, Path("scratch/gt_verify_test_1.jpg"), cols=5)
    _make_sheet(test_ids[10:], labels, sizes, Path("scratch/gt_verify_test_2.jpg"), cols=5)


if __name__ == "__main__":
    main()
