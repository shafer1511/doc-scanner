import numpy as np
from run_pipeline import ink_prf1
from src.core.thresholding import otsu_threshold, sauvola_threshold


def test_constant_image_otsu_and_sauvola():
    img = np.full((48, 48), 200, dtype=np.uint8)
    o = otsu_threshold(img)
    s = sauvola_threshold(img, window_size=15, k=0.2)
    assert o["binary"].shape == (48, 48)
    assert s["binary"].shape == (48, 48)
    assert not np.isnan(s["threshold_map"]).any()
    assert int(np.min(o["binary"])) == 255
    assert int(np.min(s["binary"])) == 255


def test_otsu_and_sauvola_on_synthetic_mask():
    img = np.full((64, 64), 220, dtype=np.uint8)
    gt_mask = np.full((64, 64), 255, dtype=np.uint8)
    img[16:48, 20:44] = 30
    gt_mask[16:48, 20:44] = 0

    o = otsu_threshold(img)
    s = sauvola_threshold(img, window_size=25, k=0.2)

    _, _, f1_o = ink_prf1(o["binary"], gt_mask)
    _, _, f1_s = ink_prf1(s["binary"], gt_mask)
    assert f1_o == 1.0
    assert f1_s > 0.85
