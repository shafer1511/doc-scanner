import numpy as np
from src.core.canny import (
    canny_edge_detection,
    double_threshold,
    hysteresis_8conn,
    non_maximum_suppression,
)


def test_constant_image_no_edges():
    img = np.full((64, 64), 128, dtype=np.uint8)
    res = canny_edge_detection(img)
    assert res["edges"].shape == (64, 64)
    assert res["edges"].dtype == np.uint8
    assert not np.isnan(res["magnitude"]).any()
    assert int(np.sum(res["edges"])) == 0


def test_step_edge_detected():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 32:] = 200
    res = canny_edge_detection(img, sigma=1.0, low_ratio=0.1, high_ratio=0.3)
    edges = res["edges"]
    assert int(np.sum(edges[:, 30:34] > 0)) > 40
    assert int(np.sum(edges[:, :25] > 0)) == 0
    assert int(np.sum(edges[:, 39:] > 0)) == 0


def test_hysteresis_8_connectivity():
    strong = np.zeros((5, 5), dtype=bool)
    weak = np.zeros((5, 5), dtype=bool)
    strong[1, 1] = True
    weak[2, 2] = True  # Diagonal 8-neighbor of strong
    weak[4, 4] = True  # Disconnected weak pixel
    out = hysteresis_8conn(strong, weak)
    assert out[1, 1] == 255
    assert out[2, 2] == 255
    assert out[4, 4] == 0


def test_nms_thins_thick_ridge():
    mag = np.zeros((5, 5), dtype=np.float64)
    mag[:, 1] = 10.0
    mag[:, 2] = 50.0
    mag[:, 3] = 10.0
    direction = np.zeros((5, 5), dtype=np.float64)  # 0 deg horizontal gradient -> vertical edge
    nms = non_maximum_suppression(mag, direction)
    assert nms[2, 2] == 50.0
    assert nms[2, 1] == 0.0
    assert nms[2, 3] == 0.0


def test_double_threshold_zero_max():
    nms = np.zeros((8, 8), dtype=np.float64)
    strong, weak, classified = double_threshold(nms)
    assert not strong.any()
    assert not weak.any()
    assert int(np.sum(classified)) == 0
