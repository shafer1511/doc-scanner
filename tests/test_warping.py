import numpy as np
from src.core.warping import bilinear_sample_gray, warp_document


def test_bilinear_sample_hand_calculated_patch():
    patch = np.array(
        [
            [10.0, 30.0, 50.0],
            [20.0, 60.0, 80.0],
            [40.0, 80.0, 120.0],
        ],
        dtype=np.float64,
    )
    # Integer coordinates
    assert bilinear_sample_gray(patch, 0.0, 0.0) == 10.0
    assert bilinear_sample_gray(patch, 1.0, 1.0) == 60.0
    # Fractional coordinate (x=0.5, y=0.5): (10+30+20+60)/4 = 30.0
    assert abs(bilinear_sample_gray(patch, 0.5, 0.5) - 30.0) < 1e-9
    # Non-symmetric fractional (x=0.25, y=0.75):
    # (0.75*0.25)*10 + (0.25*0.25)*30 + (0.75*0.75)*20 + (0.25*0.75)*60 = 1.875 + 1.875 + 11.25 + 11.25 = 26.25
    assert abs(bilinear_sample_gray(patch, 0.25, 0.75) - 26.25) < 1e-9
    # Border clamping
    assert bilinear_sample_gray(patch, -5.0, -3.0) == 10.0
    assert bilinear_sample_gray(patch, 10.0, 10.0) == 120.0


def test_identity_homography():
    img = np.arange(100 * 120, dtype=np.uint8).reshape(100, 120)
    corners = np.array([[0, 0], [119, 0], [119, 99], [0, 99]], dtype=np.float32)
    warped, H = warp_document(img, corners, target_size=(120, 100), interpolation="bilinear")
    assert np.allclose(H, np.eye(3), atol=1e-5)
    assert np.array_equal(warped, img)


def test_known_perspective_recovery():
    canvas = np.full((120, 160), 255, dtype=np.uint8)
    canvas[20:100, 30:130] = 40
    corners = np.array([[30, 20], [129, 20], [129, 99], [30, 99]], dtype=np.float32)
    warped, _ = warp_document(canvas, corners, target_size=(100, 80), interpolation="bilinear")
    assert warped.shape == (80, 100)
    assert float(np.mean(warped)) < 45.0
