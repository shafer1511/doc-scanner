import numpy as np
from src.core.hough import find_hough_peaks, hough_transform, polar_to_cartesian_segment


def test_all_zeros_edge_map():
    edges = np.zeros((80, 100), dtype=np.uint8)
    acc, rhos, thetas = hough_transform(edges)
    assert int(np.sum(acc)) == 0
    peaks = find_hough_peaks(acc, rhos, thetas, threshold=10)
    assert peaks == []


def test_vertical_and_horizontal_lines():
    edges = np.zeros((100, 120), dtype=np.uint8)
    edges[:, 40] = 255  # Vertical line x = 40 -> theta = 0 deg, rho = 40
    edges[60, :] = 255  # Horizontal line y = 60 -> theta = 90 deg, rho = 60

    acc, rhos, thetas = hough_transform(edges, rho_res=1.0, theta_res_deg=1.0)
    peaks = find_hough_peaks(acc, rhos, thetas, threshold=70, top_k=5, suppress=True)
    assert len(peaks) >= 2

    angles_deg = [np.rad2deg(p["theta"]) for p in peaks]
    has_vert = any(min(a, 180.0 - a) <= 2.0 and abs(abs(p["rho"]) - 40.0) <= 2.0 for p, a in zip(peaks, angles_deg))
    has_horiz = any(abs(a - 90.0) <= 2.0 and abs(p["rho"] - 60.0) <= 2.0 for p, a in zip(peaks, angles_deg))
    assert has_vert
    assert has_horiz

    p1, p2 = polar_to_cartesian_segment(40.0, 0.0, 120, 100)
    assert p1[0] == 40 and p2[0] == 40
    q1, q2 = polar_to_cartesian_segment(60.0, np.deg2rad(90.0), 120, 100)
    assert q1[1] == 60 and q2[1] == 60
