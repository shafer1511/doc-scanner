<!-- COVER_START -->
TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN - ĐHQG TP.HCM
KHOA ĐIỆN TỬ - VIỄN THÔNG
BÁO CÁO ĐỒ ÁN GIỮA KỲ: DOCUMENT SCANNER
Phát hiện tài liệu, hiệu chỉnh phối cảnh và nhị phân hóa ảnh
Học phần: Nhập môn Xử lý ảnh và Video
Môi trường thực thi: Python 3.11.9 | OpenCV 5.0.0 | NumPy 2.3.2 | Seed 42
Ngày hoàn thành báo cáo: 24/11/2026
<!-- COVER_END -->

## CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI VÀ MỤC TIÊU

### 1.1. Mục tiêu và phạm vi bài toán

Đề tài xây dựng hệ thống Document Scanner tự động chuyển ảnh chụp tài liệu thành trang giấy nhìn thẳng. Đầu ra gồm ảnh màu đã hiệu chỉnh phối cảnh và ảnh nhị phân rõ nét.
Chương trình giải quyết ba bài toán cốt lõi gồm định vị tứ giác, nắn thẳng phối cảnh và nhị phân hóa. Mọi thuật toán xử lý lõi chạy hoàn toàn trên CPU với seed cố định `42`.
Hệ thống thử nghiệm trên năm điều kiện chụp thực tế và một nhóm ảnh âm đối chứng. Các điều kiện gồm nền đơn giản, góc xiên, nền chữ nhật nhiễu, tương phản thấp và bóng đổ.

### 1.2. Ràng buộc kỹ thuật và môi trường thực nghiệm

Pipeline chính trong `src/core/` tự cài đặt tường minh bằng Python, NumPy và các primitive cơ bản. Mã nguồn tuyệt đối không gọi `cv2.Canny`, `cv2.HoughLines`, `cv2.threshold` hay `skimage.filters.threshold_*`.
Thực nghiệm chạy trên bộ xử lý `AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD` với Python `3.11.9`. Các thư viện cài đặt gồm OpenCV `5.0.0`, NumPy `2.3.2`, SciPy `1.16.1` và scikit-image `0.26.0`.

## CHƯƠNG 2. KIẾN TRÚC HỆ THỐNG VÀ PHÂN TÍCH TỪNG MODULE

### 2.1. Kiến trúc tổng thể Pipeline và cơ chế Early Exit

Hệ thống tổ chức theo kiến trúc ống dẫn năm giai đoạn tuần tự từ ảnh BGR thô tới ảnh nhị phân. Giai đoạn phát hiện biên và Hough xử lý ở độ phân giải chuẩn hóa `640 px` để tối ưu tốc độ CPU.
Tọa độ bốn góc tìm được sau đó quy đổi về độ phân giải gốc để biến đổi Homography giữ nguyên độ nét chữ. Nếu không tìm thấy tứ giác hợp lệ tại bước ba, nhánh Early Exit trả về `rejected` và bỏ qua warping.

![Sơ đồ kiến trúc tổng thể hệ thống](outputs/figures/fig_pipeline_architecture.png)
*Hình 1: Kiến trúc tổng thể 5 giai đoạn của hệ thống Document Scanner và nhánh Early Exit khi từ chối ảnh âm.*

*Bảng 1: Vai trò, dữ liệu vào/ra và chức năng từng module trong mã nguồn dự án.*

| Module | Mục đích chính | Đầu vào -> Đầu ra (shape/dtype) | Hàm cốt lõi |
| :--- | :--- | :--- | :--- |
| `src/core/canny.py` | Tách biên ảnh bằng Canny tự cài | `(H,W,3)` uint8 -> `edges (H,W)` uint8 | `gaussian_smooth`, `non_maximum_suppression`, `hysteresis_8conn` |
| `src/core/hough.py` | Dựng accumulator và trích xuất đường thẳng | `edges` -> `acc (R,T)` int32, `peaks` list | `hough_transform`, `find_hough_peaks`, `polar_to_cartesian_segment` |
| `src/core/quadrilateral.py` | Ghép 4 đường, chấm điểm và lọc hợp lệ | `peaks` -> `corners (4,2)` float32 hoặc `rejected` | `intersect_lines`, `is_convex_quad`, `find_document_quad` |
| `src/core/warping.py` | Tính Homography và nội suy song tuyến tính | `image`, `corners` -> `warped` BGR, `H (3,3)` | `bilinear_sample_gray`, `compute_output_size`, `warp_document` |
| `src/core/thresholding.py` | Phân ngưỡng toàn cục Otsu và cục bộ Sauvola | `warped_gray` -> `binary (H,W)` uint8 | `otsu_threshold`, `sauvola_threshold` |
| `src/baseline/baseline.py` | Nhánh đối chứng dùng thư viện OpenCV/skimage | `image` -> `status`, `otsu/sauvola` binary | `run_baseline_on_image`, `run_baseline_batch` |
| `run_pipeline.py` | Điều phối batch, đo CPU runtime và tính metrics | `manifest.csv` -> `results_*.csv` (12 cột) | `process_single_image`, `polygon_iou`, `ink_prf1` |
| `tests/verify_ast.py` | Quét AST cấm gọi hàm thư viện gian lận | `src/core/*.py` -> Exit 0 nếu 0 vi phạm | `scan_file`, `main` |

### 2.2. Module phát hiện biên Canny (`src/core/canny.py`)

Module `src/core/canny.py` tách toàn bộ công đoạn phát hiện biên thành các hàm thuần để xuất từng ảnh trung gian. Ảnh màu BGR chuyển sang mức xám `float64` rồi làm mịn bằng mặt nạ Gaussian theo tham số $\sigma$.
Toán tử Sobel 3x3 tính đạo hàm theo hai phương để thu được biên độ $M(x, y)$ và hướng gradient $\theta_g(x, y)$.

$$M(x, y) = \sqrt{G_x(x, y)^2 + G_y(x, y)^2}, \quad \theta_g(x, y) = \arctan2(G_y, G_x)$$

Hàm `non_maximum_suppression` lượng tử hóa góc về bốn hướng $0^\circ, 45^\circ, 90^\circ, 135^\circ$ bằng phép cắt mảng NumPy. Cách vector hóa này làm mảnh cạnh xuống đúng 1 pixel mà không cần vòng lặp Python.
Cuối cùng, `double_threshold` chia điểm biên thành hai mức `strong` và `weak`, còn `hysteresis_8conn` duyệt BFS theo 8 lân cận.

```python
def non_maximum_suppression(
    magnitude: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Suppress non-maximum gradient magnitudes along 4 quantized directions (0, 45, 90, 135 deg).

    Args:
        magnitude: Gradient magnitude array of shape (H, W), dtype float64.
        direction: Gradient angle array in radians of shape (H, W), dtype float64.
    Returns:
        Thinned magnitude array of shape (H, W), dtype float64.
    """
    h, w = magnitude.shape
    if h < 3 or w < 3:
        return np.zeros_like(magnitude, dtype=np.float64)

    angle = (np.rad2deg(direction) % 180.0)
    suppressed = np.zeros_like(magnitude, dtype=np.float64)

    center = magnitude[1:-1, 1:-1]
    ang = angle[1:-1, 1:-1]

    m0 = ((ang < 22.5) | (ang >= 157.5))
    m45 = (ang >= 22.5) & (ang < 67.5)
    m90 = (ang >= 67.5) & (ang < 112.5)
    m135 = (ang >= 112.5) & (ang < 157.5)

    keep0 = m0 & (center >= magnitude[1:-1, 2:]) & (center >= magnitude[1:-1, :-2])
    keep45 = m45 & (center >= magnitude[2:, 2:]) & (center >= magnitude[:-2, :-2])
```

### 2.3. Module biến đổi Hough (`src/core/hough.py`)

Module `src/core/hough.py` chuyển tập điểm biên rời rạc thành các đường thẳng tham số trong không gian cực.

$$\rho = x \cos\theta + y \sin\theta, \quad \theta \in [0, \pi)$$

Hàm `hough_transform` tính đồng loạt giá trị $\rho$ cho mọi điểm biên và cộng dồn phiếu bằng `np.bincount`. Nhờ vector hóa theo góc $\theta$, quá trình dựng accumulator trên ảnh `640 px` chỉ mất khoảng `88 ms`.
Hàm `find_hough_peaks` sắp xếp đỉnh giảm dần và triệt tiêu đường gần trùng theo $\Delta\rho$ và $\Delta\theta$ tuần hoàn. Cơ chế tuần hoàn giúp hợp nhất hai đường thẳng đứng gần biên $0^\circ$ và $180^\circ$ thành một cực đại duy nhất.

```python
def hough_transform(
    edge_map: np.ndarray, rho_res: float = 1.0, theta_res_deg: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute standard Hough accumulator for rho = x*cos(theta) + y*sin(theta), theta in [0, pi).

    Args:
        edge_map: Binary edge map of shape (H, W) where nonzero pixels vote.
        rho_res: Quantization step for rho in pixels.
        theta_res_deg: Quantization step for theta in degrees.
    Returns:
        Tuple (accumulator, rhos, thetas) with shapes (R, T), (R,), and (T,) in radians.
    """
    h, w = edge_map.shape[:2]
    max_rho = float(np.ceil(np.hypot(w, h)))
    num_rhos = int(np.floor((2.0 * max_rho) / rho_res)) + 1
    rhos = np.linspace(-max_rho, max_rho, num_rhos, dtype=np.float64)

    num_thetas = max(1, int(round(180.0 / theta_res_deg)))
    thetas = np.deg2rad(np.arange(num_thetas, dtype=np.float64) * (180.0 / num_thetas))

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    ys, xs = np.nonzero(edge_map)
    if ys.size == 0:
        return accumulator, rhos, thetas

    cos_t = np.cos(thetas)
```

### 2.4. Module chọn tứ giác tài liệu (`src/core/quadrilateral.py`)

Module `src/core/quadrilateral.py` ghép các cặp đường gần song song trực giao nhau để giải bốn giao điểm qua `np.linalg.solve`. Tứ giác ứng viên phải vượt qua các bộ lọc giới hạn khung ảnh, tính lồi, độ dài cạnh tối thiểu và góc đỉnh.
Hàm `score_quadrilateral` đánh giá ứng viên dựa trên tỷ lệ hỗ trợ biên dọc chu vi $S_{edge}$ và diện tích chuẩn hóa.

$$Score = w_s \cdot S_{edge} + w_a \cdot A_{norm} = 0.65 \cdot S_{edge} + 0.35 \cdot \frac{A_{quad}}{W \cdot H}$$

Ngoài trung bình $S_{edge}$, bộ lọc yêu cầu từng cạnh riêng lẻ đạt `min_side_support` để loại tứ giác thiếu cạnh. Nếu không có ứng viên đạt chuẩn, hệ thống trả về `rejected` kèm lý do cụ thể như `short_side` hay `out_of_bounds`.

```python
def score_quadrilateral(
    pts: np.ndarray,
    edge_map: np.ndarray,
    img_shape: tuple[int, ...],
    use_edge_support: bool = True,
    area_weight: float = 0.35,
    support_weight: float = 0.65,
    dilated_edges: np.ndarray | None = None,
) -> dict[str, float | list[float]]:
    """Score a valid quadrilateral using edge support and area ratio.

    Args:
        pts: Ordered corners of shape (4, 2).
        edge_map: Binary edge map of shape (H, W).
        img_shape: Image shape tuple (H, W, ...).
        use_edge_support: Include perimeter edge support if True, else score by area only.
        area_weight: Weight of normalized area ratio.
        support_weight: Weight of perimeter edge support.
        dilated_edges: Optional pre-dilated binary edge map of shape (H, W) for speed.
    Returns:
        Dict with keys score, edge_support, min_side_support, area_ratio.
    """
    h, w = img_shape[:2]
    area_ratio = polygon_area(pts) / float(max(1, h * w))
    if dilated_edges is not None:
        side_sups = compute_side_edge_supports(pts, dilated_edges, band=0)
    else:
        side_sups = compute_side_edge_supports(pts, edge_map, band=1)
```

### 2.5. Module hiệu chỉnh phối cảnh và nội suy Bilinear (`src/core/warping.py`)

Module `src/core/warping.py` ước lượng kích thước trang giấy đích từ độ dài lớn nhất hai cặp cạnh đối diện. Ma trận Homography $3 \times 3$ biến đổi cả ảnh màu lẫn ảnh xám gốc về khung hình chữ nhật nhìn thẳng.
Hàm `bilinear_sample_gray` cài đặt tường minh công thức nội suy song tuyến tính từ bốn điểm lân cận để kiểm chứng độ chính xác.

$$f(x, y) = (1-a)(1-b)I(x_0, y_0) + a(1-b)I(x_1, y_0) + (1-a)bI(x_0, y_1) + abI(x_1, y_1)$$

Trong công thức trên, $x_0 = \lfloor x \rfloor$, $y_0 = \lfloor y \rfloor$, $a = x - x_0$ và $b = y - y_0$. Hình 2 minh họa sự khác biệt giữa nội suy nearest-neighbor (răng cưa bậc thang) và bilinear (chuyển mức xám mượt).

```python
def bilinear_sample_gray(
    gray_img: np.ndarray, x: float | np.ndarray, y: float | np.ndarray
) -> float | np.ndarray:
    """Perform bilinear interpolation on a 2D grayscale image with border clamping.

    Formula: f(x,y) = (1-a)(1-b)I(x0,y0) + a(1-b)I(x1,y0) + (1-a)bI(x0,y1) + abI(x1,y1).

    Args:
        gray_img: 2D grayscale array of shape (H, W).
        x: Horizontal coordinate(s) (scalar or array).
        y: Vertical coordinate(s) (scalar or array).
    Returns:
        Interpolated intensity value(s) as float or float64 ndarray.
    """
    img = np.asarray(gray_img, dtype=np.float64)
    h, w = img.shape[:2]
    scalar = np.isscalar(x) and np.isscalar(y)

    xf = np.clip(np.asarray(x, dtype=np.float64), 0.0, float(w - 1))
    yf = np.clip(np.asarray(y, dtype=np.float64), 0.0, float(h - 1))

    x0 = np.floor(xf).astype(np.int32)
    y0 = np.floor(yf).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    a = xf - x0
    b = yf - y0

    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
```

![So sánh Bilinear và Nearest-Neighbor](outputs/figures/resampling_bilinear_vs_nearest.png)
*Hình 2: So sánh crop văn bản phóng to giữa nearest-neighbor và bilinear cùng bản đồ sai lệch tuyệt đối trên dev_01.*

### 2.6. Module nhị phân hóa Otsu và Sauvola (`src/core/thresholding.py`)

Module `src/core/thresholding.py` cài đặt song song hai thuật toán nhị phân hóa cho ảnh xám đã nắn thẳng. Phương pháp Otsu dựng lược đồ 256 mức bằng `np.bincount` và tìm ngưỡng $t^*$ cực đại hóa phương sai giữa hai lớp.

$$\sigma_b^2(t) = w_0(t) w_1(t) \left( \mu_0(t) - \mu_1(t) \right)^2$$

Phương pháp Sauvola tính ngưỡng cục bộ $T(x, y)$ dựa trên trung bình $m(x, y)$ và độ lệch chuẩn $s(x, y)$ trong cửa sổ $w \times w$.

$$T(x, y) = m(x, y) \left[ 1 + k \left( \frac{s(x, y)}{R} - 1 \right) \right], \quad R = 128.0$$

Tổng cục bộ của $I$ và $I^2$ tính nhanh qua bộ lọc hộp `cv2.boxFilter`, mất trung bình khoảng `42.6 ms` trên ảnh lớn.

```python
def sauvola_threshold(
    gray_img: np.ndarray, window_size: int = 25, k: float = 0.2, R: float = 128.0
) -> dict:
    """Compute local adaptive Sauvola threshold map from local mean and std.

    Formula: T(x, y) = m(x, y) * [1 + k * (s(x, y) / R - 1)].

    Args:
        gray_img: 2D grayscale image of shape (H, W).
        window_size: Odd local window size in pixels.
        k: Positive sensitivity parameter (typically 0.1 to 0.5).
        R: Dynamic range of standard deviation (128.0 for 8-bit images).
    Returns:
        Dict with keys threshold_map (float64 shape (H, W)) and binary (uint8 0/255).
    """
    ws = int(window_size)
    if ws % 2 == 0:
        ws += 1
    ws = max(3, ws)

    src = np.clip(gray_img.astype(np.float64), 0.0, 255.0)
    mean = cv2.boxFilter(src, ddepth=cv2.CV_64F, ksize=(ws, ws), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(src * src, ddepth=cv2.CV_64F, ksize=(ws, ws), borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(0.0, mean_sq - mean * mean)
    std = np.sqrt(variance)

```

## CHƯƠNG 3. DỮ LIỆU THỰC NGHIỆM, KIỂM ĐỊNH AST VÀ UNIT TESTS

### 3.1. Cơ cấu dữ liệu 72 ảnh và nguyên tắc No Data Leakage

Bộ dữ liệu gồm 60 ảnh chụp thực tế và 12 ảnh chữ tổng hợp có mask chuẩn. Tập ảnh thực chia thành nhiều nhóm tài liệu quen thuộc như sách, đề thi, vở viết tay và thẻ sinh viên. Ngoài ra còn có giấy tờ dấu đỏ, sổ tay Leaf và vé xe buýt.
Toàn bộ ảnh của cùng một loại tài liệu nằm trọn trong một tập để tránh rò rỉ dữ liệu giữa Development và Test. Các tham số chỉ được phép khảo sát trên Development trước khi đóng băng chạy Test.
Tệp `data/labels.json` lưu trữ đầy đủ tọa độ bốn góc ground-truth cho toàn bộ 50 ảnh dương thực tế. Các nhãn gồm 30 ảnh Development và 20 ảnh Test được minh họa trực quan trong Hình 3.
Mọi ảnh dương bị `rejected` được tính `IoU = 0.0000` trên toàn bộ tập dương. Sai số góc chuẩn hóa chia đường chéo ảnh chỉ tính trên các dự đoán hợp lệ.

![Toàn cảnh tập dữ liệu thực nghiệm](outputs/figures/fig_dataset_overview.png)
*Hình 3: Ảnh đại diện từ các nhóm tài liệu thực tế kèm khung ground-truth 4 góc (magenta) từ data/labels.json.*

*Bảng 2: Phân chia tập dữ liệu thực nghiệm Development và Test (tổng cộng 72 ảnh).*

| Tập dữ liệu | Ảnh thực dương | Ảnh thực âm | Ảnh tổng hợp (Mask) | Mục đích sử dụng |
| :--- | :---: | :---: | :---: | :--- |
| Development | 30 ảnh (6 tài liệu x 5 ĐK) | 6 ảnh | 6 ảnh (Layout 1–2) | Dò tìm và đóng băng bộ tham số |
| Test | 20 ảnh (4 tài liệu x 5 ĐK) | 4 ảnh | 6 ảnh (Layout 3–4) | Đánh giá độc lập cấu hình đóng băng |
| **Tổng cộng** | **50 ảnh dương** | **10 ảnh âm** | **12 ảnh tổng hợp** | **Đánh giá toàn diện pipeline** |

### 3.2. Kiểm định tĩnh AST và Bộ kiểm thử biên (17 Unit Tests)

Script `tests/verify_ast.py` phân tích cây cú pháp trừu tượng toàn bộ 6 tệp trong `src/core/`. Kết quả kiểm tra xác nhận 0 lệnh gọi hoặc import tới các hàm xử lý ảnh bị cấm.
Bộ `pytest` gồm 17 kiểm thử đơn vị phủ đủ các tình huống biên và thuật toán tính tay trong đề bài.

*Bảng 3: Danh sách các trường hợp kiểm thử đơn vị trong thư mục `tests/` (17/17 PASSED).*

| Tệp kiểm thử | Số test | Các ca biên đã kiểm chứng | Kết quả |
| :--- | :---: | :--- | :---: |
| `tests/test_canny.py` | 5 | Ảnh hằng không biên, step edge, NMS làm mảnh, ngưỡng kép 0, hysteresis liên thông 8 | PASSED |
| `tests/test_hough.py` | 2 | Edge map toàn số 0, đường thẳng đứng $0^\circ$ và đường ngang $90^\circ$ | PASSED |
| `tests/test_quadrilateral.py` | 5 | Sắp xếp góc CW, 3 điểm thẳng hàng suy biến, đường song song, tứ giác hợp lệ & thiếu đường | PASSED |
| `tests/test_warping.py` | 3 | `bilinear_sample_gray` tính tay (nguyên/phân số/biên), ma trận đơn vị, phối cảnh biết trước | PASSED |
| `tests/test_thresholding.py` | 2 | Ảnh hằng Otsu/Sauvola không NaN, tách chữ trên mask chuẩn (F1 = 1.00 / >0.85) | PASSED |

## CHƯƠNG 4. KHẢO SÁT ĐỘ NHẠY THAM SỐ TRÊN DEVELOPMENT

### 4.1. Khảo sát tham số Canny (`sigma` và cặp ngưỡng)

Trên tập Development 36 ảnh, thực nghiệm lần lượt thay đổi $\sigma$ và cặp ngưỡng kép để theo dõi biên giấy và kết quả cuối.

*Bảng 4: Độ nhạy của bộ tham số Canny trên tập Development (36 ảnh).*

| Tham số | Giá trị thử | Hợp lệ | Mean IoU | Succ@0.8 | CE Norm | Ảnh âm FP | CPU TB (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sigma` | `0.8` | 24/30 | 0.5286 | 10/30 | 0.0876 | 2/6 | 200.28 |
| `sigma` | `1.4` | 23/30 | 0.5223 | 9/30 | 0.0824 | 3/6 | 173.93 |
| `sigma` | `2.2` | 21/30 | 0.5348 | 9/30 | 0.0603 | 3/6 | 154.84 |
| `thresholds` | `(0.04,0.12)` | 24/30 | 0.5315 | 8/30 | 0.0861 | 3/6 | 184.24 |
| `thresholds` | `(0.06,0.16)` | 23/30 | 0.5223 | 9/30 | 0.0824 | 3/6 | 177.13 |
| `thresholds` | `(0.1,0.25)` | 20/30 | 0.4568 | 7/30 | 0.0854 | 2/6 | 147.90 |

Trên ảnh `dev_01` (`640x480`), khi giảm $\sigma = 0.8$ số điểm biên tăng lên `32,235` (`10.49%`) do nhiễu nền. Khi tăng $\sigma = 2.2$, số điểm biên giảm còn `20,952` (`6.82%`) làm mờ góc giấy.
Cấu hình $\sigma = 1.4$ và ngưỡng `(0.06, 0.16)` giữ `27,108` điểm biên cân bằng giữa nhiễu và độ liền mạch.

![Khảo sát trực quan Canny trên dev_01](outputs/figures/fig_canny_sweep_visual.png)
*Hình 4: Edge map của dev_01 khi thay đổi độ lệch chuẩn Gaussian sigma và cặp ngưỡng kép Canny.*

### 4.2. Khảo sát ngưỡng Hough và chế độ chấm điểm Quadrilateral

Thực nghiệm đối chiếu hàm mục tiêu chỉ dùng diện tích (`area_only`) và hàm kết hợp hỗ trợ biên (`edge_support+area`). Quá trình thử nghiệm chạy qua ba mức ngưỡng Hough khác nhau.

*Bảng 5: So sánh chế độ chấm điểm Area-Only và Edge Support + Area trên Development (36 ảnh).*

| Ngưỡng Hough | Chế độ chấm điểm | Hợp lệ | Mean IoU | Succ@0.8 | CE Norm | Ảnh âm FP | CPU TB (ms) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.18` | `area_only` | 28/30 | 0.6068 | 11/30 | 0.1006 | 5/6 | 179.14 |
| `0.18` | `edge_support+area` | 23/30 | 0.5223 | 9/30 | 0.0824 | 3/6 | 169.19 |
| `0.24` | `area_only` | 27/30 | 0.5965 | 11/30 | 0.0950 | 5/6 | 171.12 |
| `0.24` | `edge_support+area` | 23/30 | 0.5223 | 9/30 | 0.0824 | 3/6 | 166.77 |
| `0.30` | `area_only` | 27/30 | 0.5982 | 11/30 | 0.0946 | 5/6 | 168.61 |
| `0.30` | `edge_support+area` | 23/30 | 0.5223 | 9/30 | 0.0824 | 3/6 | 160.31 |

Chế độ `area_only` chấp nhận khung lớn trên nền nên Mean IoU đạt cao hơn một chút nhưng FPR ảnh âm lên tới 5/6 (`83.3%`). Kết hợp `edge_support+area` ở ngưỡng `0.24` giảm FPR ảnh âm xuống 3/6 (`50.0%`) và cải thiện sai số góc còn `0.0824`.

### 4.3. Khảo sát tham số Otsu và Sauvola trên tập Development

Trên 6 ảnh tổng hợp Development, thuật toán thay đổi từng yếu tố `window_size` và hệ số `k` của Sauvola để chọn cấu hình tối ưu.

*Bảng 6: Kết quả quét tham số Otsu và Sauvola trên 6 ảnh chữ tổng hợp Development.*

| Phương pháp | Window Size | Tham số k | Precision TB | Recall TB | F1 TB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| otsu | N/A | N/A | 0.7022 | 1.0000 | 0.7321 |
| sauvola | 15 | 0.2 | 0.9757 | 1.0000 | 0.9874 |
| sauvola | 25 | 0.2 | 0.9765 | 1.0000 | 0.9878 |
| sauvola | 41 | 0.2 | 0.9728 | 1.0000 | 0.9858 |
| sauvola | 25 | 0.1 | 0.8614 | 1.0000 | 0.9124 |
| sauvola | 25 | 0.2 | 0.9765 | 1.0000 | 0.9878 |
| sauvola | 25 | 0.35 | 0.9998 | 1.0000 | 0.9999 |

Sauvola tại `w = 25` và `k = 0.35` đạt F1 trung bình `0.9999`, triệt tiêu hoàn toàn nền gradient. Trên crop bóng đổ ảnh thật (Hình 5), `k = 0.10` còn chấm nhiễu nền trong khi `k = 0.35` làm sạch hẳn vùng râm.

![Lưới khảo sát tham số Sauvola trên ảnh bóng đổ thật](outputs/figures/fig_sauvola_param_grid.png)
*Hình 5: Ảnh hưởng của window_size và k của Sauvola trên vùng chữ thật chịu bóng đổ kèm threshold map T(x,y).*

### 4.4. Cấu hình đóng băng (`outputs/frozen_config.json`)

Toàn bộ tham số chốt trên tập Development được đóng băng trước khi chạy đánh giá tập Test:

```json
{
  "work_max_dim": 640,
  "sigma": 1.4,
  "low_ratio": 0.06,
  "high_ratio": 0.16,
  "use_gaussian": true,
  "rho_res": 1.0,
  "theta_res_deg": 1.0,
  "hough_thresh_ratio": 0.24,
  "hough_min_votes": 45,
  "top_k_lines": 20,
  "hough_suppress": true,
  "rho_supp": 18.0,
  "theta_supp_deg": 12.0,
  "use_edge_support": true,
  "min_area_ratio": 0.14,
  "max_area_ratio": 0.98,
  "min_edge_support": 0.72,
  "min_side_support": 0.5,
  "min_angle_deg": 45.0,
  "max_angle_deg": 135.0,
  "sauvola_window": 25,
  "sauvola_k": 0.35,
  "warp_max_dim": 1200
}
```

## CHƯƠNG 5. KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG VÀ SO SÁNH TRỰC QUAN

### 5.1. So sánh Custom Pipeline và OpenCV Baseline theo từng điều kiện

Trên tập Development, pipeline tự cài đạt Mean IoU `0.5222` và `9/30` ảnh đạt `IoU >= 0.8`. OpenCV Baseline chỉ đạt Mean IoU `0.3522` với `7/30` ảnh vượt ngưỡng `0.8`.
Trên tập Test đóng băng, pipeline tự cài đạt Mean IoU `0.4668` (hợp lệ `17/20`), cao hơn Baseline (`0.2050`). Tỷ lệ báo giả trên ảnh âm Test đạt `1/4` (`25.0%`).

*Bảng 7a: Kết quả định lượng theo từng điều kiện trên tập Development (36 ảnh).*

| Điều kiện chụp (Dev) | Custom Hợp lệ | Custom IoU | Custom Succ@0.8 | Custom CE | Baseline IoU | Baseline Succ@0.8 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Nền đơn giản, đủ sáng | 6/6 | 0.7322 | 3/6 | 0.0736 | 0.4496 | 2/6 |
| Góc chụp xiên | 3/6 | 0.3084 | 1/6 | 0.1002 | 0.2892 | 1/6 |
| Nền chữ nhật gây nhiễu | 6/6 | 0.7346 | 3/6 | 0.0666 | 0.4831 | 2/6 |
| Biên tương phản thấp | 4/6 | 0.4721 | 1/6 | 0.0746 | 0.2616 | 1/6 |
| Ánh sáng không đều / bóng đổ | 4/6 | 0.3640 | 1/6 | 0.1141 | 0.2774 | 1/6 |
| Ảnh âm (không có tài liệu) | FP 3/6 | - | - | - | FP 1/6 | - |
| **Tổng ảnh dương Dev** | **23/30** | **0.5222** | **9/30 (30.0%)** | **0.0825** | **0.3522** | **7/30 (23.3%)** |

*Bảng 7b: Kết quả định lượng theo từng điều kiện trên tập Test đóng băng (24 ảnh).*

| Điều kiện chụp (Test) | Custom Hợp lệ | Custom IoU | Custom Succ@0.8 | Custom CE | Baseline IoU | Baseline Succ@0.8 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Nền đơn giản, đủ sáng | 4/4 | 0.5560 | 0/4 | 0.1236 | 0.3063 | 1/4 |
| Góc chụp xiên | 2/4 | 0.2098 | 0/4 | 0.1740 | 0.1002 | 0/4 |
| Nền chữ nhật gây nhiễu | 4/4 | 0.4035 | 0/4 | 0.1397 | 0.1443 | 0/4 |
| Biên tương phản thấp | 3/3 | 0.5802 | 1/3 | 0.1004 | 0.5112 | 1/3 |
| Ánh sáng không đều / bóng đổ | 4/5 | 0.5837 | 2/5 | 0.0688 | 0.0726 | 0/5 |
| Ảnh âm (không có tài liệu) | FP 1/4 | - | - | - | FP 0/4 | - |
| **Tổng ảnh dương Test** | **17/20** | **0.4668** | **3/20 (15.0%)** | **0.1163** | **0.2050** | **2/20 (10.0%)** |

### 5.2. Nghiên cứu tách biệt thành phần (Ablation Study)

Ba ca Ablation chạy trên Development với cùng giới hạn `top_k_lines = 20` và toàn bộ tham số còn lại cố định. Bỏ Gaussian smoothing làm `Succ@0.8` giảm từ `9/30` xuống `7/30` và thời gian CPU tăng lên `228.2 ms`.
Bỏ triệt tiêu đường Hough trùng làm Mean IoU tụt sâu từ `0.5222` xuống `0.3643`. Nguyên nhân là các đường song song sát nhau ở một cạnh mạnh chiếm hết 20 ứng viên đầu tiên.
Bỏ thành phần edge-support giúp chọn khung lớn hơn (`Mean IoU = 0.5965`) nhưng ảnh âm báo giả vọt lên `5/6` (`83.3%`).

*Bảng 8: Đối chiếu định lượng giữa Custom, Baseline và 3 ca Ablation trên tập Development (36 ảnh).*

| Biến thể cấu hình | Hợp lệ | Mean IoU | Succ@0.8 | CE Norm | Ảnh âm FP | CPU Acc (ms) | CPU Rej (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Custom (Đầy đủ) | 23/30 | 0.5222 | 9/30 (30.0%) | 0.0825 | 3/6 | 182.1 | 128.0 |
| Baseline (OpenCV) | 13/30 | 0.3522 | 7/30 (23.3%) | 0.0446 | 1/6 | 122.9 | 29.1 |
| Ablation 1: Bỏ Gaussian | 23/30 | 0.5132 | 7/30 (23.3%) | 0.0833 | 2/6 | 228.2 | 150.9 |
| Ablation 2: Bỏ Hough Supp. | 16/30 | 0.3643 | 9/30 (30.0%) | 0.0877 | 3/6 | 177.8 | 121.7 |
| Ablation 3: Bỏ Edge Support | 27/30 | 0.5965 | 11/30 (36.7%) | 0.0950 | 5/6 | 177.4 | 119.9 |

![Biểu đồ Ablation và Baseline](outputs/figures/chart_ablation_and_baseline.png)
*Hình 6: So sánh tỷ lệ nhận diện ảnh dương, tỷ lệ báo giả ảnh âm và thời gian CPU trên tập Development.*

![Minh họa trực quan tác động của Ablation](outputs/figures/fig_ablation_visual.png)
*Hình 7: Tác động trực quan của 3 ca Ablation trên mẫu dương dev_06 (mất tứ giác khi bỏ Hough suppression) và mẫu âm dev_33 (nhận nhầm khi bỏ edge support).*

### 5.3. Đánh giá nhị phân hóa trên 12 ảnh tổng hợp và Crop ảnh thực tế

Trên 12 ảnh chữ tổng hợp (`480x360`), Sauvola tự cài đạt F1 trung bình `0.9999` (Dev) và `0.9999` (Test). Otsu toàn cục sụt giảm xuống `0.7321` (Dev) và `0.5979` (Test) khi có gradient hoặc bóng đổ.
Hình 10 đối chiếu trực tiếp ba vùng crop văn bản thực tế giữa Otsu toàn cục và Sauvola cục bộ. Trên thẻ sinh viên `dev_26` (nền hoa văn gradient), Otsu bôi đen vùng tối trong khi Sauvola giữ sạch nét chữ.
Trên vé xe buýt `test_01` (mực in kim nhạt), Otsu với ngưỡng $T^* = 164$ sai khác `32.22%` điểm ảnh và làm mất chấm chữ mảnh. Trên mẫu bóng đổ `dev_10`, Sauvola bảo tồn trọn vẹn chữ vùng tối vượt trội so với ngưỡng đơn $T^* = 166$.

*Bảng 9: Precision, Recall và F1 lớp ink trên 12 ảnh tổng hợp tách riêng Development và Test.*

| Tập dữ liệu | Điều kiện | Otsu (P / R / F1) | Sauvola (P / R / F1) |
| :--- | :--- | :---: | :---: |
| Development (6 ảnh) | Nền đều | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| Development (6 ảnh) | Gradient sáng | 0.9967 / 1.0000 / 0.9983 | 1.0000 / 1.0000 / 1.0000 |
| Development (6 ảnh) | Bóng + nhiễu | 0.1099 / 1.0000 / 0.1980 | 0.9992 / 1.0000 / 0.9996 |
| Development (6 ảnh) | Trung bình toàn tập | 0.7022 / 1.0000 / 0.7321 | 0.9998 / 1.0000 / 0.9999 |
| Test (6 ảnh) | Nền đều | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| Test (6 ảnh) | Gradient sáng | 0.5757 / 1.0000 / 0.6343 | 1.0000 / 1.0000 / 1.0000 |
| Test (6 ảnh) | Bóng + nhiễu | 0.0865 / 1.0000 / 0.1593 | 0.9993 / 1.0000 / 0.9997 |
| Test (6 ảnh) | Trung bình toàn tập | 0.5541 / 1.0000 / 0.5979 | 0.9998 / 1.0000 / 0.9999 |

![Biểu đồ F1 trên 12 ảnh tổng hợp](outputs/figures/chart_synthetic_f1.png)
*Hình 8: So sánh chỉ số F1 lớp ink giữa Otsu toàn cục và Sauvola cục bộ trên 12 ảnh chữ tổng hợp.*

![So sánh Otsu và Sauvola trên ảnh bóng đổ](outputs/figures/thresholding_otsu_vs_sauvola.png)
*Hình 9: Lược đồ mức xám, ngưỡng Otsu, bản đồ ngưỡng Sauvola và ảnh nhị phân trên dev_20 và syn_test_03.*

![So sánh phóng to crop nét chữ thực tế Otsu vs Sauvola](outputs/figures/fig_crop_otsu_vs_sauvola_real.png)
*Hình 10: Đối chiếu crop nét chữ thực tế trên Thẻ sinh viên (dev_26), Vé xe buýt (test_01) và Tài liệu bóng đổ (dev_10).*

### 5.4. Minh họa chuỗi 6 bước xử lý trên nhiều loại tài liệu khác nhau

Chuỗi ảnh trung gian từ grayscale, Canny, Hough, tứ giác, nắn phẳng đến Sauvola được xuất đầy đủ trên năm loại tài liệu khác nhau. Các mẫu gồm sách HP Book (`dev_01`), thẻ sinh viên (`dev_26`) và giấy tờ dấu đỏ (`dev_21`). Danh sách còn có vé xe buýt (`test_01`) và sổ tay Leaf (`dev_16`).

![Các bước trung gian trên dev_01](outputs/figures/pipeline_stages_dev01.png)
*Hình 11: Tiến trình 6 bước trung gian trên mẫu sách HP Book dev_01 (edge support = 0.87, score = 0.71).*

![Hình 12: Chuỗi 6 bước trên Thẻ sinh viên (dev_26, IoU = 0.9828, CE = 0.0021).](outputs/figures/fig_stages_thesv_dev26.png)
*Hình 12: Chuỗi 6 bước trên Thẻ sinh viên (dev_26, IoU = 0.9828, CE = 0.0021).*

![Hình 13: Chuỗi 6 bước trên Giấy tờ dấu đỏ (dev_21, IoU = 0.5601, CE = 0.1304).](outputs/figures/fig_stages_sign_dev21.png)
*Hình 13: Chuỗi 6 bước trên Giấy tờ dấu đỏ (dev_21, IoU = 0.5601, CE = 0.1304).*

![Hình 14: Chuỗi 6 bước trên Vé xe buýt tập Test (test_01, IoU = 0.6628, CE = 0.0723).](outputs/figures/fig_stages_bus_test01.png)
*Hình 14: Chuỗi 6 bước trên Vé xe buýt tập Test (test_01, IoU = 0.6628, CE = 0.0723).*

![Hình 15: Chuỗi 6 bước trên Sổ tay bìa Leaf (dev_16, IoU = 0.9830, CE = 0.0046).](outputs/figures/fig_stages_leaf_dev16.png)
*Hình 15: Chuỗi 6 bước trên Sổ tay bìa Leaf (dev_16, IoU = 0.9830, CE = 0.0046).*

### 5.5. Phân bổ thời gian CPU theo từng module

Thời gian xử lý được đo thuần từ mảng ảnh trong RAM đến đầu ra, loại trừ đọc ghi đĩa và vẽ hình. Trên 36 ảnh Development (gốc tới `2560x1922`), ca `accepted` mất `185.10 ms` và ca `rejected` mất `123.56 ms`.
Trên tập Test (24 ảnh), thời gian trung bình là `141.88 ms` cho nhóm `accepted` và `120.17 ms` cho nhóm `rejected`.

*Bảng 10: Thời gian thực thi trung bình trên CPU theo từng module tại tập Development.*

| Giai đoạn xử lý | Module | Thời gian TB (ms) | Tỷ trọng (%) |
| :--- | :--- | :---: | :---: |
| 1. Resize + Canny Edge | `src/core/canny.py` | 45.06 | 25.5% |
| 2. Hough Transform + NMS | `src/core/hough.py` | 58.39 | 33.0% |
| 3. Quadrilateral Selection | `src/core/quadrilateral.py` | 41.56 | 23.5% |
| 4. Homography + Warping | `src/core/warping.py` | 3.63 | 2.1% |
| 5. Otsu + Sauvola Thresholding | `src/core/thresholding.py` | 28.21 | 16.0% |

![Phân bổ thời gian CPU theo từng module](outputs/figures/fig_runtime_breakdown.png)
*Hình 16: Phân bổ thời gian xử lý CPU thuần trên 5 module lõi (ba bước định vị chiếm 84.6% tổng thời gian).*

## CHƯƠNG 6. PHÂN TÍCH CHI TIẾT 6 TRƯỜNG HỢP TIÊU BIỂU

### 6.1. Trường hợp 1: Ảnh chuẩn, nền đơn giản (dev_01)

- *Đầu vào & Điều kiện chụp:* Ảnh chụp trang sách nền đồng nhất, kích thước gốc `2560x1922` (`640x480` khi xử lý), ánh sáng đều.
- *Điểm gãy trong pipeline:* Trạng thái `accepted` sau `208.15 ms`, đạt `IoU = 0.6922` và `CE = 0.0674`.
- *Bằng chứng trên ảnh trung gian:* Edge map sạch (8.82% pixel biên), accumulator đạt đỉnh `254` phiếu, edge support `0.8744`.
- *Nguyên nhân & Tham số liên quan:* Khung tranh in trên bìa sách có hỗ trợ biên rất cao. Do đó, tứ giác chọn bị co nhẹ vào khung trong.

![Hình 17: Ảnh trung gian chẩn đoán ca dev_01 (Xanh=Dự đoán, Hồng=Ground Truth).](outputs/figures/case1_dev01_good.png)
*Hình 17: Ảnh trung gian chẩn đoán ca dev_01 (Xanh=Dự đoán, Hồng=Ground Truth).*

### 6.2. Trường hợp 2: Góc chụp xiên mạnh (dev_07)

- *Đầu vào & Điều kiện chụp:* Ảnh trang tài liệu đặt nghiêng phối cảnh lớn (`2560x1922`), bề mặt chứa nhiều dòng chữ ngang.
- *Điểm gãy trong pipeline:* Pipeline dừng tại bước Quadrilateral với trạng thái `rejected`, lý do `low_edge_support` và `IoU = 0.0000`.
- *Bằng chứng trên ảnh trung gian:* Edge map chiếm `7.44%`, accumulator có cực đại `259` phiếu nhưng các ứng viên tứ giác không đạt ngưỡng.
- *Nguyên nhân & Tham số liên quan:* Góc nghiêng làm cạnh ngắn bị đứt đoạn hoặc góc giao lệch chuẩn. Hỗ trợ biên tụt xuống dưới ngưỡng `0.72`.

![Hình 18: Ảnh trung gian chẩn đoán ca góc xiên dev_07.](outputs/figures/case2_dev07_oblique.png)
*Hình 18: Ảnh trung gian chẩn đoán ca góc xiên dev_07.*

### 6.3. Trường hợp 3: Nền có vật hình chữ nhật gây nhiễu (dev_03)

- *Đầu vào & Điều kiện chụp:* Ảnh trang sách đặt trên mặt bàn phím nhiều phím chữ nhật (`2560x1922`).
- *Điểm gãy trong pipeline:* Không gãy; loại bỏ khung bàn phím nhiễu, đạt `accepted` với `IoU = 0.8615` và `CE = 0.0448`.
- *Bằng chứng trên ảnh trung gian:* Hough accumulator có đỉnh `303` phiếu; ứng viên thắng đạt edge support `0.9491` và `area_ratio = 0.5455`.
- *Nguyên nhân & Tham số liên quan:* Hàm điểm kết hợp `0.65 * edge_support + 0.35 * area_ratio`. Trọng số này ưu tiên tứ giác bám sát biên trang sách.

![Hình 19: Ảnh trung gian chẩn đoán ca nền nhiễu chữ nhật dev_03.](outputs/figures/case3_dev03_clutter.png)
*Hình 19: Ảnh trung gian chẩn đoán ca nền nhiễu chữ nhật dev_03.*

### 6.4. Trường hợp 4: Biên giấy tương phản thấp (dev_29)

- *Đầu vào & Điều kiện chụp:* Ảnh thẻ sinh viên đặt trên nền sáng có độ chênh mức xám biên rất thấp (`1536x2048`).
- *Điểm gãy trong pipeline:* Pipeline gãy tại bước Quadrilateral với lý do `short_side` và `IoU = 0.0000`.
- *Bằng chứng trên ảnh trung gian:* Edge map (`7.44%`) xuất hiện nhiều cạnh chữ bên trong thẻ. Ngược lại, cạnh biên ngoài bị đứt đoạn.
- *Nguyên nhân & Tham số liên quan:* Top-20 đường Hough (`max = 345` phiếu) bị các đường chữ bên trong chiếm chỗ. Ứng viên tạo thành nhỏ hơn `min_side_ratio`.

![Hình 20: Ảnh trung gian chẩn đoán ca biên tương phản thấp dev_29.](outputs/figures/case4_dev29_lowcontrast.png)
*Hình 20: Ảnh trung gian chẩn đoán ca biên tương phản thấp dev_29.*

### 6.5. Trường hợp 5: Ánh sáng không đều và bóng đổ (dev_05)

- *Đầu vào & Điều kiện chụp:* Ảnh trang vở chịu bóng đổ chéo mạnh làm tối một phần cạnh biên (`1922x2560`).
- *Điểm gãy trong pipeline:* Pipeline dừng tại bước Quadrilateral với mã lỗi `out_of_bounds` và `IoU = 0.0000`.
- *Bằng chứng trên ảnh trung gian:* Tỷ lệ pixel biên chỉ đạt `5.31%`, đỉnh Hough `227` phiếu. Các đường cùng hướng giao ngoài khung ảnh.
- *Nguyên nhân & Tham số liên quan:* Bóng đổ làm suy giảm gradient tại cạnh tối. Hệ thống ghép nhầm đường gân chữ gần song song tạo giao điểm ngoài biên.

![Hình 21: Ảnh trung gian chẩn đoán ca bóng đổ dev_05.](outputs/figures/case5_dev05_shadow.png)
*Hình 21: Ảnh trung gian chẩn đoán ca bóng đổ dev_05.*

### 6.6. Trường hợp 6: Ảnh âm không có tài liệu hợp lệ (dev_32 và dev_33)

- *Đầu vào & Điều kiện chụp:* Ảnh âm `dev_32` (`2048x1536`) chụp màn hình chữ nhật; ảnh `dev_33` (`960x1280`) chụp mảnh giấy xé.
- *Điểm gãy trong pipeline:* `dev_32` gây false-positive (`accepted`, `IoU = 0.0000`); `dev_33` bị loại đúng (`rejected: short_side`).
- *Bằng chứng trên ảnh trung gian:* Trên `dev_32`, khung màn hình tạo `13` peaks sắc nét (`edge_support = 0.9271`, `area_ratio = 0.6236`); trên `dev_33`, đỉnh Hough chỉ đạt `161` phiếu.
- *Nguyên nhân & Tham số liên quan:* Bộ lọc hình học thuần túy chưa phân biệt được viền màn hình phẳng với trang giấy. Hệ thống còn thiếu bước kiểm tra mật độ chữ.

![Hình 22: Ảnh trung gian chẩn đoán ca ảnh âm false-positive dev_32.](outputs/figures/case6_dev32_negative_fp.png)
*Hình 22: Ảnh trung gian chẩn đoán ca ảnh âm false-positive dev_32.*

![Đối chiếu ảnh âm bị từ chối dev_33](outputs/figures/case6b_dev33_negative_rej.png)
*Hình 23: Minh chứng ảnh âm giấy xé dev_33 bị từ chối đúng với lý do short_side.*

## CHƯƠNG 7. BÀN LUẬN KỸ THUẬT, HẠN CHẾ VÀ HƯỚNG DẪN TÁI LẬP

### 7.1. Bàn luận ưu điểm, hạn chế và hướng phát triển

Đối chiếu với 50 nhãn ground-truth, pipeline tự cài đạt Mean IoU `0.5222` (Dev) và `0.4668` (Test). Nhiều mẫu sạch đạt độ khớp gần tuyệt đối (`IoU > 0.98` như `dev_02`, `dev_16`, `dev_18`, `dev_26`), vượt trội Baseline OpenCV.
Tuy nhiên, chỉ số `Success@0.8` còn hạn chế ở nhóm sách có khung hình lồng bên trong (`dev_01`, `dev_21`, `test_06`). Nguyên nhân là hàm điểm ưu tiên độ khớp biên cao của khung chữ nhật nội tại thay vì mép giấy ngoài cùng.
Hướng nâng cấp gồm ưu tiên khung bao ngoài cùng khi các tứ giác lồng nhau. Đồng thời, cần thêm bộ kiểm tra mật độ chữ để giảm FPR ảnh âm.

### 7.2. Lệnh tái lập toàn bộ mã nguồn và thực nghiệm

Toàn bộ kết quả trên CPU (với `np.random.seed(42)`) tái lập bằng chuỗi lệnh sau:

```powershell
python tests/verify_ast.py
python -m pytest tests/ -v
python scripts/generate_synthetic.py
python scripts/run_all_experiments.py
python scripts/generate_figures.py
python scripts/generate_rich_visuals.py
python scripts/build_report.py
python scripts/audit_report.py
```

## CHƯƠNG 8. TÀI LIỆU THAM KHẢO

1. Canny, J. (1986). A Computational Approach to Edge Detection. *IEEE TPAMI*, 8(6), 679–698.
2. Duda, R. O., & Hart, P. E. (1972). Use of the Hough Transformation to Detect Lines and Curves in Pictures. *CACM*, 15(1), 11–15.
3. Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE TSMC*, 9(1), 62–66.
4. Sauvola, J., & Pietikainen, M. (2000). Adaptive document image binarization. *Pattern Recognition*, 33(2), 225–236.
5. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.

<!-- LANDSCAPE_APPENDIX -->
## PHỤ LỤC A. BẢNG KẾT QUẢ CHI TIẾT TỪNG ẢNH (DEVELOPMENT & TEST)

*Bảng A.1: Kết quả chi tiết 36 ảnh tập Development trích xuất từ `outputs/results_development.csv`.*

| Mã ảnh | Điều kiện | Trạng thái | Lý do từ chối | IoU | Corner Err | Runtime (ms) | Độ phân giải |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| dev_01 | nen_don_gian_du_sang | accepted | - | 0.6922 | 0.0674 | 208.15 | 2560x1922 |
| dev_02 | goc_chup_xien | accepted | - | 0.9876 | 0.0034 | 225.76 | 1922x2560 |
| dev_03 | nen_chu_nhat_gay_nhieu | accepted | - | 0.8615 | 0.0448 | 222.35 | 2560x1922 |
| dev_04 | bien_tuong_phan_thap | accepted | - | 0.8967 | 0.0305 | 222.54 | 2560x1922 |
| dev_05 | anh_sang_khong_deu_bong_do | rejected | out_of_bounds | 0.0000 | N/A | 102.04 | 1922x2560 |
| dev_06 | nen_don_gian_du_sang | accepted | - | 0.9370 | 0.0152 | 185.79 | 1922x2560 |
| dev_07 | goc_chup_xien | rejected | low_edge_support | 0.0000 | N/A | 146.55 | 2560x1922 |
| dev_08 | nen_chu_nhat_gay_nhieu | accepted | - | 0.8149 | 0.0568 | 194.26 | 2560x1922 |
| dev_09 | bien_tuong_phan_thap | accepted | - | 0.7022 | 0.0934 | 212.67 | 2560x1922 |
| dev_10 | anh_sang_khong_deu_bong_do | accepted | - | 0.2260 | 0.2130 | 149.14 | 1922x2560 |
| dev_11 | nen_don_gian_du_sang | accepted | - | 0.2382 | 0.2216 | 203.00 | 2560x1922 |
| dev_12 | goc_chup_xien | accepted | - | 0.2251 | 0.2245 | 184.69 | 2560x1922 |
| dev_13 | nen_chu_nhat_gay_nhieu | accepted | - | 0.6996 | 0.0843 | 203.78 | 1922x2560 |
| dev_14 | bien_tuong_phan_thap | rejected | short_side | 0.0000 | N/A | 163.83 | 2560x1922 |
| dev_15 | anh_sang_khong_deu_bong_do | rejected | out_of_bounds | 0.0000 | N/A | 127.99 | 1922x2560 |
| dev_16 | nen_don_gian_du_sang | accepted | - | 0.9830 | 0.0046 | 96.66 | 1536x2048 |
| dev_17 | goc_chup_xien | rejected | out_of_bounds | 0.0000 | N/A | 73.51 | 2048x1536 |
| dev_18 | nen_chu_nhat_gay_nhieu | accepted | - | 0.9897 | 0.0032 | 109.73 | 1536x2048 |
| dev_19 | bien_tuong_phan_thap | accepted | - | 0.6207 | 0.0945 | 182.37 | 1536x2048 |
| dev_20 | anh_sang_khong_deu_bong_do | accepted | - | 0.8750 | 0.0432 | 153.15 | 1536x2048 |
| dev_21 | nen_don_gian_du_sang | accepted | - | 0.5601 | 0.1304 | 222.47 | 1536x2048 |
| dev_22 | goc_chup_xien | accepted | - | 0.6378 | 0.0728 | 200.92 | 1536x2048 |
| dev_23 | nen_chu_nhat_gay_nhieu | accepted | - | 0.7909 | 0.0495 | 200.97 | 1536x2048 |
| dev_24 | bien_tuong_phan_thap | accepted | - | 0.6127 | 0.0802 | 218.62 | 1536x2048 |
| dev_25 | anh_sang_khong_deu_bong_do | accepted | - | 0.6540 | 0.0798 | 176.80 | 2048x1536 |
| dev_26 | nen_don_gian_du_sang | accepted | - | 0.9828 | 0.0021 | 129.72 | 1536x2048 |
| dev_27 | goc_chup_xien | rejected | short_side | 0.0000 | N/A | 170.77 | 1536x2048 |
| dev_28 | nen_chu_nhat_gay_nhieu | accepted | - | 0.2509 | 0.1607 | 193.08 | 1536x2048 |
| dev_29 | bien_tuong_phan_thap | rejected | short_side | 0.0000 | N/A | 138.19 | 1536x2048 |
| dev_30 | anh_sang_khong_deu_bong_do | accepted | - | 0.4288 | 0.1206 | 145.14 | 1536x2048 |
| dev_31 | khong_co_tai_lieu_hop_le | accepted | - | 0.0000 | N/A | 264.53 | 1536x2048 |
| dev_32 | khong_co_tai_lieu_hop_le | accepted | - | 0.0000 | N/A | 129.58 | 2048x1536 |
| dev_33 | khong_co_tai_lieu_hop_le | rejected | short_side | N/A | N/A | 130.05 | 960x1280 |
| dev_34 | khong_co_tai_lieu_hop_le | rejected | short_side | N/A | N/A | 115.66 | 960x1280 |
| dev_35 | khong_co_tai_lieu_hop_le | rejected | short_side | N/A | N/A | 110.99 | 960x1280 |
| dev_36 | khong_co_tai_lieu_hop_le | accepted | - | 0.0000 | N/A | 98.02 | 960x1280 |

*Bảng A.2: Kết quả chi tiết 24 ảnh tập Test trích xuất từ `outputs/results_test.csv`.*

| Mã ảnh | Điều kiện | Trạng thái | Lý do từ chối | IoU | Corner Err | Runtime (ms) | Độ phân giải |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| test_01 | nen_don_gian_du_sang | accepted | - | 0.6628 | 0.0723 | 127.43 | 1536x2048 |
| test_02 | goc_chup_xien | rejected | out_of_bounds | 0.0000 | N/A | 89.58 | 1536x2048 |
| test_03 | nen_chu_nhat_gay_nhieu | accepted | - | 0.5765 | 0.0760 | 139.87 | 1536x2048 |
| test_04 | bien_tuong_phan_thap | accepted | - | 0.8523 | 0.0267 | 141.01 | 1536x2048 |
| test_05 | anh_sang_khong_deu_bong_do | rejected | no_orthogonal_line_pairs | 0.0000 | N/A | 85.38 | 1536x2048 |
| test_06 | nen_don_gian_du_sang | accepted | - | 0.2732 | 0.2235 | 166.56 | 1922x2560 |
| test_07 | goc_chup_xien | accepted | - | 0.3150 | 0.2027 | 138.33 | 960x1280 |
| test_08 | nen_chu_nhat_gay_nhieu | accepted | - | 0.3178 | 0.1875 | 118.38 | 960x1280 |
| test_09 | bien_tuong_phan_thap | accepted | - | 0.4224 | 0.1482 | 112.78 | 960x1280 |
| test_10 | anh_sang_khong_deu_bong_do | accepted | - | 0.8347 | 0.0352 | 128.22 | 960x1280 |
| test_11 | nen_don_gian_du_sang | accepted | - | 0.6443 | 0.1146 | 123.13 | 960x1280 |
| test_12 | goc_chup_xien | accepted | - | 0.5241 | 0.1453 | 113.73 | 960x1280 |
| test_13 | nen_chu_nhat_gay_nhieu | accepted | - | 0.3751 | 0.1449 | 89.36 | 960x1280 |
| test_14 | bien_tuong_phan_thap | accepted | - | 0.4660 | 0.1264 | 93.84 | 960x1280 |
| test_15 | anh_sang_khong_deu_bong_do | accepted | - | 0.9208 | 0.0227 | 155.95 | 960x1280 |
| test_16 | nen_don_gian_du_sang | accepted | - | 0.6437 | 0.0842 | 180.37 | 2560x1922 |
| test_17 | goc_chup_xien | rejected | out_of_bounds | 0.0000 | N/A | 125.38 | 1922x2560 |
| test_18 | nen_chu_nhat_gay_nhieu | accepted | - | 0.3444 | 0.1503 | 173.85 | 1922x2560 |
| test_19 | anh_sang_khong_deu_bong_do | accepted | - | 0.6214 | 0.1150 | 212.94 | 1922x2560 |
| test_20 | anh_sang_khong_deu_bong_do | accepted | - | 0.5418 | 0.1023 | 85.27 | 960x1280 |
| test_21 | khong_co_tai_lieu_hop_le | rejected | short_side | N/A | N/A | 154.74 | 1536x2048 |
| test_22 | khong_co_tai_lieu_hop_le | accepted | - | 0.0000 | N/A | 252.80 | 1536x2048 |
| test_23 | khong_co_tai_lieu_hop_le | rejected | short_side | N/A | N/A | 113.38 | 960x1280 |
| test_24 | khong_co_tai_lieu_hop_le | rejected | short_side | N/A | N/A | 152.57 | 960x1280 |
