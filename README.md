# Document Scanner: Phát hiện tài liệu, Hiệu chỉnh phối cảnh & Nhị phân hóa

> **Đồ án Giữa kỳ — Học phần: Nhập môn Xử lý ảnh và Video**  
> **Trường Đại học Khoa học Tự nhiên — ĐHQG TP.HCM**  
> **Khoa Điện tử - Viễn thông — Bộ môn Máy tính - Hệ thống nhúng (Lớp 23MTNhung)**

---

## 0. Bản đồ nộp bài nhanh (Đối chiếu 5 mục yêu cầu đồ án)

| Mục | Yêu cầu nộp bài | Vị trí tệp / thư mục tương ứng (bấm mở trực tiếp) |
| :---: | :--- | :--- |
| **1** | **Source** (thuật toán tự cài, kiểm tra, thí nghiệm, ảnh trung gian, outputs) | [src/core/](src/core/) · [src/baseline/](src/baseline/) · [run_pipeline.py](run_pipeline.py) · [tests/](tests/) · [scripts/](scripts/) · [outputs/intermediates/](outputs/intermediates/) |
| **2** | **Dữ liệu & Kết quả** (ảnh thật, split, nhãn GT 4 góc, mask synthetic, cấu hình baseline/cuối/ablation, CSV & ảnh minh chứng) | [data/development/](data/development/) · [data/test/](data/test/) · [data/labels.json](data/labels.json) · [data/synthetic/](data/synthetic/) · [outputs/frozen_config.json](outputs/frozen_config.json) · [outputs/baseline_config.json](outputs/baseline_config.json) · [outputs/ablation_configs.json](outputs/ablation_configs.json) · [outputs/results_development.csv](outputs/results_development.csv) · [outputs/results_test.csv](outputs/results_test.csv) · [outputs/figures/](outputs/figures/) |
| **3** | **Thông tin chung & Hướng dẫn** (thành viên, đóng góp, dependency, seed, lệnh chạy đường dẫn tương đối) | Xem Mục **1**, **2**, **4**, **5** ngay trong [README.md](README.md) và [requirements.txt](requirements.txt) |
| **4** | **Báo cáo PDF** (pipeline, tham số, so sánh, 6 ca phân tích, kết luận, tài liệu tham khảo) | [outputs/Bao_cao_Giua_Ky_Document_Scanner.pdf](outputs/Bao_cao_Giua_Ky_Document_Scanner.pdf) *(kèm bản Word [.docx](outputs/Bao_cao_Giua_Ky_Document_Scanner.docx))* |
| **5** | **Slides trình chiếu** (mục tiêu, pipeline, phần tự cài, dữ liệu, thí nghiệm & kết quả tiêu biểu) | [outputs/Slides_Giua_Ky_Document_Scanner.pptx](outputs/Slides_Giua_Ky_Document_Scanner.pptx) *(kèm bản PDF [.pdf](outputs/Slides_Giua_Ky_Document_Scanner.pdf))* |

---

## 1. Thông tin thành viên và Bảng đóng góp

| STT | Họ và tên | MSSV | Tỷ lệ đóng góp | Nhiệm vụ chính |
| :---: | :--- | :---: | :---: | :--- |
| 1 | Nguyễn Vũ Tuấn Khoa | 23200096 | 34% | Thiết kế kiến trúc pipeline, cài đặt `canny.py`, `hough.py`, `quadrilateral.py`, chạy thực nghiệm, đánh giá và tổng hợp báo cáo |
| 2 | Lê Ngọc Long | 23200107 | 33% | Cài đặt `warping.py`, `thresholding.py`, xây dựng bộ sinh 12 ảnh synthetic kèm ground-truth mask và 17 unit tests kiểm thử biên |
| 3 | Nguyễn Thanh Tuấn | 23200162 | 33% | Thu thập & phân tách 60 ảnh thực tế, gắn nhãn 50 ground-truth góc (`data/labels.json`), chạy baseline/ablation và trực quan hóa kết quả |
| **Tổng** | **3 thành viên** | — | **100%** | — |

---

## 2. Môi trường thực thi, Phiên bản thư viện và Seed

- **Hệ điều hành & CPU:** Windows (`AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD`), chạy hoàn toàn trên CPU.
- **Python:** `3.11.9`
- **Seed cố định:** `np.random.seed(42)` (áp dụng cho sinh dữ liệu tổng hợp và toàn bộ batch thực nghiệm).
- **Phiên bản thư viện (xem thêm `requirements.txt`):**
  - `numpy==2.3.2`
  - `scipy==1.16.1`
  - `opencv-python==5.0.0.93` (chỉ dùng cho I/O ảnh, chuyển màu/resize tiện ích, warp phối cảnh và nhánh đối chứng `src/baseline/baseline.py`)
  - `scikit-image==0.26.0` (chỉ dùng cho Sauvola đối chứng trong `src/baseline/baseline.py` và script sweep)
  - `shapely==2.2.0`
  - `matplotlib==3.10.5`
  - `pillow==11.3.0`
  - `python-docx==1.2.0`
  - `python-pptx==1.0.2`
  - `lxml==6.1.3`
  - `pytest==9.2.0`

---

## 3. Kiến trúc hệ thống & Kết quả thực nghiệm chính

Toàn bộ thuật toán lõi trong `src/core/` được tự cài đặt từ mảng NumPy (vượt qua kiểm tra tĩnh AST `tests/verify_ast.py` với 0 vi phạm), tuyệt đối không gọi `cv2.Canny`, `cv2.HoughLines`, `cv2.threshold` hay `skimage.filters.threshold_*`:

```
Ảnh BGR thô (H x W x 3)
  ├─► [1] Resize (max_dim = 640 px) -> Canny tự cài (Gaussian -> Sobel -> NMS -> Hysteresis 8-conn)
  ├─► [2] Hough Line Transform tự cài (Accumulator np.bincount -> Peak NMS tuần hoàn)
  ├─► [3] Quadrilateral Selection (Giao 4 đường -> Lồi -> Edge Support + Area -> Early Exit nếu rejected)
  ├─► [4] Perspective Warping (Homography 3x3 + Nội suy song tuyến tính ở độ phân giải gốc)
  └─► [5] Thresholding (Otsu toàn cục & Sauvola cục bộ dùng Integral Images)
```

### Tóm tắt kết quả định lượng (50 ảnh dương có GT + 10 ảnh âm + 12 ảnh synthetic, `seed = 42`)

| Phiên bản / Tập dữ liệu | Accepted (Dương) | Mean Polygon IoU | Success@0.8 | Corner Error | FPR (Ảnh âm) | Runtime Acc / Rej (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom Pipeline (Development)** | `23/30` | `0.5222` | `9/30 (30.0%)` | `0.0825` (`23/30`) | `3/6` | `251.8 / 218.7` |
| **Custom Pipeline (Test)** | `17/20` | `0.4668` | `3/20 (15.0%)` | `0.1163` (`17/20`) | `1/4` | `219.2 / 169.5` |
| **Baseline OpenCV (Development)** | `13/30` | `0.3522` | `7/30 (23.3%)` | `0.0446` (`13/30`) | `1/6` | `166.7 / 43.8` |
| **Baseline OpenCV (Test)** | `7/20` | `0.2050` | `2/20 (10.0%)` | `0.1102` (`7/20`) | `0/4` | `113.6 / 44.3` |
| **Ablation 1: Bỏ Gaussian (Dev)** | `23/30` | `0.5132` | `7/30 (23.3%)` | `0.0833` (`23/30`) | `2/6` | `310.4 / 198.8` |
| **Ablation 2: Bỏ Hough NMS (Dev)** | `16/30` | `0.3643` | `9/30 (30.0%)` | `0.0877` (`16/30`) | `3/6` | `243.6 / 157.5` |
| **Ablation 3: Bỏ Edge Support (Dev)** | `27/30` | `0.5965` | `11/30 (36.7%)` | `0.0950` (`27/30`) | `5/6` | `240.3 / 156.6` |

- **12 ảnh tổng hợp (`outputs/results_synthetic.csv`):**
  - **Development:** Otsu Ink F1 = `0.7321`, Sauvola Ink F1 = `0.9999`.
  - **Test:** Otsu Ink F1 = `0.5979`, Sauvola Ink F1 = `0.9999`.

---

## 4. Cấu trúc thư mục (đường dẫn tương đối)

```text
├── src/
│   ├── core/                              # 5 module thuật toán lõi tự cài đặt
│   │   ├── canny.py                       # Gaussian, Sobel, NMS, ngưỡng kép, BFS 8-lân cận
│   │   ├── hough.py                       # Hough accumulator (np.bincount) & peak suppression
│   │   ├── quadrilateral.py               # Giao đường thẳng, kiểm tra lồi, chấm điểm edge support
│   │   ├── warping.py                     # Giải Homography DLT, warp & bilinear_sample_gray
│   │   └── thresholding.py                # Otsu histogram & Sauvola Integral Images
│   └── baseline/
│       └── baseline.py                    # Pipeline đối chứng OpenCV/skimage
├── tests/
│   ├── verify_ast.py                      # Kiểm tra AST cấm gọi hàm thư viện gian lận
│   ├── test_canny.py                      # Kiểm thử biên Canny (ảnh hằng, step edge, hysteresis, NMS)
│   ├── test_hough.py                      # Kiểm thử Hough (edge map rỗng, đường ngang/đứng)
│   ├── test_quadrilateral.py              # Kiểm thử tứ giác (hợp lệ, suy biến, song song)
│   ├── test_warping.py                    # Kiểm thử Homography (identity, phối cảnh, bilinear tính tay)
│   └── test_thresholding.py               # Kiểm thử ngưỡng (ảnh hằng, mask tổng hợp chuẩn)
├── data/
│   ├── development/                       # 36 ảnh thực tế tập Development (30 dương + 6 âm)
│   ├── test/                              # 24 ảnh thực tế tập Test (20 dương + 4 âm)
│   ├── synthetic/                         # 12 ảnh chữ tổng hợp + 12 ground-truth mask + manifest_synthetic.csv
│   ├── manifest.csv                       # Danh sách tổng 60 ảnh thực tế
│   ├── manifest_development.csv           # Split Development (36 ảnh)
│   ├── manifest_test.csv                  # Split Test (24 ảnh)
│   └── labels.json                        # Tọa độ 4 góc ground-truth cho 50 ảnh dương (30 Dev + 20 Test)
├── scripts/
│   ├── generate_synthetic.py              # Sinh 12 ảnh synthetic + GT mask (seed=42)
│   ├── run_all_experiments.py             # Sweep Dev, đóng băng cấu hình, chạy Dev/Test/Baseline/Ablation
│   ├── generate_figures.py                # Vẽ biểu đồ và ảnh minh họa 6 trường hợp + stages
│   ├── generate_rich_visuals.py           # Vẽ sơ đồ kiến trúc, lưới tham số, trực quan ablation/crop
│   ├── prepare_gt_grid.py                 # Hỗ trợ vẽ lưới tọa độ phục vụ gắn nhãn GT
│   ├── build_labels_json.py               # Tổng hợp nhãn tọa độ vào data/labels.json
│   ├── render_gt_verify.py                # Xuất ảnh kiểm tra trực quan ground-truth
│   ├── build_report.py                    # Biên dịch báo cáo .md và .docx chuẩn OOXML
│   ├── build_slides.py                    # Tạo bộ trình chiếu PowerPoint 12 slides (.pptx) & xuất PDF
│   └── audit_report.py                    # Kiểm tra chất lượng, traceability CSV và xuất PDF từ Word COM
├── outputs/
│   ├── frozen_config.json                 # Cấu hình đóng băng trên tập Development
│   ├── baseline_config.json               # Cấu hình tường minh của nhánh đối chứng OpenCV/skimage
│   ├── ablation_configs.json              # Cấu hình 3 biến thể Ablation
│   ├── results_development.csv            # Kết quả từng ảnh trên Development (36 ảnh)
│   ├── results_test.csv                   # Kết quả từng ảnh trên Test (24 ảnh)
│   ├── results_baseline_dev.csv           # Kết quả Baseline trên Development
│   ├── results_baseline_test.csv          # Kết quả Baseline trên Test
│   ├── ablation_no_gaussian.csv           # Ablation 1: Bỏ Gaussian smoothing
│   ├── ablation_no_hough_suppression.csv  # Ablation 2: Bỏ Hough peak suppression
│   ├── ablation_no_edge_support.csv       # Ablation 3: Bỏ edge support trong chấm điểm tứ giác
│   ├── results_synthetic.csv              # Kết quả Custom Otsu/Sauvola trên 12 ảnh tổng hợp
│   ├── results_baseline_synthetic.csv     # Kết quả Baseline Otsu/Sauvola trên 12 ảnh tổng hợp
│   ├── sweep_canny_dev.csv                # Kết quả sweep sigma và ngưỡng Canny trên Dev
│   ├── sweep_hough_quad_dev.csv           # Kết quả sweep ngưỡng Hough và scoring mode trên Dev
│   ├── sweep_thresholding_dev.csv         # Kết quả sweep Otsu và lưới (window, k) Sauvola trên Dev
│   ├── figures/                           # Toàn bộ 23 hình minh họa và metadata báo cáo
│   ├── intermediates/                     # Ảnh trung gian từng bước cho 8 mẫu tiêu biểu
│   ├── Bao_cao_Giua_Ky_Document_Scanner.md
│   ├── Bao_cao_Giua_Ky_Document_Scanner.docx
│   ├── Bao_cao_Giua_Ky_Document_Scanner.pdf
│   ├── Slides_Giua_Ky_Document_Scanner.pptx
│   └── Slides_Giua_Ky_Document_Scanner.pdf
├── run_pipeline.py                        # CLI chạy batch pipeline và xuất CSV / ảnh trung gian
├── requirements.txt                       # Danh sách phiên bản thư viện chuẩn
└── README.md
```

---

## 5. Hướng dẫn cài đặt và chạy tái lập

Mọi lệnh dưới đây chạy trực tiếp từ thư mục gốc của repository bằng đường dẫn tương đối:

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Kiểm tra AST (hàm cấm) và chạy 17 Unit Tests
```bash
python tests/verify_ast.py
python -m pytest tests/ -v
```

### Chạy lại toàn bộ thực nghiệm, tạo hình và biên soạn báo cáo
```bash
# 1. Sinh lại 12 ảnh tổng hợp + mask chuẩn (seed = 42)
python scripts/generate_synthetic.py

# 2. Chạy sweep trên Dev, đóng băng cấu hình và xuất toàn bộ CSV + intermediates + configs
python scripts/run_all_experiments.py

# 3. Sinh toàn bộ 23 hình trực quan và biểu đồ phân tích
python scripts/generate_figures.py
python scripts/generate_rich_visuals.py

# 4. Tạo báo cáo Markdown + Word (.docx), kiểm duyệt + xuất PDF và sinh Slides trình chiếu
python scripts/build_report.py
python scripts/audit_report.py
python scripts/build_slides.py --export-pdf
```

### Chạy riêng CLI pipeline trên một tập bất kỳ
```bash
python run_pipeline.py --manifest data/manifest_development.csv --images-dir data/development --output-csv outputs/results_development.csv --config outputs/frozen_config.json --labels-json data/labels.json --seed 42 --save-intermediates outputs/intermediates
```
