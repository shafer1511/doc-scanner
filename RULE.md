# PROJECT RULES: DOCUMENT SCANNER & ANTI-AI WRITING PROTOCOL

## 1. KIẾN TRÚC ĐIỀU PHỐI (ORCHESTRATOR -> WORKER)
- **Orchestrator (Gemini 3.8 Flash):** Chỉ lập kế hoạch, chia nhỏ task, viết test case và nghiệm thu. KHÔNG tự ý viết toàn bộ code pipeline thay Worker.
- **Worker (DeepSeek V4.1 qua OMP):** Chỉ thực thi đúng phạm vi hàm/file được giao. Không tự ý sửa cấu trúc thư mục hoặc thêm dependency ngoài danh sách cho phép.
- **Giao tiếp qua CLI (`omp`):** Orchestrator bắt buộc gọi `omp` ở chế độ non-interactive (truyền prompt qua flag dòng lệnh hoặc file tạm `.tmp_task.md`), tuyệt đối không mở phiên TUI tương tác gây treo terminal. Chờ lệnh thoát với exit code `0` rồi mới chạy bước kiểm duyệt.
- **Quy tắc giao việc (Atomic Task):** Mỗi prompt giao cho Worker bắt buộc có đủ 4 mục:
  1. `Target File & Function`: Tên file, tên hàm, input/output shape và dtype (ví dụ: `img: np.ndarray (H, W), uint8`).
  2. `Algorithmic Constraints`: Công thức/thuật toán bắt buộc tự cài.
  3. `Forbidden Calls`: Danh sách hàm thư viện bị cấm trong task này.
  4. `Acceptance Test`: Lệnh chạy kiểm thử cụ thể (ví dụ: test trên ảnh hằng, step edge, edge map rỗng).

---

## 2. RÀNG BUỘC KỸ THUẬT ĐỒ ÁN (BẮT BUỘC TUÂN THỦ)
- **Môi trường:** Python, NumPy và các primitive cơ bản; toàn bộ pipeline phải chạy tái lập (reproducible) trên CPU với seed cố định.
- **Cấm gian lận thư viện ở Pipeline chính (`src/core/`):**
  - **Canny:** Phải tự cài gradient magnitude/direction, Non-Maximum Suppression (NMS), phân loại ngưỡng kép (strong/weak) và hysteresis liên thông 8. Cấm gọi `cv2.Canny` hoặc `skimage.feature.canny`.
  - **Hough:** Phải tự cài standard accumulator, chọn peaks, suppression các đường gần trùng và quy đổi peak thành tham số đường thẳng. Cấm gọi `cv2.HoughLines` / `HoughLinesP`.
  - **Quadrilateral:** Phải tự ghép 4 đường, giải giao điểm (được dùng `np.linalg.solve`), kiểm tra tứ giác lồi, sắp xếp 4 góc theo chu vi, tính score kết hợp diện tích + mức hỗ trợ biên (edge support) và trả về trạng thái `rejected` kèm lý do nếu không hợp lệ.
  - **Homography & Bilinear:** Được dùng `cv2.getPerspectiveTransform` và `cv2.warpPerspective` để warp ảnh, nhưng BẮT BUỘC tự viết hàm `bilinear_sample_gray` nhỏ để kiểm chứng giá trị tại điểm nguyên, điểm phân số và biên ảnh.
  - **Otsu & Sauvola:** Phải tự cài hàm tính ngưỡng Otsu (tối đa hóa phương sai giữa các lớp từ histogram) và công thức threshold map của Sauvola từ local mean/std.
- **Nhánh Baseline (`src/baseline/`):** Chỉ tại thư mục này mới được phép dùng các hàm có sẵn của OpenCV/scikit-image để làm mốc đối chiếu.
- **Phân chia dữ liệu & Tham số (No Data Leakage):**
  - Toàn bộ tham số phải được dò và chốt trên tập **Development (36 ảnh thực + 6 ảnh synthetic)**.
  - Đóng băng cấu hình trước khi chạy trên tập **Test (24 ảnh thực + 6 ảnh synthetic)**. Tuyệt đối không tinh chỉnh tham số dựa trên kết quả tập Test.
- **Quy tắc tính Metrics:**
  - *Corner Error:* Sắp xếp 4 góc theo chu vi và lấy khớp quay cho sai số nhỏ nhất so với ground-truth; chuẩn hóa chia cho độ dài đường chéo ảnh.
  - *Ảnh dương bị reject:* Tính **IoU = 0**; tỷ lệ phát hiện thành công tại **IoU > 0.8** tính trên toàn bộ số ảnh dương.
  - *CPU Runtime:* Chỉ đo từ mảng ảnh đã nạp đến đầu ra, **loại trừ hoàn toàn thời gian đọc/ghi file (I/O)** và vẽ hình (`matplotlib`). Thống kê riêng cho nhóm `accepted` và `rejected`.
- **Ba bài kiểm tra Ablation bắt buộc:** Thiết kế cờ điều khiển để thực nghiệm đúng 3 ca: (1) Bỏ Gaussian smoothing, (2) Bỏ suppression đường Hough trùng, (3) Bỏ edge-support trong hàm tính điểm tứ giác. **Bắt buộc giữ nguyên toàn bộ tham số còn lại và giới hạn số ứng viên (`top_k_lines`)** giữa các lần chạy.
- **Định dạng xuất kết quả (`outputs/results_<split>.csv`):** Mọi lần chạy batch trên thư mục ảnh phải xuất CSV chứa đủ toàn bộ ảnh (bao gồm cả các ca `rejected`) với các cột tối thiểu: `image_id, split, condition, status, reject_reason, iou, corner_error_norm, precision, recall, f1, runtime_ms, resolution`.
- **Quy tắc viết Code:**
  - Tách bạch từng bước xử lý thành các hàm thuần (pure functions) để xuất được ảnh trung gian (edge map, accumulator, peaks, tứ giác ứng viên, histogram, threshold map).
  - Cấm comment hiển nhiên. Chỉ viết docstring 1–3 dòng mô tả `Args`, `Returns` (kèm shape/dtype mảng) và công thức toán học nếu cần.

---

## 3. LUẬT CHỐNG VĂN PHONG AI (ÁP DỤNG CHO BÁO CÁO PDF, SLIDES, README)

### 3.1. Danh sách đen (Cấm tuyệt đối)
- **Cấm từ/cụm từ sáo rỗng:** *"trong bối cảnh"*, *"đóng vai trò then chốt/quan trọng"*, *"không chỉ... mà còn..."*, *"một cách hiệu quả/toàn diện/mượt mà"*, *"tối ưu hóa trải nghiệm"*, *"nhìn chung"*, *"tóm lại"*, *"kết luận lại"*, *"có thể thấy rằng"*, *"đáng chú ý là"*.
- **Cấm cấu trúc liệt kê kiểu AI:** Không dùng mẫu `- **Tiêu đề in đậm:** Một câu giải thích dài dòng vô thưởng vô phạt`. Thay bằng câu văn trực diện hoặc bảng số liệu.
- **Cấm mở đầu/kết thúc kiểu bài văn mẫu:** Đi thẳng vào vấn đề ngay câu đầu tiên của mỗi mục. Không viết câu dẫn nhập chuyển đoạn kiểu *"Sau khi đã tìm hiểu X, chúng ta sẽ tiếp tục khám phá Y"*.

### 3.2. Quy tắc hành văn kỹ thuật (Chuẩn báo cáo khoa học/kỹ thuật)
- **Cấu trúc câu:** Ưu tiên câu đơn hoặc câu ghép ngắn (< 25 từ). Trình bày theo thứ tự: **[Kết quả / Hiện tượng] -> [Số liệu / Bằng chứng] -> [Nguyên nhân kỹ thuật]**.
  - *Sai (Văn AI):* "Phương pháp Sauvola cho thấy khả năng vượt trội trong việc xử lý các tài liệu có bóng đổ phức tạp nhờ vào cơ chế tính ngưỡng cục bộ linh hoạt."
  - *Đúng (Văn kỹ thuật):* "Trên tập ảnh bóng đổ, Sauvola ($w=25, k=0.2$) đạt F1 lớp ink là 0.84, cao hơn 0.31 so với ngưỡng toàn cục Otsu do ngưỡng được điều chỉnh theo độ lệch chuẩn cục bộ từng vùng."
- **Thuật ngữ chuyên ngành:** Giữ nguyên tiếng Anh đối với các thuật ngữ xử lý ảnh chuẩn (*edge map, accumulator, hysteresis, NMS, homography, bilinear resampling, ablation, baseline, IoU, false-positive rate*). Không dịch gượng ép sang tiếng Việt.
- **Phân tích lỗi (6 trường hợp bắt buộc):** Mọi ca phân tích thất bại hoặc ca khó đều phải trình bày theo đúng 4 gạch đầu dòng:
  1. *Đầu vào & Điều kiện chụp:* (ví dụ: Ảnh `doc04_lowcontrast`, biên giấy lệch nền < 15 mức xám).
  2. *Điểm gãy trong pipeline:* (xảy ra ở Canny, Hough hay Quadrilateral).
  3. *Bằng chứng trên ảnh trung gian:* (mô tả cụ thể hình ảnh accumulator/edge map).
  4. *Nguyên nhân & Tham số liên quan:* (giải thích trực tiếp bằng toán học/ngưỡng).

---

## 4. CỔNG KIỂM DUYỆT CỦA ORCHESTRATOR (VERIFICATION GATE)
Trước khi đánh dấu hoàn thành bất kỳ task nào từ DeepSeek V4.1, Orchestrator bắt buộc kiểm tra:
1. **Static Check:** Quét file code mới tạo, từ chối ngay nếu phát hiện gọi hàm thư viện bị cấm ở Mục 2.
2. **Edge-case Unit Test:** Bắt buộc Worker chạy pass các ca kiểm thử biên: ảnh hằng (constant image), step edge, đường thẳng đứng/ngang ($\theta = 0^\circ, 90^\circ$), edge map rỗng (all-zeros), tứ giác suy biến (3 điểm thẳng hàng), ma trận đơn vị (identity homography).
3. **Traceability Check (Kiểm tra nguồn số liệu):** Mọi số liệu trong báo cáo phải được nạp tự động từ file log/CSV sinh ra từ lần chạy thật (`outputs/results_*.csv`), cấm gõ tay số liệu ước đoán.
4. **Style Audit (Với file `.md` / báo cáo):** Quét toàn bộ văn bản theo Mục 3.1. Nếu xuất hiện bất kỳ cụm từ nào trong danh sách đen, trả về cho Worker viết lại đoạn đó trước khi xuất bản.
