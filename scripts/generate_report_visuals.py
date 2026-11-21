"""Generate high-resolution visual comparison figures and dataset overview
for Document Scanner Technical Report, showcasing the actual user document dataset.
"""
import os
import cv2
import numpy as np

os.makedirs("outputs/figures", exist_ok=True)

def create_dataset_overview():
    """Create a 2x3 grid showcasing the 6 main document classes from the user's 60 photos."""
    samples = [
        ("filedata/thesv_01.jpg", "1. Thẻ sinh viên (dev_26)"),
        ("filedata/bus_01.jpg", "2. Vé xe buýt (test_01)"),
        ("filedata/sign_01.jpg", "3. Giấy tờ dấu đỏ (dev_21)"),
        ("filedata/Vat_1/1.jpg", "4. Checklist Bosch (test_11)"),
        ("filedata/book_01.jpg", "5. Sổ tay bìa Leaf (dev_16)"),
        ("filedata/negative_01.jpg", "6. Ảnh âm tính - Gạch sàn (dev_31)")
    ]
    
    target_h, target_w = 400, 300
    images = []
    
    for path, title in samples:
        if os.path.exists(path):
            img = cv2.imread(path)
            # Resize preserving aspect ratio with padding
            h, w = img.shape[:2]
            scale = min(target_w / w, target_h / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            
            canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
            y_off = (target_h - nh) // 2
            x_off = (target_w - nw) // 2
            canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
            
            # Add text label banner at bottom
            cv2.rectangle(canvas, (0, target_h - 35), (target_w, target_h), (30, 30, 30), -1)
            cv2.putText(canvas, title, (8, target_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            images.append(canvas)
        else:
            blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            images.append(blank)
            
    row1 = np.hstack(images[:3])
    row2 = np.hstack(images[3:6])
    grid = np.vstack([row1, row2])
    
    out_path = "outputs/figures/dataset_overview.png"
    cv2.imwrite(out_path, grid)
    print(f"Created {out_path} ({grid.shape[1]}x{grid.shape[0]})")

def create_document_crops():
    """Extract side-by-side zoomed crops of Otsu vs Sauvola on user's real documents."""
    # 1. Thẻ sinh viên (dev_26) - text on student ID
    warped_thesv = cv2.imread("outputs/intermediates_dev/dev_26/05_warped.png", cv2.IMREAD_GRAYSCALE)
    otsu_thesv = cv2.imread("outputs/intermediates_dev/dev_26/06_binary_otsu.png", cv2.IMREAD_GRAYSCALE)
    sauvola_thesv = cv2.imread("outputs/intermediates_dev/dev_26/07_binary_sauvola.png", cv2.IMREAD_GRAYSCALE)
    
    if warped_thesv is not None:
        h, w = warped_thesv.shape
        # Crop around center text area
        cy, cx = int(h * 0.45), int(w * 0.35)
        ch, cw = int(h * 0.18), int(w * 0.45)
        crop_g = warped_thesv[cy:cy+ch, cx:cx+cw]
        crop_o = otsu_thesv[cy:cy+ch, cx:cx+cw]
        crop_s = sauvola_thesv[cy:cy+ch, cx:cx+cw]
        cv2.imwrite("outputs/figures/thesv_crop_gray.png", crop_g)
        cv2.imwrite("outputs/figures/thesv_crop_otsu.png", crop_o)
        cv2.imwrite("outputs/figures/thesv_crop_sauvola.png", crop_s)
        print("Created thesv crops")

    # 2. Vé xe buýt (test_01) - dot matrix printed text
    warped_bus = cv2.imread("outputs/intermediates_test/test_01/05_warped.png", cv2.IMREAD_GRAYSCALE)
    otsu_bus = cv2.imread("outputs/intermediates_test/test_01/06_binary_otsu.png", cv2.IMREAD_GRAYSCALE)
    sauvola_bus = cv2.imread("outputs/intermediates_test/test_01/07_binary_sauvola.png", cv2.IMREAD_GRAYSCALE)
    
    if warped_bus is not None:
        h, w = warped_bus.shape
        cy, cx = int(h * 0.35), int(w * 0.15)
        ch, cw = int(h * 0.20), int(w * 0.70)
        crop_g = warped_bus[cy:cy+ch, cx:cx+cw]
        crop_o = otsu_bus[cy:cy+ch, cx:cx+cw]
        crop_s = sauvola_bus[cy:cy+ch, cx:cx+cw]
        cv2.imwrite("outputs/figures/bus_crop_gray.png", crop_g)
        cv2.imwrite("outputs/figures/bus_crop_otsu.png", crop_o)
        cv2.imwrite("outputs/figures/bus_crop_sauvola.png", crop_s)
        print("Created bus crops")

    # 3. Giấy tờ dấu đỏ (dev_21) - red seal area
    warped_sign = cv2.imread("outputs/intermediates_dev/dev_21/05_warped.png", cv2.IMREAD_GRAYSCALE)
    otsu_sign = cv2.imread("outputs/intermediates_dev/dev_21/06_binary_otsu.png", cv2.IMREAD_GRAYSCALE)
    sauvola_sign = cv2.imread("outputs/intermediates_dev/dev_21/07_binary_sauvola.png", cv2.IMREAD_GRAYSCALE)
    
    if warped_sign is not None:
        h, w = warped_sign.shape
        cy, cx = int(h * 0.60), int(w * 0.45)
        ch, cw = int(h * 0.22), int(w * 0.40)
        crop_g = warped_sign[cy:cy+ch, cx:cx+cw]
        crop_o = otsu_sign[cy:cy+ch, cx:cx+cw]
        crop_s = sauvola_sign[cy:cy+ch, cx:cx+cw]
        cv2.imwrite("outputs/figures/sign_crop_gray.png", crop_g)
        cv2.imwrite("outputs/figures/sign_crop_otsu.png", crop_o)
        cv2.imwrite("outputs/figures/sign_crop_sauvola.png", crop_s)
        print("Created sign crops")

    # 4. Shadow crop on real photo dev_25 (Giấy tờ dấu đỏ có bóng đổ)
    warped_shadow = cv2.imread("outputs/intermediates_dev/dev_25/05_warped.png", cv2.IMREAD_GRAYSCALE)
    otsu_shadow = cv2.imread("outputs/intermediates_dev/dev_25/06_binary_otsu.png", cv2.IMREAD_GRAYSCALE)
    sauvola_shadow = cv2.imread("outputs/intermediates_dev/dev_25/07_binary_sauvola.png", cv2.IMREAD_GRAYSCALE)
    
    if warped_shadow is not None:
        h, w = warped_shadow.shape
        cy, cx = int(h * 0.20), int(w * 0.20)
        ch, cw = int(h * 0.25), int(w * 0.50)
        crop_g = warped_shadow[cy:cy+ch, cx:cx+cw]
        crop_o = otsu_shadow[cy:cy+ch, cx:cx+cw]
        crop_s = sauvola_shadow[cy:cy+ch, cx:cx+cw]
        cv2.imwrite("outputs/figures/shadow_real_crop_gray.png", crop_g)
        cv2.imwrite("outputs/figures/shadow_real_crop_otsu.png", crop_o)
        cv2.imwrite("outputs/figures/shadow_real_crop_sauvola.png", crop_s)
        print("Created real shadow crops")

def create_failure_case_visuals():
    """Generate 6 failure case visualization cards."""
    # Ca 1: dev_31 (negative_floor) - Hough lines on tile grid
    img_floor = cv2.imread("outputs/intermediates_dev/dev_31/04_quadrilateral.png")
    if img_floor is not None:
        cv2.imwrite("outputs/figures/fail_case1_floor.png", img_floor)
        
    # Ca 2: dev_32 (negative_screen) - Laptop bezel
    img_screen = cv2.imread("outputs/intermediates_dev/dev_32/04_quadrilateral.png")
    if img_screen is not None:
        cv2.imwrite("outputs/figures/fail_case2_screen.png", img_screen)
        
    # Ca 3: dev_33 (negative torn paper)
    img_torn = cv2.imread("outputs/intermediates_dev/dev_33/02_canny_edge.png")
    if img_torn is not None:
        cv2.imwrite("outputs/figures/fail_case3_torn.png", img_torn)
        
    # Ca 4: dev_09 (low contrast white paper on light desk)
    img_lowcon = cv2.imread("outputs/intermediates_dev/dev_09/02_canny_edge.png")
    if img_lowcon is not None:
        cv2.imwrite("outputs/figures/fail_case4_lowcon.png", img_lowcon)
        
    # Ca 5: dev_12 (oblique perspective notebook)
    img_oblique = cv2.imread("outputs/intermediates_dev/dev_12/04_quadrilateral.png")
    if img_oblique is not None:
        cv2.imwrite("outputs/figures/fail_case5_oblique.png", img_oblique)
        
    # Ca 6: dev_03 (complex cover pattern)
    img_complex = cv2.imread("outputs/intermediates_dev/dev_03/03_hough_lines.png")
    if img_complex is not None:
        cv2.imwrite("outputs/figures/fail_case6_complex.png", img_complex)
        
    print("Created failure case visual assets")

if __name__ == "__main__":
    create_dataset_overview()
    create_document_crops()
    create_failure_case_visuals()
