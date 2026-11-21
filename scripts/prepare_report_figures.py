"""Prepare visual comparison crops and diagnostic figures for the final report."""
import os
import cv2

def prepare_figures():
    os.makedirs("outputs/figures", exist_ok=True)
    
    # 1. dev_01 text crop
    warped_dev01 = cv2.imread("outputs/intermediates_dev/dev_01/05_warped.png")
    otsu_dev01 = cv2.imread("outputs/intermediates_dev/dev_01/06_binary_otsu.png")
    sauvola_dev01 = cv2.imread("outputs/intermediates_dev/dev_01/07_binary_sauvola.png")
    
    if warped_dev01 is not None:
        H, W = warped_dev01.shape[:2]
        # Crop text area on book cover (e.g. title text 'HP Book')
        y0, y1 = int(0.10 * H), int(0.35 * H)
        x0, x1 = int(0.65 * W), int(0.95 * W)
        crop_gray = warped_dev01[y0:y1, x0:x1]
        crop_otsu = otsu_dev01[y0:y1, x0:x1]
        crop_sauv = sauvola_dev01[y0:y1, x0:x1]
        
        cv2.imwrite("outputs/figures/dev01_crop_gray.png", crop_gray)
        cv2.imwrite("outputs/figures/dev01_crop_otsu.png", crop_otsu)
        cv2.imwrite("outputs/figures/dev01_crop_sauvola.png", crop_sauv)
        print("Generated dev_01 crops.")
        
    # 2. syn_dev_05 shadow text crop
    warped_syn05 = cv2.imread("outputs/intermediates_dev/syn_dev_05/05_warped.png")
    otsu_syn05 = cv2.imread("outputs/intermediates_dev/syn_dev_05/06_binary_otsu.png")
    sauv_syn05 = cv2.imread("outputs/intermediates_dev/syn_dev_05/07_binary_sauvola.png")
    
    if warped_syn05 is not None:
        H, W = warped_syn05.shape[:2]
        # Crop across the diagonal shadow boundary
        y0, y1 = int(0.30 * H), int(0.65 * H)
        x0, x1 = int(0.10 * W), int(0.90 * W)
        crop_syn_gray = warped_syn05[y0:y1, x0:x1]
        crop_syn_otsu = otsu_syn05[y0:y1, x0:x1]
        crop_syn_sauv = sauv_syn05[y0:y1, x0:x1]
        
        cv2.imwrite("outputs/figures/shadow_crop_gray.png", crop_syn_gray)
        cv2.imwrite("outputs/figures/shadow_crop_otsu.png", crop_syn_otsu)
        cv2.imwrite("outputs/figures/shadow_crop_sauvola.png", crop_syn_sauv)
        print("Generated syn_dev_05 shadow crops.")

if __name__ == "__main__":
    prepare_figures()
