"""Generate high-resolution (300 DPI) publication-quality system architecture flowchart
for Document Scanner pipeline using Matplotlib, formatted for Word document embedding.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

os.makedirs("outputs/figures", exist_ok=True)

def draw_rounded_rect(ax, x, y, w, h, radius=0.03, facecolor='#FFFFFF', edgecolor='#1F497D', linewidth=1.5, zorder=2):
    """Draw a rectangle with rounded corners."""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder
    )
    ax.add_patch(box)
    return box

def generate_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(16, 8.5), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Background canvas
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Main Title Header
    ax.text(8.0, 8.1, "KIẾN TRÚC TỔNG THỂ HỆ THỐNG DOCUMENT SCANNER PIPELINE", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#0F172A', fontfamily='sans-serif')
    ax.text(8.0, 7.7, "Pipeline 5 giai đoạn xử lý độc lập từ ảnh chụp thiết bị di động đến ảnh nhị phân số hóa", 
            ha='center', va='center', fontsize=11, fontstyle='italic', color='#475569', fontfamily='sans-serif')

    # 1. INPUT CARD (Left)
    draw_rounded_rect(ax, 0.4, 4.4, 2.0, 2.4, radius=0.08, facecolor='#EFF6FF', edgecolor='#3B82F6', linewidth=2)
    draw_rounded_rect(ax, 0.4, 6.2, 2.0, 0.6, radius=0.08, facecolor='#3B82F6', edgecolor='#3B82F6')
    ax.text(1.4, 6.5, "ẢNH ĐẦU VÀO", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#FFFFFF')
    ax.text(1.4, 5.7, "Ảnh chụp di động\n(Camera RGB)\n(H x W x 3, uint8)", ha='center', va='center', fontsize=9.5, color='#1E293B')
    ax.text(1.4, 4.8, "• 5 điều kiện chụp\n• Góc xiên / Bóng đổ\n• Nền nhiễu / Âm tính", ha='center', va='center', fontsize=8.5, color='#64748B')

    # Arrow 0 -> 1
    ax.annotate('', xy=(2.7, 5.6), xytext=(2.4, 5.6),
                arrowprops=dict(facecolor='#3B82F6', edgecolor='#3B82F6', width=2.5, headwidth=8, headlength=8))
    ax.text(2.55, 5.85, "to_gray", ha='center', va='center', fontsize=8, color='#3B82F6', fontweight='bold')

    # 2. STAGE 1: CANNY
    draw_rounded_rect(ax, 2.7, 3.8, 2.4, 3.2, radius=0.08, facecolor='#FFFFFF', edgecolor='#0284C7', linewidth=1.8)
    draw_rounded_rect(ax, 2.7, 6.4, 2.4, 0.6, radius=0.08, facecolor='#0284C7', edgecolor='#0284C7')
    ax.text(3.9, 6.7, "1. CANNY EDGE", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#FFFFFF')
    ax.text(3.9, 6.15, "src/core/canny.py", ha='center', va='center', fontsize=8.5, color='#0369A1', fontstyle='italic')
    
    t1 = ("• Gaussian 1D (σ=1.2)\n"
          "• Sobel 3x3: G_x, G_y\n"
          "  Magnitude & Hướng θ\n"
          "• Non-Max Suppression\n"
          "  (Lượng tử hóa 4 góc)\n"
          "• Ngưỡng kép (40, 120)\n"
          "• Hysteresis 8-liên thông")
    ax.text(2.85, 4.9, t1, ha='left', va='center', fontsize=8.5, color='#334155', linespacing=1.2)

    # Arrow 1 -> 2
    ax.annotate('', xy=(5.4, 5.6), xytext=(5.1, 5.6),
                arrowprops=dict(facecolor='#0284C7', edgecolor='#0284C7', width=2.5, headwidth=8, headlength=8))
    ax.text(5.25, 5.9, "Edge Map\n(H x W)", ha='center', va='center', fontsize=7.5, color='#0284C7', fontweight='bold')

    # 3. STAGE 2: HOUGH
    draw_rounded_rect(ax, 5.4, 3.8, 2.4, 3.2, radius=0.08, facecolor='#FFFFFF', edgecolor='#0D9488', linewidth=1.8)
    draw_rounded_rect(ax, 5.4, 6.4, 2.4, 0.6, radius=0.08, facecolor='#0D9488', edgecolor='#0D9488')
    ax.text(6.6, 6.7, "2. HOUGH TRANSFORM", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#FFFFFF')
    ax.text(6.6, 6.15, "src/core/hough.py", ha='center', va='center', fontsize=8.5, color='#0F766E', fontstyle='italic')
    
    t2 = ("• Tọa độ cực (ρ, θ)\n"
          "  ρ = x·cosθ + y·sinθ\n"
          "  Δρ=1 px, Δθ=1°\n"
          "• Tích lũy np.bincount\n"
          "• Max filter 5x5 (T=50)\n"
          "• Triệt tiêu đường trùng\n"
          "  Δρ<20 px, Δθ<10°")
    ax.text(5.55, 4.9, t2, ha='left', va='center', fontsize=8.5, color='#334155', linespacing=1.2)

    # Arrow 2 -> 3
    ax.annotate('', xy=(8.1, 5.6), xytext=(7.8, 5.6),
                arrowprops=dict(facecolor='#0D9488', edgecolor='#0D9488', width=2.5, headwidth=8, headlength=8))
    ax.text(7.95, 5.9, "K đường\n{(ρ, θ)}", ha='center', va='center', fontsize=7.5, color='#0D9488', fontweight='bold')

    # 4. STAGE 3: QUADRILATERAL
    draw_rounded_rect(ax, 8.1, 3.8, 2.6, 3.2, radius=0.08, facecolor='#FFFFFF', edgecolor='#D97706', linewidth=1.8)
    draw_rounded_rect(ax, 8.1, 6.4, 2.6, 0.6, radius=0.08, facecolor='#D97706', edgecolor='#D97706')
    ax.text(9.4, 6.7, "3. QUAD SELECTION", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#FFFFFF')
    ax.text(9.4, 6.15, "src/core/quadrilateral.py", ha='center', va='center', fontsize=8.5, color='#B45309', fontstyle='italic')
    
    t3 = ("• Tổ hợp 4 đường Hough\n"
          "• Giải giao điểm solve()\n"
          "• Kiểm tra lồi (Cross-prod)\n"
          "• Sắp xếp 4 góc CW\n"
          "• Chấm điểm ứng viên:\n"
          "  Score = 0.5·Supp + 0.5·Area\n"
          "• Lọc diện tích [0.05, 0.98]")
    ax.text(8.25, 4.9, t3, ha='left', va='center', fontsize=8.5, color='#334155', linespacing=1.2)

    # REJECTION BRANCH (Downwards from Stage 3)
    ax.annotate('', xy=(9.4, 2.4), xytext=(9.4, 3.8),
                arrowprops=dict(facecolor='#EF4444', edgecolor='#EF4444', width=2.5, headwidth=8, headlength=8))
    ax.text(9.8, 3.1, "Không hợp lệ / Âm tính\n(No valid quad)", ha='left', va='center', fontsize=8, color='#DC2626', fontweight='bold')
    
    draw_rounded_rect(ax, 8.2, 1.2, 2.4, 1.2, radius=0.08, facecolor='#FEF2F2', edgecolor='#EF4444', linewidth=1.8)
    ax.text(9.4, 2.0, "TRẢ VỀ REJECTED", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#DC2626')
    ax.text(9.4, 1.5, "• Ghi nhận status='rejected'\n• Xuất lý do từ chối (Reason)", ha='center', va='center', fontsize=8, color='#991B1B')

    # Arrow 3 -> 4 (Valid Acceptance)
    ax.annotate('', xy=(11.0, 5.6), xytext=(10.7, 5.6),
                arrowprops=dict(facecolor='#16A34A', edgecolor='#16A34A', width=2.5, headwidth=8, headlength=8))
    ax.text(10.85, 5.9, "4 Đỉnh lồi\n(4x2)", ha='center', va='center', fontsize=7.5, color='#16A34A', fontweight='bold')

    # 5. STAGE 4: WARPING
    draw_rounded_rect(ax, 11.0, 3.8, 2.3, 3.2, radius=0.08, facecolor='#FFFFFF', edgecolor='#16A34A', linewidth=1.8)
    draw_rounded_rect(ax, 11.0, 6.4, 2.3, 0.6, radius=0.08, facecolor='#16A34A', edgecolor='#16A34A')
    ax.text(12.15, 6.7, "4. WARPING", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#FFFFFF')
    ax.text(12.15, 6.15, "src/core/warping.py", ha='center', va='center', fontsize=8.5, color='#15803D', fontstyle='italic')
    
    t4 = ("• Tính kích thước phẳng:\n"
          "  W_max, H_max đối diện\n"
          "• Ma trận phối cảnh H\n"
          "  getPerspectiveTransform\n"
          "• Nắn thẳng góc nhìn\n"
          "  warpPerspective\n"
          "• bilinear_sample_gray\n"
          "  kiểm chứng nội suy")
    ax.text(11.15, 4.9, t4, ha='left', va='center', fontsize=8.5, color='#334155', linespacing=1.2)

    # Arrow 4 -> 5
    ax.annotate('', xy=(13.6, 5.6), xytext=(13.3, 5.6),
                arrowprops=dict(facecolor='#16A34A', edgecolor='#16A34A', width=2.5, headwidth=8, headlength=8))
    ax.text(13.45, 5.9, "Ảnh phẳng\n(H' x W')", ha='center', va='center', fontsize=7.5, color='#16A34A', fontweight='bold')

    # 6. STAGE 5: THRESHOLDING
    draw_rounded_rect(ax, 13.6, 3.8, 2.1, 3.2, radius=0.08, facecolor='#FFFFFF', edgecolor='#7C3AED', linewidth=1.8)
    draw_rounded_rect(ax, 13.6, 6.4, 2.1, 0.6, radius=0.08, facecolor='#7C3AED', edgecolor='#7C3AED')
    ax.text(14.65, 6.7, "5. NHỊ PHÂN HÓA", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#FFFFFF')
    ax.text(14.65, 6.15, "src/core/thresholding.py", ha='center', va='center', fontsize=8.5, color='#6D28D9', fontstyle='italic')
    
    t5 = ("• Otsu toàn cục:\n"
          "  Tối đa hóa phương sai\n"
          "  giữa 2 lớp từ Hist\n"
          "• Sauvola cục bộ:\n"
          "  T = m·[1 + k(s/R - 1)]\n"
          "  w=25, k=0.2, R=128\n"
          "• uniform_filter m, s\n"
          "• Khử bóng, bóc tách chữ")
    ax.text(13.75, 4.9, t5, ha='left', va='center', fontsize=8.5, color='#334155', linespacing=1.2)

    # 7. FINAL OUTPUT CARD (Bottom Right)
    ax.annotate('', xy=(14.65, 2.7), xytext=(14.65, 3.8),
                arrowprops=dict(facecolor='#7C3AED', edgecolor='#7C3AED', width=2.5, headwidth=8, headlength=8))
    ax.text(14.75, 3.25, "Binary Mask\n(0 / 255)", ha='left', va='center', fontsize=8, color='#7C3AED', fontweight='bold')

    draw_rounded_rect(ax, 13.5, 1.2, 2.3, 1.5, radius=0.08, facecolor='#F5F3FF', edgecolor='#7C3AED', linewidth=2)
    draw_rounded_rect(ax, 13.5, 2.3, 2.3, 0.4, radius=0.08, facecolor='#7C3AED', edgecolor='#7C3AED')
    ax.text(14.65, 2.5, "KẾT QUẢ ĐẦU RA", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#FFFFFF')
    ax.text(14.65, 1.9, "• Trang scan nhìn thẳng\n• Ảnh nhị phân chữ rõ nét\n• Tách nền sạch bóng râm", ha='center', va='center', fontsize=8.5, color='#4C1D95')
    ax.text(14.65, 1.4, "status='accepted'", ha='center', va='center', fontsize=8, color='#15803D', fontweight='bold')

    # BOTTOM FOOTER NOTE: CPU RUNTIME & REPRODUCIBILITY
    draw_rounded_rect(ax, 0.4, 0.3, 15.2, 0.65, radius=0.04, facecolor='#F1F5F9', edgecolor='#CBD5E1', linewidth=1)
    footer_text = (
        "• Ràng buộc kỹ thuật đồ án: 100% tự cài đặt các module lõi từ NumPy/SciPy (không gọi cv2.Canny, cv2.HoughLines, cv2.threshold).  "
        "• Tốc độ CPU thuần: ~728.8 - 1098.5 ms/ảnh.  • Trích xuất tự động ra CSV."
    )
    ax.text(8.0, 0.62, footer_text, ha='center', va='center', fontsize=8.5, color='#475569', fontfamily='sans-serif', fontweight='bold')

    plt.tight_layout()
    out_path = "outputs/figures/pipeline_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Generated {out_path} successfully!")

if __name__ == "__main__":
    generate_pipeline_diagram()
