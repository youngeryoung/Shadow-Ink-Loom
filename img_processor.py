# =============================================================
# [Shadow-Ink Loom]  
# Author: 烛鵼 Young 
# “For the Shadow-bird to mend the world, 
#     it first needs to see in black and white.”
# =============================================================

import cv2
import numpy as np

# Numba 加速检测
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not found. Dithering might be slower.")

def _dither_core_impl(image_float):
    """Floyd-Steinberg 抖动核心算法"""
    h, w = image_float.shape
    for y in range(h):
        for x in range(w):
            old_pixel = image_float[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            image_float[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x + 1 < w:
                image_float[y, x + 1] += quant_error * 0.4375
            if y + 1 < h:
                if x - 1 >= 0:
                    image_float[y + 1, x - 1] += quant_error * 0.1875
                image_float[y + 1, x] += quant_error * 0.3125
                if x + 1 < w:
                    image_float[y + 1, x + 1] += quant_error * 0.0625
    return image_float

if HAS_NUMBA:
    _dither_core = jit(nopython=True)(_dither_core_impl)
else:
    _dither_core = _dither_core_impl

def process_image(img_array, target_width_px, target_height_px, use_dithering=True, line_thickness=1):
    """
    处理图像的主函数
    Args:
        img_array: OpenCV BGR 图像数组
        target_width_px: 目标宽度(像素)
        target_height_px: 目标高度(像素)
        use_dithering: 是否抖动
        line_thickness: 描边宽度(像素)
    """
    if img_array is None:
        return None

    # 转灰度
    if len(img_array.shape) == 3:
        gray_src = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray_src = img_array

    h_orig, w_orig = gray_src.shape
    
    # 防止除零错误
    if target_width_px <= 0 or target_height_px <= 0:
        return gray_src

    # 1. 计算缩放比例 (用于矢量轮廓映射)
    scale_x = target_width_px / w_orig
    scale_y = target_height_px / h_orig

    # 2. 高质量缩放 (Lanczos)
    gray_resized = cv2.resize(gray_src, (target_width_px, target_height_px), interpolation=cv2.INTER_LANCZOS4)

    # 3. 生成底图
    if use_dithering:
        dither_input = gray_resized.astype(np.float32)
        canvas = _dither_core(dither_input).astype(np.uint8)
    else:
        # 自适应阈值混合模式
        block_size = 11
        # 确保 block_size 不超过图像最小边且为奇数
        min_side = min(target_width_px, target_height_px)
        if block_size >= min_side:
            block_size = min_side if min_side % 2 != 0 else min_side - 1
        if block_size < 3: block_size = 3
            
        canvas = cv2.adaptiveThreshold(
            gray_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, 2
        )

    # 4. 轮廓重绘 (在原图尺寸提取，映射到新尺寸)
    blurred = cv2.GaussianBlur(gray_src, (5, 5), 0)
    median_val = np.median(blurred)
    sigma = 0.33
    canny_low = int(max(0, (1.0 - sigma) * median_val))
    canny_high = int(min(255, (1.0 + sigma) * median_val))
    
    edges = cv2.Canny(blurred, canny_low, canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        for c in contours:
            c_transformed = c.astype(np.float32)
            c_transformed[:, 0, 0] *= scale_x
            c_transformed[:, 0, 1] *= scale_y
            c_final = c_transformed.astype(np.int32)
            # 画黑线 (0)
            cv2.drawContours(canvas, [c_final], -1, 0, thickness=line_thickness)

    return canvas