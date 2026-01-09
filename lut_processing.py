"""
LUT 处理核心模块
包含 LUT 文件解析和图像处理功能
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter
from PySide6.QtCore import QThread, Signal


def parse_cube_lut(file_path):
    """
    解析 .cube 格式的 3D LUT 文件。

    Args:
        file_path (str): .cube 文件路径

    Returns:
        tuple: (lut_table_list, size)
               lut_table_list 为扁平化的浮点列表，size 为 LUT 的维度 (N)
    """
    lut_table = []
    size = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                # 解析尺寸定义
                if line.startswith('LUT_3D_SIZE'):
                    size = int(line.split()[-1])
                    continue
                # 解析数据点 (检查是否以数字或负号开头)
                if line[0].isdigit() or line[0] == '-':
                    parts = line.split()
                    lut_table.extend([float(v) for v in parts])

        if size == 0:
            raise ValueError("未找到 LUT_3D_SIZE 定义")
        if len(lut_table) != size * size * size * 3:
            raise ValueError(f"数据点数量不匹配。预期: {size ** 3 * 3}, 实际: {len(lut_table)}")

        return lut_table, size

    except UnicodeDecodeError:
        # 尝试使用 latin-1 再次读取，防止某些特殊编码文件报错
        with open(file_path, 'r', encoding='latin-1') as f:
            # 简化重复逻辑，实际生产中可封装
            pass
        raise ValueError("文件编码格式不支持")


def apply_ordered_dithering(img_uint8):
    """
    应用有序抖动（Ordered Dithering）消除Color Banding，基于Bayer矩阵。
    
    Args:
        img_uint8: uint8 格式的RGB图像 (H, W, 3)
    
    Returns:
        抖动后的uint8图像
    """
    # 创建 4x4 Bayer 矩阵
    bayer_matrix = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ], dtype=np.float32) / 16.0 - 0.5  # 归一化到 [-0.5, 0.5]
    
    h, w, c = img_uint8.shape
    result = img_uint8.astype(np.float32)
    
    # 应用Bayer矩阵抖动
    for y in range(h):
        for x in range(w):
            threshold = bayer_matrix[y % 4, x % 4]
            result[y, x] += threshold * 32  # 32 是量化步长，调整强度
    
    # 限制到 [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_noise_dithering(img_uint8, seed=None):
    """
    应用噪声抖动消除Color Banding，使用高斯噪声。
    
    Args:
        img_uint8: uint8 格式的RGB图像 (H, W, 3)
        seed: 随机种子，为None则不固定
    
    Returns:
        抖动后的uint8图像
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成高斯噪声
    noise = np.random.normal(0, 1.5, img_uint8.shape).astype(np.float32)
    result = img_uint8.astype(np.float32) + noise
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_floyd_steinberg_dithering(img_uint8):
    """
    应用Floyd-Steinberg抖动（误差扩散）消除Color Banding，高质量但较慢。
    
    Args:
        img_uint8: uint8 格式的RGB图像 (H, W, 3)
    
    Returns:
        抖动后的uint8图像
    """
    # 转为float处理
    img_float = img_uint8.astype(np.float32)
    result = img_float.copy()
    
    h, w, c = img_uint8.shape
    
    # 遍历每个像素进行误差扩散
    for y in range(h):
        for x in range(w):
            old_pixel = result[y, x].copy()
            # 量化到最近的8级
            new_pixel = np.round(old_pixel / 32) * 32
            new_pixel = np.clip(new_pixel, 0, 255)
            result[y, x] = new_pixel
            
            # 计算量化误差
            quant_error = old_pixel - new_pixel
            
            # 扩散误差到相邻像素（Floyd-Steinberg权重）
            if x + 1 < w:
                result[y, x + 1] += quant_error * 7 / 16
            if x - 1 >= 0 and y + 1 < h:
                result[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < h:
                result[y + 1, x] += quant_error * 5 / 16
            if x + 1 < w and y + 1 < h:
                result[y + 1, x + 1] += quant_error * 1 / 16
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_lut_to_image(source_img, lut_table, lut_size, strength=1.0, dithering=None):
    """
    将 3D LUT 应用到图像上。

    Args:
        source_img: OpenCV BGR 格式的图像
        lut_table: LUT 数据表
        lut_size: LUT 维度大小
        strength: LUT 强度 (0.0-1.0)，1.0为完全应用
        dithering: 抖动方法 (None / 'ordered' / 'noise' / 'floyd')

    Returns:
        处理后的 OpenCV BGR 格式图像
    """
    # 1. 颜色空间转换: OpenCV (BGR) -> Pillow (RGB)
    img_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    # 2. 构建 3D LUT 滤镜
    lut_filter = ImageFilter.Color3DLUT(
        size=lut_size,
        table=lut_table,
        channels=3,
        target_mode=None
    )

    # 3. 应用滤镜 (计算密集型步骤)
    processed_pil = pil_image.filter(lut_filter)

    # 4. 转换回 OpenCV 格式: Pillow (RGB) -> OpenCV (BGR)
    processed_np = np.asarray(processed_pil)
    result_bgr = cv2.cvtColor(processed_np, cv2.COLOR_RGB2BGR)
    
    # 5. 根据强度混合原图和处理后的图
    if strength < 1.0:
        result_bgr = cv2.addWeighted(source_img, 1.0 - strength, result_bgr, strength, 0)
    
    # 6. 应用抖动消除Color Banding
    if dithering:
        # 转为RGB便于处理
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        if dithering == 'ordered':
            result_rgb = apply_ordered_dithering(result_rgb)
        elif dithering == 'noise':
            result_rgb = apply_noise_dithering(result_rgb)
        elif dithering == 'floyd':
            result_rgb = apply_floyd_steinberg_dithering(result_rgb)
        
        # 转回BGR
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    return result_bgr


class ImageProcessingThread(QThread):
    """
    后台图像处理线程，防止阻塞 UI 主线程。
    使用 Pillow 的 C 语言底层滤镜进行加速。
    """
    processing_finished = Signal(object)  # 成功信号，携带处理后的 OpenCV 图像
    processing_error = Signal(str)  # 失败信号，携带错误信息

    def __init__(self, source_img, lut_table, lut_size, strength=1.0, dithering=None):
        super().__init__()
        self.source_img = source_img
        self.lut_table = lut_table
        self.lut_size = lut_size
        self.strength = strength
        self.dithering = dithering

    def run(self):
        try:
            result_bgr = apply_lut_to_image(
                self.source_img, 
                self.lut_table, 
                self.lut_size,
                self.strength,
                self.dithering
            )
            self.processing_finished.emit(result_bgr)

        except Exception as e:
            self.processing_error.emit(str(e))


class BatchProcessingThread(QThread):
    """
    批量图像处理线程，用于处理多张图片。
    """
    processing_finished = Signal(list)  # 成功信号，携带处理后的图像列表
    processing_error = Signal(str)  # 失败信号，携带错误信息
    progress_update = Signal(str)  # 进度更新信号
    
    def __init__(self, image_paths, lut_table, lut_size, strength=1.0, dithering=None):
        super().__init__()
        self.image_paths = image_paths
        self.lut_table = lut_table
        self.lut_size = lut_size
        self.strength = strength
        self.dithering = dithering
    
    def run(self):
        try:
            processed_images = []
            total = len(self.image_paths)
            
            for i, img_path in enumerate(self.image_paths, 1):
                try:
                    # 加载图像
                    data = np.fromfile(img_path, dtype=np.uint8)
                    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                    
                    if image is None:
                        raise ValueError(f"文件解码失败: {img_path}")
                    
                    # 移除 Alpha 通道
                    if len(image.shape) == 3 and image.shape[-1] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    
                    # 应用 LUT
                    result = apply_lut_to_image(image, self.lut_table, self.lut_size, self.strength, self.dithering)
                    processed_images.append(result)
                    
                    # 发送进度更新
                    import os
                    filename = os.path.basename(img_path)
                    self.progress_update.emit(f"[{i}/{total}] 处理完成: {filename}")
                    
                except Exception as e:
                    self.progress_update.emit(f"[警告] 处理失败: {os.path.basename(img_path)} - {e}")
                    continue
            
            if processed_images:
                self.processing_finished.emit(processed_images)
            else:
                self.processing_error.emit("所有图片处理失败")
                
        except Exception as e:
            self.processing_error.emit(f"批处理过程出错: {str(e)}")
