"""
LUT 处理核心模块
包含 LUT 文件解析和图像处理功能
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter
from PySide6.QtCore import QThread, Signal


# ==================== 蓝噪声贴图生成 ====================
def _generate_blue_noise_texture(size=256, seed=42):
    """
    生成固定的蓝噪声贴图（平铺用）。
    使用 Poisson Disk Sampling 的简化版本生成蓝噪声。
    
    Args:
        size: 贴图大小 (size x size)
        seed: 随机种子，保证可重现性
    
    Returns:
        shape (size, size) 的蓝噪声贴图，值域 [-0.5, 0.5]
    """
    np.random.seed(seed)
    
    # 使用高频 Perlin-like 噪声生成蓝噪声的近似
    # 方法：多个正弦波频率叠加形成蓝噪声特性
    x = np.linspace(0, 4 * np.pi, size)
    y = np.linspace(0, 4 * np.pi, size)
    xx, yy = np.meshgrid(x, y)
    
    # 多频率组合形成蓝噪声
    noise = np.sin(xx) * np.cos(yy) * 0.3
    noise += np.sin(xx * 0.7) * np.cos(yy * 0.9) * 0.3
    noise += np.sin(xx * 1.3) * np.cos(yy * 1.1) * 0.2
    noise += np.sin(xx * 2.1) * np.cos(yy * 1.7) * 0.2
    
    # 添加小幅随机扰动以增加蓝噪声特性
    random_noise = (np.random.uniform(-1, 1, (size, size)) * 0.1)
    noise = noise + random_noise
    
    # 归一化到 [-0.5, 0.5]
    noise_min = noise.min()
    noise_max = noise.max()
    noise = (noise - noise_min) / (noise_max - noise_min) - 0.5
    
    return noise.astype(np.float32)


# 全局蓝噪声贴图（模块加载时生成一次）
_BLUE_NOISE_TEXTURE = _generate_blue_noise_texture(256)


def _apply_blue_noise_dither(image, intensity=1.5):
    """
    使用蓝噪声贴图对图像进行抖动。
    
    Args:
        image: OpenCV BGR 格式 uint8 图像
        intensity: 抖动强度 (通常 1.0-3.0)
    
    Returns:
        抖动后的图像 (uint8)
    """
    img_float = image.astype(np.float32)
    h, w, c = img_float.shape
    
    # 将蓝噪声贴图平铺到图像大小
    texture_h, texture_w = _BLUE_NOISE_TEXTURE.shape
    dither_map = np.tile(_BLUE_NOISE_TEXTURE, (h // texture_h + 1, w // texture_w + 1))[:h, :w]
    
    # 扩展到3通道
    dither_map_3ch = np.stack([dither_map] * c, axis=-1)
    
    # 应用抖动
    dithered = img_float + dither_map_3ch * intensity
    
    return np.clip(dithered, 0, 255).astype(np.uint8)


# ==================== LUT 解析 ====================
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


# ==================== Debanding ====================
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


def apply_edge_preserving_debanding(image):
    """
    使用 Domain Transform Filter 进行 Debanding 处理。
    
    Args:
        image: OpenCV BGR 格式的图像 (uint8)
    
    Returns:
        处理后的 OpenCV BGR 格式图像 (uint8)
    """
    try:
        # 使用 Domain Transform Filter 进行边缘保持平滑
        result = cv2.ximgproc.dtFilter(
            image,  # guide image
            image,  # source image
            sigmaSpatial=30,
            sigmaColor=30,
            mode=cv2.ximgproc.DTF_NC,
            numIters=3
        )
        
    except AttributeError:
        # 如果没有安装 opencv-contrib，回退到双边滤波
        result = cv2.bilateralFilter(image, d=9, sigmaColor=35, sigmaSpace=35)
    
    return result


def apply_lut_to_image(source_img, lut_table, lut_size, strength=1.0, debanding=False):
    """
    将 3D LUT 应用到图像上。

    Args:
        source_img: OpenCV BGR 格式的图像
        lut_table: LUT 数据表
        lut_size: LUT 维度大小
        strength: LUT 强度 (0.0-1.0)，1.0为完全应用
        debanding: 是否启用 Debanding 处理

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
    
    # 6. 应用LUT之后、debanding之前的蓝噪声抖动
    if debanding:
        result_bgr = _apply_blue_noise_dither(result_bgr, intensity=2.0)
    
    # 7. 应用后处理 Debanding（dtFilter + 梯度重建）
    if debanding:
        result_bgr = apply_edge_preserving_debanding(result_bgr)

    return result_bgr


class ImageProcessingThread(QThread):
    """
    后台图像处理线程，防止阻塞 UI 主线程。
    使用 Pillow 的 C 语言底层滤镜进行加速。
    """
    processing_finished = Signal(object)  # 成功信号，携带处理后的 OpenCV 图像
    processing_error = Signal(str)  # 失败信号，携带错误信息

    def __init__(self, source_img, lut_table, lut_size, strength=1.0, debanding=False):
        super().__init__()
        self.source_img = source_img
        self.lut_table = lut_table
        self.lut_size = lut_size
        self.strength = strength
        self.debanding = debanding

    def run(self):
        try:
            result_bgr = apply_lut_to_image(
                self.source_img, 
                self.lut_table, 
                self.lut_size,
                self.strength,
                self.debanding
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
    
    def __init__(self, image_paths, lut_table, lut_size, strength=1.0, debanding=False):
        super().__init__()
        self.image_paths = image_paths
        self.lut_table = lut_table
        self.lut_size = lut_size
        self.strength = strength
        self.debanding = debanding
    
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
                    result = apply_lut_to_image(image, self.lut_table, self.lut_size, self.strength, self.debanding)
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
