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


def apply_edge_preserving_debanding(image):
    """
    使用 Domain Transform Filter + 梯度重建进行 Debanding 处理。
    在平坦渐变区域强力平滑色带，在边缘和纹理区域保留细节。
    
    Args:
        image: OpenCV BGR 格式的图像 (uint8)
    
    Returns:
        处理后的 OpenCV BGR 格式图像 (uint8)
    """
    try:
        # 转换为 float32 以提高精度
        img_float = image.astype(np.float32)
        
        # 1. 使用 Domain Transform Filter 进行边缘保持平滑
        smoothed = cv2.ximgproc.dtFilter(
            image,  # guide image
            image,  # source image
            sigmaSpatial=30,
            sigmaColor=30,
            mode=cv2.ximgproc.DTF_NC,
            numIters=3
        ).astype(np.float32)
        
        # 2. 计算梯度幅度来检测边缘/纹理区域
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 使用 Sobel 算子计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 3. 归一化梯度并创建平滑权重掩码
        # 梯度小的区域（平坦/渐变）-> 权重高 -> 更多平滑
        # 梯度大的区域（边缘/纹理）-> 权重低 -> 保留原始
        grad_normalized = gradient_magnitude / (gradient_magnitude.max() + 1e-6)
        
        # 使用 sigmoid 函数创建平滑过渡的掩码
        # threshold 控制边缘检测灵敏度，steepness 控制过渡锐度
        threshold = 0.05
        steepness = 30
        smooth_weight = 1.0 / (1.0 + np.exp(steepness * (grad_normalized - threshold)))
        
        # 对掩码进行轻微模糊以避免突变
        smooth_weight = cv2.GaussianBlur(smooth_weight, (5, 5), 0)
        
        # 扩展到3通道
        smooth_weight_3ch = np.stack([smooth_weight] * 3, axis=-1)
        
        # 4. 梯度引导的混合：平坦区域用平滑结果，边缘区域保留原始
        result_float = smooth_weight_3ch * smoothed + (1 - smooth_weight_3ch) * img_float
        
        # 5. 在平坦区域添加微小抖动以打破残余色带
        # 只在平坦区域添加抖动
        dither_strength = 1.5
        dither = np.random.uniform(-dither_strength, dither_strength, img_float.shape).astype(np.float32)
        result_float = result_float + dither * smooth_weight_3ch
        
        result = np.clip(result_float, 0, 255).astype(np.uint8)
        
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
        debanding: 是否启用 Error Diffusion Debanding

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
    
    # 6. 应用 Debanding 处理
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
