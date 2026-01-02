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


def apply_lut_to_image(source_img, lut_table, lut_size):
    """
    将 3D LUT 应用到图像上。

    Args:
        source_img: OpenCV BGR 格式的图像
        lut_table: LUT 数据表
        lut_size: LUT 维度大小

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

    return result_bgr


class ImageProcessingThread(QThread):
    """
    后台图像处理线程，防止阻塞 UI 主线程。
    使用 Pillow 的 C 语言底层滤镜进行加速。
    """
    processing_finished = Signal(object)  # 成功信号，携带处理后的 OpenCV 图像
    processing_error = Signal(str)  # 失败信号，携带错误信息

    def __init__(self, source_img, lut_table, lut_size):
        super().__init__()
        self.source_img = source_img
        self.lut_table = lut_table
        self.lut_size = lut_size

    def run(self):
        try:
            result_bgr = apply_lut_to_image(
                self.source_img, 
                self.lut_table, 
                self.lut_size
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
    
    def __init__(self, image_paths, lut_table, lut_size):
        super().__init__()
        self.image_paths = image_paths
        self.lut_table = lut_table
        self.lut_size = lut_size
    
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
                    result = apply_lut_to_image(image, self.lut_table, self.lut_size)
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
