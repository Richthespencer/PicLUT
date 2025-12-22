import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFileDialog, QSizePolicy
)
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtCore import Qt, QThread, Signal, Slot


# ==================== 工具函数 ====================

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


# ==================== 工作线程 ====================

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
            # 1. 颜色空间转换: OpenCV (BGR) -> Pillow (RGB)
            img_rgb = cv2.cvtColor(self.source_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)

            # 2. 构建 3D LUT 滤镜
            lut_filter = ImageFilter.Color3DLUT(
                size=self.lut_size,
                table=self.lut_table,
                channels=3,
                target_mode=None
            )

            # 3. 应用滤镜 (计算密集型步骤)
            processed_pil = pil_image.filter(lut_filter)

            # 4. 转换回 OpenCV 格式: Pillow (RGB) -> OpenCV (BGR)
            processed_np = np.asarray(processed_pil)
            result_bgr = cv2.cvtColor(processed_np, cv2.COLOR_RGB2BGR)

            self.processing_finished.emit(result_bgr)

        except Exception as e:
            self.processing_error.emit(str(e))


# ==================== 自定义控件 ====================

class AutoResizingLabel(QLabel):
    """
    自定义 QLabel，支持根据窗口大小自动缩放显示的图像。
    避免了标准 QLabel 被高分辨率图像撑大导致窗口无法缩小的问题。
    """

    def __init__(self, placeholder_text=""):
        super().__init__(placeholder_text)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #444; 
                background-color: #2b2b2b; 
                color: #888; 
                font-size: 14px;
                font-family: Arial;
            }
        """)
        # 忽略内容尺寸，允许布局管理器自由压缩或拉伸此控件
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._pixmap = None

    def set_image(self, cv_img):
        """
        设置要显示的 OpenCV 图像。
        """
        if cv_img is None:
            return

        # BGR -> RGB -> QImage -> QPixmap
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 保存原始分辨率的 Pixmap，用于动态重绘
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()  # 触发 paintEvent

    def paintEvent(self, event):
        """
        动态绘制事件。根据控件当前的实时尺寸缩放并居中绘制图像。
        """
        if not self._pixmap:
            super().paintEvent(event)  # 显示占位文字
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # 开启平滑抗锯齿

        # 计算适应当前窗口的尺寸 (保持纵横比)
        target_size = self.size()
        scaled_pixmap = self._pixmap.scaled(
            target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # 计算居中坐标
        x = (target_size.width() - scaled_pixmap.width()) // 2
        y = (target_size.height() - scaled_pixmap.height()) // 2

        painter.drawPixmap(x, y, scaled_pixmap)


# ==================== 主窗口 ====================

class LutAppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PicLUT - Apply LUT to Images")
        self.resize(1200, 800)

        # 数据状态
        self.source_image = None
        self.processed_image = None
        self.lut_table = None
        self.lut_size = None
        self.worker_thread = None

        self._init_ui()
        self._apply_theme()

    def _init_ui(self):
        """初始化 UI 布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 1. 图像预览区域
        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(15)

        self.lbl_source = AutoResizingLabel("原始图像 (Source)")
        self.lbl_result = AutoResizingLabel("处理结果 (Result)")

        preview_layout.addWidget(self.lbl_source, stretch=1)
        preview_layout.addWidget(self.lbl_result, stretch=1)

        main_layout.addLayout(preview_layout, stretch=1)

        # 2. 控制按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        self.btn_open_img = QPushButton("打开图片")
        self.btn_open_lut = QPushButton("导入 LUT (.cube)")
        self.btn_process = QPushButton("应用处理")
        self.btn_save = QPushButton("导出结果")

        # 绑定信号
        self.btn_open_img.clicked.connect(self.on_open_image)
        self.btn_open_lut.clicked.connect(self.on_open_lut)
        self.btn_process.clicked.connect(self.on_process_start)
        self.btn_save.clicked.connect(self.on_save_result)

        # 设置按钮统一高度
        for btn in [self.btn_open_img, self.btn_open_lut, self.btn_process, self.btn_save]:
            btn.setMinimumHeight(45)
            btn_layout.addWidget(btn)

        main_layout.addLayout(btn_layout)

        # 3. 日志输出区域
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMaximumHeight(120)
        main_layout.addWidget(self.log_viewer)

    def _apply_theme(self):
        """应用暗色系样式表"""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }

            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 6px;
                color: #ffffff;
                font-size: 14px;
                padding: 0 15px;
            }
            QPushButton:hover { background-color: #4a4a4a; border-color: #666; }
            QPushButton:pressed { background-color: #2a2a2a; border-color: #444; }
            QPushButton:disabled { background-color: #252525; color: #666; border-color: #333; }

            QTextEdit {
                background-color: #252526;
                border: 1px solid #333;
                border-radius: 4px;
                color: #cccccc;
                font-family: Consolas, monospace;
            }

            /* 滚动条样式优化 */
            QScrollBar:vertical {
                border: none; background: #2b2b2b; width: 10px; margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #555; min-height: 20px; border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

    def log(self, message):
        """向日志区域添加信息"""
        self.log_viewer.append(f"» {message}")
        # 自动滚动到底部
        scrollbar = self.log_viewer.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ==================== 槽函数 (业务逻辑) ====================

    @Slot()
    def on_open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if file_path:
            try:
                # 使用 imdecode 处理包含非 ASCII 字符的路径
                data = np.fromfile(file_path, dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

                if image is None:
                    raise ValueError("文件解码失败或格式不支持")

                # 移除可能存在的 Alpha 通道，简化处理
                if image.shape[-1] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

                self.source_image = image
                self.lbl_source.set_image(self.source_image)
                self.log(f"已加载图像: {os.path.basename(file_path)}")

            except Exception as e:
                self.log(f"[错误] 加载图像失败: {e}")

    @Slot()
    def on_open_lut(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 LUT 文件", "", "LUT Files (*.cube)")
        if file_path:
            try:
                self.lut_table, self.lut_size = parse_cube_lut(file_path)
                self.log(f"已加载 LUT: {os.path.basename(file_path)} (尺寸: {self.lut_size}^3)")
            except Exception as e:
                self.log(f"[错误] 解析 LUT 失败: {e}")

    @Slot()
    def on_process_start(self):
        if self.source_image is None:
            self.log("[警告] 请先加载原始图像")
            return
        if self.lut_table is None:
            self.log("[警告] 请先加载 LUT 文件")
            return

        self.btn_process.setEnabled(False)
        self.btn_process.setText("正在处理...")
        self.log("开始应用 3D LUT，请稍候...")

        # 启动后台线程
        self.worker_thread = ImageProcessingThread(
            self.source_image, self.lut_table, self.lut_size
        )
        self.worker_thread.processing_finished.connect(self.on_process_finished)
        self.worker_thread.processing_error.connect(self.on_process_error)
        self.worker_thread.start()

    @Slot(object)
    def on_process_finished(self, result_image):
        self.processed_image = result_image
        self.lbl_result.set_image(self.processed_image)
        self.log("处理完成")
        self._reset_process_btn()

    @Slot(str)
    def on_process_error(self, error_msg):
        self.log(f"[错误] 处理过程中发生异常: {error_msg}")
        self._reset_process_btn()

    def _reset_process_btn(self):
        self.btn_process.setEnabled(True)
        self.btn_process.setText("应用处理")

    @Slot()
    def on_save_result(self):
        if self.processed_image is None:
            self.log("[警告] 没有可保存的处理结果")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tiff)"
        )

        if file_path:
            try:
                # 自动补全后缀
                valid_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
                ext = os.path.splitext(file_path)[1].lower()
                if not ext or ext not in valid_extensions:
                    file_path += ".png"
                    ext = ".png"

                # 使用 imencode 处理中文路径保存
                is_success, buffer = cv2.imencode(ext, self.processed_image)
                if is_success:
                    with open(file_path, "wb") as f:
                        buffer.tofile(f)
                    self.log(f"已保存至: {file_path}")
                else:
                    self.log("[错误] 图像编码失败")
            except Exception as e:
                self.log(f"[错误] 保存失败: {e}")


# ==================== 程序入口 ====================

if __name__ == "__main__":
    # 创建应用程序实例
    app = QApplication(sys.argv)

    # 创建并显示主窗口
    window = LutAppWindow()
    window.show()

    # 进入事件循环
    sys.exit(app.exec())