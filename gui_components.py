"""
GUI 自定义组件模块
包含自动缩放图像标签等控件
"""

import cv2
from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtCore import Qt


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
