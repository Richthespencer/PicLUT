"""
PicLUT - 主应用程序
图像 LUT 处理工具的主窗口和程序入口
"""

import sys
import os
import shutil
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QListWidget, QListWidgetItem, QLabel
)
from PySide6.QtCore import Slot, Qt

# 导入自定义模块
from lut_processing import parse_cube_lut, ImageProcessingThread
from gui_components import AutoResizingLabel


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
        
        # 批处理状态
        self.image_paths = []  # 存储所有选择的图片路径
        self.loaded_images = []  # 存储加载的图片数据
        self.batch_mode = False  # 是否处于批处理模式

        # LUT 目录
        self._ensure_lut_dirs()

        self._init_ui()
        self._apply_theme()

    def _init_ui(self):
        """初始化 UI 布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        main_layout.addLayout(content_layout, stretch=1)

        # 左侧 LUT 管理面板
        lut_panel = QVBoxLayout()
        lut_panel.setSpacing(10)

        lbl_lut_title = QLabel("LUT 管理")
        lbl_lut_title.setStyleSheet("font-size: 15px; font-weight: 600;")

        self.lut_list = QListWidget()
        self.lut_list.setMinimumWidth(220)
        self.lut_list.itemDoubleClicked.connect(self.on_lut_double_clicked)

        lut_btn_layout = QHBoxLayout()
        self.btn_add_lut = QPushButton("添加LUT")
        self.btn_del_lut = QPushButton("删除LUT")
        for btn in [self.btn_add_lut, self.btn_del_lut]:
            btn.setMinimumHeight(36)
        self.btn_add_lut.clicked.connect(self.on_add_lut)
        self.btn_del_lut.clicked.connect(self.on_delete_lut)
        lut_btn_layout.addWidget(self.btn_add_lut)
        lut_btn_layout.addWidget(self.btn_del_lut)

        lut_panel.addWidget(lbl_lut_title)
        lut_panel.addWidget(self.lut_list, stretch=1)
        lut_panel.addLayout(lut_btn_layout)

        content_layout.addLayout(lut_panel, stretch=0)

        # 右侧主内容区域
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

        # 1. 图像预览区域
        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(15)

        self.lbl_source = AutoResizingLabel("原始图像 (Source)")
        self.lbl_result = AutoResizingLabel("处理结果 (Result)")

        preview_layout.addWidget(self.lbl_source, stretch=1)
        preview_layout.addWidget(self.lbl_result, stretch=1)

        right_layout.addLayout(preview_layout, stretch=1)

        # 2. 控制按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        self.btn_open_img = QPushButton("打开图片")
        self.btn_open_lut = QPushButton("导入 LUT (.cube)")
        self.btn_preview = QPushButton("预览效果")
        self.btn_process = QPushButton("应用处理")
        self.btn_save = QPushButton("导出结果")

        # 绑定信号
        self.btn_open_img.clicked.connect(self.on_open_image)
        self.btn_open_lut.clicked.connect(self.on_open_lut)
        self.btn_preview.clicked.connect(self.on_preview)
        self.btn_process.clicked.connect(self.on_process_start)
        self.btn_save.clicked.connect(self.on_save_result)
        
        # 初始隐藏预览按钮
        self.btn_preview.setVisible(False)

        # 设置按钮统一高度
        for btn in [self.btn_open_img, self.btn_open_lut, self.btn_preview, self.btn_process, self.btn_save]:
            btn.setMinimumHeight(45)
            btn_layout.addWidget(btn)

        right_layout.addLayout(btn_layout)

        # 3. 日志输出区域
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMaximumHeight(120)
        right_layout.addWidget(self.log_viewer)

        content_layout.addLayout(right_layout, stretch=1)

        # 初始化列表
        self._load_lut_list()

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

            QListWidget {
                background-color: #252526;
                border: 1px solid #333;
                border-radius: 6px;
                color: #e0e0e0;
                padding: 4px;
            }
            QListWidget::item { padding: 6px 8px; }
            QListWidget::item:selected { background-color: #3a3a3a; color: #ffffff; }
            QListWidget::item:hover { background-color: #333; }

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

    # ==================== LUT 管理 ====================

    def _ensure_lut_dirs(self):
        """确保 LUT 基础目录和自定义目录存在"""
        base_dir = os.path.join(os.path.dirname(__file__), "LUT")
        custom_dir = os.path.join(base_dir, "Custom")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(custom_dir, exist_ok=True)
        self.lut_base_dir = base_dir
        self.custom_lut_dir = custom_dir

    def _load_lut_list(self):
        """加载 LUT 列表（内置 + 自定义）"""
        self.lut_list.clear()
        if not os.path.isdir(self.lut_base_dir):
            return

        lut_files = []
        for root, _, files in os.walk(self.lut_base_dir):
            for name in files:
                if name.lower().endswith('.cube'):
                    lut_files.append((root, name))

        lut_files.sort(key=lambda x: x[1].lower())

        for root, name in lut_files:
            full_path = os.path.join(root, name)
            try:
                is_custom = os.path.commonpath([self.custom_lut_dir, full_path]) == self.custom_lut_dir
            except ValueError:
                is_custom = False
            prefix = "自定义" if is_custom else "内置"
            rel = os.path.relpath(full_path, self.lut_base_dir)
            item = QListWidgetItem(f"[{prefix}] {rel}")
            item.setData(Qt.UserRole, full_path)
            self.lut_list.addItem(item)

    @Slot()
    def on_add_lut(self):
        """添加自定义 LUT 到本地目录"""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "添加 LUT", "", "LUT Files (*.cube)")
        if not file_paths:
            return

        added = 0
        for src in file_paths:
            try:
                filename = os.path.basename(src)
                target = os.path.join(self.custom_lut_dir, filename)

                # 避免重名覆盖
                base, ext = os.path.splitext(target)
                counter = 1
                while os.path.exists(target):
                    target = f"{base}_{counter}{ext}"
                    counter += 1

                shutil.copy2(src, target)
                added += 1
            except Exception as e:
                self.log(f"[错误] 添加失败: {os.path.basename(src)} - {e}")

        if added:
            self.log(f"已添加 {added} 个 LUT 到本地库")
            self._load_lut_list()

    @Slot()
    def on_delete_lut(self):
        """删除自定义目录中的 LUT"""
        item = self.lut_list.currentItem()
        if not item:
            self.log("[警告] 请先选择要删除的 LUT")
            return

        path = item.data(Qt.UserRole)
        if not path.startswith(self.custom_lut_dir):
            self.log("[警告] 仅支持删除自定义目录中的 LUT")
            return

        try:
            os.remove(path)
            self.log(f"已删除 LUT: {os.path.basename(path)}")
            self._load_lut_list()
        except Exception as e:
            self.log(f"[错误] 删除失败: {e}")

    @Slot(object)
    def on_lut_double_clicked(self, item):
        """双击列表加载并选择 LUT"""
        path = item.data(Qt.UserRole)
        try:
            self.lut_table, self.lut_size = parse_cube_lut(path)
            self.log(f"已选择 LUT: {os.path.basename(path)} (尺寸: {self.lut_size}^3)")
        except Exception as e:
            self.log(f"[错误] 加载 LUT 失败: {e}")

    # ==================== 槽函数 (业务逻辑) ====================

    @Slot()
    def on_open_image(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片（可多选）", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if file_paths:
            try:
                self.image_paths = file_paths
                self.loaded_images = []
                
                # 加载第一张图片用于预览
                data = np.fromfile(file_paths[0], dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

                if image is None:
                    raise ValueError("文件解码失败或格式不支持")

                # 移除可能存在的 Alpha 通道，简化处理
                if image.shape[-1] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

                self.source_image = image
                self.lbl_source.set_image(self.source_image)
                
                # 判断是否为批处理模式
                if len(file_paths) > 1:
                    self.batch_mode = True
                    self.btn_preview.setVisible(True)
                    self.log(f"已选择 {len(file_paths)} 张图像，显示第一张预览")
                    self.log("提示：点击'预览效果'查看第一张图片的LUT效果")
                else:
                    self.batch_mode = False
                    self.btn_preview.setVisible(False)
                    self.log(f"已加载图像: {os.path.basename(file_paths[0])}")

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
    def on_preview(self):
        """预览第一张图片的LUT效果"""
        if self.source_image is None:
            self.log("[警告] 请先加载图像")
            return
        if self.lut_table is None:
            self.log("[警告] 请先加载 LUT 文件")
            return
        
        self.btn_preview.setEnabled(False)
        self.btn_preview.setText("预览中...")
        self.log("正在预览第一张图片的效果...")
        
        # 启动后台线程处理预览
        self.worker_thread = ImageProcessingThread(
            self.source_image, self.lut_table, self.lut_size
        )
        self.worker_thread.processing_finished.connect(self.on_preview_finished)
        self.worker_thread.processing_error.connect(self.on_process_error)
        self.worker_thread.start()
    
    @Slot(object)
    def on_preview_finished(self, result_image):
        """预览完成"""
        self.processed_image = result_image
        self.lbl_result.set_image(self.processed_image)
        self.log("预览完成，如果效果满意可点击'应用处理'批量处理所有图片")
        self.btn_preview.setEnabled(True)
        self.btn_preview.setText("预览效果")

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
        
        if self.batch_mode:
            # 批处理模式
            self.log(f"开始批量处理 {len(self.image_paths)} 张图片...")
            
            # 导入批处理线程
            from lut_processing import BatchProcessingThread
            
            self.worker_thread = BatchProcessingThread(
                self.image_paths, self.lut_table, self.lut_size
            )
            self.worker_thread.progress_update.connect(self.on_batch_progress)
            self.worker_thread.processing_finished.connect(self.on_batch_finished)
            self.worker_thread.processing_error.connect(self.on_process_error)
            self.worker_thread.start()
        else:
            # 单张处理模式
            self.log("开始应用 3D LUT，请稍候...")
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
    def on_batch_progress(self, message):
        """批处理进度更新"""
        self.log(message)
    
    @Slot(list)
    def on_batch_finished(self, processed_images):
        """批处理完成"""
        self.loaded_images = processed_images
        
        # 显示第一张处理后的图片
        if processed_images:
            self.processed_image = processed_images[0]
            self.lbl_result.set_image(self.processed_image)
        
        self.log(f"批量处理完成！共处理 {len(processed_images)} 张图片")
        self._reset_process_btn()
        
        # 自动弹出保存对话框
        self.on_batch_save()

    @Slot(str)
    def on_process_error(self, error_msg):
        self.log(f"[错误] 处理过程中发生异常: {error_msg}")
        self._reset_process_btn()

    def _reset_process_btn(self):
        self.btn_process.setEnabled(True)
        self.btn_process.setText("应用处理")

    @Slot()
    def on_save_result(self):
        if self.batch_mode and self.loaded_images:
            # 批处理模式，调用批量保存
            self.on_batch_save()
        elif self.processed_image is not None:
            # 单张保存模式
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
        else:
            self.log("[警告] 没有可保存的处理结果")
    
    def on_batch_save(self):
        """批量保存处理后的图片"""
        if not self.loaded_images:
            self.log("[警告] 没有可保存的处理结果")
            return
        
        # 选择保存文件夹
        save_dir = QFileDialog.getExistingDirectory(
            self, "选择保存文件夹", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if save_dir:
            try:
                success_count = 0
                for i, (img, original_path) in enumerate(zip(self.loaded_images, self.image_paths)):
                    # 获取原文件名
                    base_name = os.path.basename(original_path)
                    name, ext = os.path.splitext(base_name)
                    
                    # 生成新文件名（添加_lut后缀）
                    new_name = f"{name}_lut{ext}"
                    save_path = os.path.join(save_dir, new_name)
                    
                    # 保存图片
                    is_success, buffer = cv2.imencode(ext if ext else '.png', img)
                    if is_success:
                        with open(save_path, "wb") as f:
                            buffer.tofile(f)
                        success_count += 1
                    else:
                        self.log(f"[错误] 编码失败: {base_name}")
                
                self.log(f"批量保存完成！成功保存 {success_count}/{len(self.loaded_images)} 张图片")
                self.log(f"保存位置: {save_dir}")
                
            except Exception as e:
                self.log(f"[错误] 批量保存失败: {e}")


if __name__ == "__main__":
    # 创建应用程序实例
    app = QApplication(sys.argv)

    # 创建并显示主窗口
    window = LutAppWindow()
    window.show()

    # 进入事件循环
    sys.exit(app.exec())
