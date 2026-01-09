"""
PicLUT - 主应用程序
图像 LUT 处理工具的主窗口和程序入口
"""

import sys
import os
import shutil
import json
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QListWidget, QListWidgetItem, QLabel,
    QMenu, QInputDialog, QTreeWidget, QTreeWidgetItem, QSlider, QComboBox
)
from PySide6.QtCore import Slot, Qt, QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtGui import QDragEnterEvent, QDropEvent

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
        
        # LUT 强度
        self.lut_strength = 1.0  # 默认100%
        self.last_preview_strength = None  # 最近一次预览使用的强度
        
        # 抖动选项
        self.dithering_mode = None  # None / 'ordered' / 'noise' / 'floyd'

        # LUT 目录
        self._ensure_lut_dirs()
        self.config_file = os.path.join(self.lut_base_dir, '.lut_config.json')
        self.pinned_luts = self._load_config()
        
        # 文件系统监视定时器
        self.lut_refresh_timer = QTimer(self)
        self.lut_refresh_timer.timeout.connect(self._refresh_lut_tree)
        self.lut_refresh_timer.start(3000)  # 每3秒检查一次
        self._last_lut_mtime = 0

        # 预览与滑条强度同步定时器
        self.preview_sync_timer = QTimer(self)
        self.preview_sync_timer.setInterval(200)
        self.preview_sync_timer.timeout.connect(self._ensure_preview_synced)

        self._init_ui()
        self._apply_theme()
        self.preview_sync_timer.start()
        
        # 启用拖放功能
        self.setAcceptDrops(True)

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

        self.lut_tree = QTreeWidget()
        self.lut_tree.setMinimumWidth(250)
        self.lut_tree.setHeaderHidden(True)
        self.lut_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lut_tree.customContextMenuRequested.connect(self.on_lut_context_menu)
        self.lut_tree.itemDoubleClicked.connect(self.on_lut_double_clicked)

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
        lut_panel.addWidget(self.lut_tree, stretch=1)
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

        # 1.5 LUT 强度滑块
        strength_layout = QHBoxLayout()
        strength_layout.setSpacing(10)
        
        self.lbl_strength_title = QLabel("LUT 强度:")
        self.lbl_strength_value = QLabel("100%")
        self.lbl_strength_value.setMinimumWidth(45)
        self.lbl_strength_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setValue(100)
        self.strength_slider.setTickPosition(QSlider.TicksBelow)
        self.strength_slider.setTickInterval(10)
        self.strength_slider.valueChanged.connect(self.on_strength_changed)
        
        strength_layout.addWidget(self.lbl_strength_title)
        strength_layout.addWidget(self.strength_slider, stretch=1)
        strength_layout.addWidget(self.lbl_strength_value)
        
        # 初始隐藏强度控制
        self.lbl_strength_title.setVisible(False)
        self.strength_slider.setVisible(False)
        self.lbl_strength_value.setVisible(False)
        
        right_layout.addLayout(strength_layout)

        # 1.6 去条纹（抖动）选项
        dithering_layout = QHBoxLayout()
        dithering_layout.setSpacing(10)
        
        self.lbl_dithering_title = QLabel("去条纹:")
        self.dithering_combo = QComboBox()
        self.dithering_combo.addItem("无", None)
        self.dithering_combo.addItem("有序抖动 (快)", "ordered")
        self.dithering_combo.addItem("噪声抖动 (平衡)", "noise")
        self.dithering_combo.addItem("Floyd-Steinberg (高质量)", "floyd")
        self.dithering_combo.setCurrentIndex(0)
        self.dithering_combo.currentIndexChanged.connect(self.on_dithering_changed)
        
        dithering_layout.addWidget(self.lbl_dithering_title)
        dithering_layout.addWidget(self.dithering_combo, stretch=1)
        
        # 初始隐藏抖动控制
        self.lbl_dithering_title.setVisible(False)
        self.dithering_combo.setVisible(False)
        
        right_layout.addLayout(dithering_layout)

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
        self._load_lut_tree()

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

            QTreeWidget {
                background-color: #252526;
                border: 1px solid #333;
                border-radius: 6px;
                color: #e0e0e0;
                padding: 4px;
                outline: none;
                show-decoration-selected: 0;
            }
            QTreeWidget::item { 
                padding: 6px 4px;
                border-radius: 3px;
                outline: none;
                border: none;
            }
            QTreeWidget::item:selected { 
                background-color: #3a3a3a; 
                color: #ffffff; 
                outline: none;
                border: none;
            }
            QTreeWidget::item:focus {
                background-color: #3a3a3a;
                outline: none;
                border: none;
            }
            QTreeWidget::item:hover { 
                background-color: #333; 
            }
            QTreeWidget::branch {
                background: transparent;
            }
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {
                border-image: none;
                image: none;
            }
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {
                border-image: none;
                image: none;
            }
            QTreeWidget::branch:has-siblings:!adjoins-item {
                border-image: none;
            }
            QTreeWidget::branch:has-siblings:adjoins-item {
                border-image: none;
            }
            QTreeWidget::branch:!has-children:!has-siblings:adjoins-item {
                border-image: none;
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #333;
                height: 6px;
                background: #2b2b2b;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4a9eff;
                border: 1px solid #3a8eef;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #5aafff;
            }
            QSlider::sub-page:horizontal {
                background: #4a9eff;
                border: 1px solid #333;
                height: 6px;
                border-radius: 3px;
            }

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

    # ==================== 拖放事件处理 ====================
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """处理拖放进入事件"""
        if event.mimeData().hasUrls():
            # 检查是否包含图片文件
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if self._is_image_file(file_path):
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """处理拖放释放事件"""
        urls = event.mimeData().urls()
        image_files = []
        lut_files = []
        
        # 分类拖放的文件
        for url in urls:
            file_path = url.toLocalFile()
            if self._is_image_file(file_path):
                image_files.append(file_path)
            elif self._is_lut_file(file_path):
                lut_files.append(file_path)
        
        # 处理图片文件
        if image_files:
            self._load_images_from_paths(image_files)
        
        # 处理 LUT 文件
        if lut_files:
            # 只加载第一个 LUT 文件
            self._load_lut_from_path(lut_files[0])
        
        event.acceptProposedAction()
    
    def _is_image_file(self, file_path: str) -> bool:
        """判断是否为图片文件"""
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        _, ext = os.path.splitext(file_path)
        return ext.lower() in supported_formats
    
    def _is_lut_file(self, file_path: str) -> bool:
        """判断是否为 LUT 文件"""
        _, ext = os.path.splitext(file_path)
        return ext.lower() == '.cube'
    
    def _load_images_from_paths(self, file_paths: list):
        """从文件路径加载图片"""
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
            self.last_preview_strength = None  # 重新加载图片后重置预览状态
            
            # 显示LUT强度滑块和抖动控制
            self.lbl_strength_title.setVisible(True)
            self.strength_slider.setVisible(True)
            self.lbl_strength_value.setVisible(True)
            self.lbl_dithering_title.setVisible(True)
            self.dithering_combo.setVisible(True)
            
            # 判断是否为批处理模式
            if len(file_paths) > 1:
                self.batch_mode = True
                self.btn_preview.setVisible(True)
                self.log(f"已拖放 {len(file_paths)} 张图像，显示第一张预览")
            else:
                self.batch_mode = False
                self.btn_preview.setVisible(False)
                self.log(f"已拖放图像: {os.path.basename(file_paths[0])}")
            
            # 如果已加载LUT，自动预览
            if self.lut_table is not None:
                self._apply_lut_preview()

        except Exception as e:
            self.log(f"[错误] 加载图像失败: {e}")
    
    def _load_lut_from_path(self, file_path: str):
        """从文件路径加载 LUT"""
        try:
            self.lut_table, self.lut_size = parse_cube_lut(file_path)
            self.log(f"已拖放 LUT: {os.path.basename(file_path)} (尺寸: {self.lut_size}^3)")
            self.last_preview_strength = None  # 新 LUT 需重新预览
            
            # 如果已加载图像，自动预览
            if self.source_image is not None:
                self._apply_lut_preview()
        except Exception as e:
            self.log(f"[错误] 加载 LUT 失败: {e}")

    # ==================== LUT 管理 ====================

    def _ensure_lut_dirs(self):
        """确保 LUT 基础目录和自定义目录存在"""
        base_dir = os.path.join(os.path.dirname(__file__), "LUT")
        custom_dir = os.path.join(base_dir, "Custom")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(custom_dir, exist_ok=True)
        self.lut_base_dir = base_dir
        self.custom_lut_dir = custom_dir
    
    def _load_config(self):
        """加载配置文件（置顶列表）"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return set(config.get('pinned', []))
            except:
                return set()
        return set()
    
    def _save_config(self):
        """保存配置文件"""
        try:
            config = {'pinned': list(self.pinned_luts)}
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"[错误] 保存配置失败: {e}")

    def _load_lut_tree(self):
        """加载LUT树状结构（文件夹+文件）"""
        # 保存当前展开状态
        expanded_paths = self._get_expanded_paths()
        
        self.lut_tree.clear()
        if not os.path.isdir(self.lut_base_dir):
            return
        
        # 添加根目录的内容（置顶项会在目录内排在前面）
        self._add_directory_contents(self.lut_tree, self.lut_base_dir, self.lut_base_dir)
        
        # 恢复展开状态
        self._restore_expanded_paths(expanded_paths)
    
    def _get_expanded_paths(self):
        """获取当前所有展开的文件夹路径"""
        expanded = set()
        
        def collect_expanded(item):
            if item.isExpanded():
                path = item.data(0, Qt.UserRole)
                item_type = item.data(0, Qt.UserRole + 2)
                if path and item_type == "folder":
                    expanded.add(path)
            
            for i in range(item.childCount()):
                collect_expanded(item.child(i))
        
        root = self.lut_tree.invisibleRootItem()
        for i in range(root.childCount()):
            collect_expanded(root.child(i))
        
        return expanded
    
    def _restore_expanded_paths(self, expanded_paths):
        """恢复文件夹的展开状态"""
        if not expanded_paths:
            return
        
        def restore_item(item):
            path = item.data(0, Qt.UserRole)
            item_type = item.data(0, Qt.UserRole + 2)
            
            if path in expanded_paths and item_type == "folder":
                item.setExpanded(True)
            
            for i in range(item.childCount()):
                restore_item(item.child(i))
        
        root = self.lut_tree.invisibleRootItem()
        for i in range(root.childCount()):
            restore_item(root.child(i))
    
    def _add_directory_contents(self, parent, dir_path, base_path):
        """递归添加目录内容"""
        try:
            items = os.listdir(dir_path)
        except PermissionError:
            return
        
        # 过滤隐藏文件
        items = [item for item in items if not item.startswith('.')]
        
        # 分类：文件夹和文件
        folders = []
        files = []
        
        for item_name in items:
            item_path = os.path.join(dir_path, item_name)
            
            if os.path.isdir(item_path):
                folders.append((item_name, item_path))
            elif item_name.lower().endswith('.cube'):
                files.append((item_name, item_path))
        
        # 按名称排序
        folders.sort(key=lambda x: x[0].lower())
        files.sort(key=lambda x: x[0].lower())
        
        # 分离置顶和非置顶文件
        pinned_files = [(name, path) for name, path in files if path in self.pinned_luts]
        unpinned_files = [(name, path) for name, path in files if path not in self.pinned_luts]
        
        # 先添加置顶文件
        for item_name, item_path in pinned_files:
            self._add_file_item(parent, item_path, is_pinned=True)
        
        # 再添加非置顶文件
        for item_name, item_path in unpinned_files:
            self._add_file_item(parent, item_path, is_pinned=False)
        
        # 最后添加文件夹（并递归处理其内容）
        for item_name, item_path in folders:
            folder_item = QTreeWidgetItem(parent, [f"📁 {item_name}"])
            folder_item.setData(0, Qt.UserRole, item_path)
            folder_item.setData(0, Qt.UserRole + 1, False)  # is_pinned
            folder_item.setData(0, Qt.UserRole + 2, "folder")
            folder_item.setExpanded(False)
            
            # 递归添加子内容
            self._add_directory_contents(folder_item, item_path, base_path)
    
    def _add_file_item(self, parent, file_path, is_pinned=False):
        """添加文件项"""
        file_name = os.path.basename(file_path)
        pin_icon = "📌 " if is_pinned else ""
        display_name = f"{pin_icon}🎬 {file_name}"
        
        item = QTreeWidgetItem(parent, [display_name])
        item.setData(0, Qt.UserRole, file_path)
        item.setData(0, Qt.UserRole + 1, is_pinned)
        item.setData(0, Qt.UserRole + 2, "file")
    
    def _refresh_lut_tree(self):
        """检查文件系统变化并刷新树"""
        try:
            current_mtime = self._get_dir_mtime(self.lut_base_dir)
            if current_mtime != self._last_lut_mtime:
                self._last_lut_mtime = current_mtime
                self._load_lut_tree()
        except:
            pass
    
    def _get_dir_mtime(self, dir_path):
        """递归获取目录的最后修改时间"""
        try:
            mtime = os.path.getmtime(dir_path)
            for root, dirs, files in os.walk(dir_path):
                for d in dirs:
                    if not d.startswith('.'):
                        mtime = max(mtime, os.path.getmtime(os.path.join(root, d)))
                for f in files:
                    if f.endswith('.cube'):
                        mtime = max(mtime, os.path.getmtime(os.path.join(root, f)))
            return mtime
        except:
            return 0

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
            self._load_lut_tree()

    @Slot()
    def on_delete_lut(self):
        """删除自定义目录中的 LUT"""
        item = self.lut_tree.currentItem()
        if not item:
            self.log("[警告] 请先选择要删除的 LUT")
            return

        path = item.data(0, Qt.UserRole)
        item_type = item.data(0, Qt.UserRole + 2)
        
        if not path or item_type != "file":
            self.log("[警告] 请选择一个LUT文件")
            return
            
        if not path.startswith(self.custom_lut_dir):
            self.log("[警告] 仅支持删除自定义目录中的 LUT")
            return

        try:
            os.remove(path)
            self.pinned_luts.discard(path)
            self._save_config()
            self.log(f"已删除 LUT: {os.path.basename(path)}")
            self._load_lut_tree()
        except Exception as e:
            self.log(f"[错误] 删除失败: {e}")

    @Slot(object)
    def on_lut_double_clicked(self, item, column):
        """双击加载LUT（仅文件）"""
        path = item.data(0, Qt.UserRole)
        item_type = item.data(0, Qt.UserRole + 2)
        
        # 只有文件才能加载
        if item_type != "file" or not path:
            return
        
        try:
            self.lut_table, self.lut_size = parse_cube_lut(path)
            self.log(f"已选择 LUT: {os.path.basename(path)} (尺寸: {self.lut_size}^3)")
            
            # 如果已加载图像，自动预览
            if self.source_image is not None:
                self._apply_lut_preview()
        except Exception as e:
            self.log(f"[错误] 加载 LUT 失败: {e}")
    
    @Slot(object)
    def on_lut_context_menu(self, position):
        """显示右键菜单"""
        item = self.lut_tree.itemAt(position)
        if not item:
            return
        
        path = item.data(0, Qt.UserRole)
        is_pinned = item.data(0, Qt.UserRole + 1)
        item_type = item.data(0, Qt.UserRole + 2)
        
        menu = QMenu(self)
        
        # 只有文件才能置顶
        if item_type == "file":
            if is_pinned:
                pin_action = QAction("取消置顶", self)
                pin_action.triggered.connect(lambda: self.on_unpin_lut(path))
            else:
                pin_action = QAction("📌 置顶", self)
                pin_action.triggered.connect(lambda: self.on_pin_lut(path))
            menu.addAction(pin_action)
            menu.addSeparator()
        
        # 检查是否在自定义目录
        is_custom = path and path.startswith(self.custom_lut_dir) if path else False
        
        # 重命名（仅自定义）
        if is_custom and item_type in ["file", "folder"]:
            rename_action = QAction("✏️ 重命名", self)
            if item_type == "file":
                rename_action.triggered.connect(lambda: self.on_rename_lut(path))
            else:
                rename_action.triggered.connect(lambda: self.on_rename_folder(path))
            menu.addAction(rename_action)
        
        # 删除（仅自定义）
        if is_custom and item_type in ["file", "folder"]:
            delete_action = QAction("🗑️ 删除", self)
            if item_type == "file":
                delete_action.triggered.connect(lambda: self.on_delete_lut_context(path))
            else:
                delete_action.triggered.connect(lambda: self.on_delete_folder(path))
            menu.addAction(delete_action)
        
        if not menu.isEmpty():
            menu.exec(self.lut_tree.viewport().mapToGlobal(position))
    
    def on_pin_lut(self, path):
        """置顶LUT"""
        self.pinned_luts.add(path)
        self._save_config()
        self._load_lut_tree()
        self.log(f"已置顶: {os.path.basename(path)}")
    
    def on_unpin_lut(self, path):
        """取消置顶LUT"""
        self.pinned_luts.discard(path)
        self._save_config()
        self._load_lut_tree()
        self.log(f"已取消置顶: {os.path.basename(path)}")
    
    def on_rename_lut(self, old_path):
        """重命名LUT"""
        old_name = os.path.basename(old_path)
        name_without_ext = os.path.splitext(old_name)[0]
        
        new_name, ok = QInputDialog.getText(
            self, "重命名 LUT", "输入新名称:",
            text=name_without_ext
        )
        
        if not ok or not new_name.strip():
            return
        
        new_name = new_name.strip()
        if not new_name.endswith('.cube'):
            new_name += '.cube'
        
        new_path = os.path.join(os.path.dirname(old_path), new_name)
        
        if os.path.exists(new_path):
            self.log(f"[错误] 文件名已存在: {new_name}")
            return
        
        try:
            os.rename(old_path, new_path)
            
            # 更新置顶列表中的路径
            if old_path in self.pinned_luts:
                self.pinned_luts.discard(old_path)
                self.pinned_luts.add(new_path)
                self._save_config()
            
            self._load_lut_tree()
            self.log(f"重命名成功: {old_name} → {new_name}")
        except Exception as e:
            self.log(f"[错误] 重命名失败: {e}")
    
    def on_rename_folder(self, old_path):
        """重命名文件夹"""
        old_name = os.path.basename(old_path)
        
        new_name, ok = QInputDialog.getText(
            self, "重命名文件夹", "输入新名称:",
            text=old_name
        )
        
        if not ok or not new_name.strip():
            return
        
        new_name = new_name.strip()
        new_path = os.path.join(os.path.dirname(old_path), new_name)
        
        if os.path.exists(new_path):
            self.log(f"[错误] 文件夹名已存在: {new_name}")
            return
        
        try:
            os.rename(old_path, new_path)
            
            # 更新置顶列表中所有受影响的路径
            updated_pinned = set()
            for pinned_path in self.pinned_luts:
                if pinned_path.startswith(old_path + os.sep):
                    new_pinned = pinned_path.replace(old_path, new_path, 1)
                    updated_pinned.add(new_pinned)
                else:
                    updated_pinned.add(pinned_path)
            self.pinned_luts = updated_pinned
            self._save_config()
            
            self._load_lut_tree()
            self.log(f"文件夹重命名成功: {old_name} → {new_name}")
        except Exception as e:
            self.log(f"[错误] 重命名失败: {e}")
    
    def on_rename_folder(self, old_path):
        """重命名文件夹"""
        old_name = os.path.basename(old_path)
        
        new_name, ok = QInputDialog.getText(
            self, "重命名文件夹", "输入新名称:",
            text=old_name
        )
        
        if not ok or not new_name.strip():
            return
        
        new_name = new_name.strip()
        new_path = os.path.join(os.path.dirname(old_path), new_name)
        
        if os.path.exists(new_path):
            self.log(f"[错误] 文件夹名已存在: {new_name}")
            return
        
        try:
            os.rename(old_path, new_path)
            
            # 更新置顶列表中所有受影响的路径
            updated_pinned = set()
            for pinned_path in self.pinned_luts:
                if pinned_path.startswith(old_path + os.sep):
                    new_pinned = pinned_path.replace(old_path, new_path, 1)
                    updated_pinned.add(new_pinned)
                else:
                    updated_pinned.add(pinned_path)
            self.pinned_luts = updated_pinned
            self._save_config()
            
            self._load_lut_tree()
            self.log(f"文件夹重命名成功: {old_name} → {new_name}")
        except Exception as e:
            self.log(f"[错误] 重命名失败: {e}")
    
    def on_delete_lut_context(self, path):
        """通过右键菜单删除LUT"""
        if not path.startswith(self.custom_lut_dir):
            self.log("[警告] 仅支持删除自定义目录中的 LUT")
            return
        
        try:
            os.remove(path)
            
            # 从置顶列表中移除
            self.pinned_luts.discard(path)
            self._save_config()
            
            self.log(f"已删除 LUT: {os.path.basename(path)}")
            self._load_lut_list()
        except Exception as e:
            self.log(f"[错误] 删除失败: {e}")    
    def on_delete_folder(self, path):
        """删除文件夹"""
        if not path.startswith(self.custom_lut_dir):
            self.log("[警告] 仅支持删除自定义目录中的文件夹")
            return
        
        try:
            shutil.rmtree(path)
            
            # 从置顶列表中移除所有相关路径
            self.pinned_luts = {p for p in self.pinned_luts if not p.startswith(path + os.sep)}
            self._save_config()
            
            self.log(f"已删除文件夹: {os.path.basename(path)}")
            self._load_lut_tree()
        except Exception as e:
            self.log(f"[错误] 删除文件夹失败: {e}")
    # ==================== 槽函数 (业务逻辑) ====================

    @Slot(int)
    def on_strength_changed(self, value):
        """强度滑块变化时实时预览"""
        self.lut_strength = value / 100.0
        self.lbl_strength_value.setText(f"{value}%")
        
        # 如果已加载图像和LUT，则实时预览
        if self.source_image is not None and self.lut_table is not None:
            self._apply_lut_preview(silent=True)
    
    @Slot(int)
    def on_dithering_changed(self, index):
        """抖动模式变化时更新设置并预览"""
        self.dithering_mode = self.dithering_combo.currentData()
        
        # 如果已加载图像和LUT，则实时预览
        if self.source_image is not None and self.lut_table is not None:
            self._apply_lut_preview(silent=True)
    
    def _apply_lut_preview(self, silent=False):
        """应用LUT到当前图像（实时预览）"""
        if hasattr(self, 'worker_thread') and self.worker_thread and self.worker_thread.isRunning():
            return  # 如果有线程正在运行，跳过

        # 捕获当前滑条强度，避免线程处理中途滑条变化导致不一致
        strength = self.lut_strength

        self.worker_thread = ImageProcessingThread(
            self.source_image, self.lut_table, self.lut_size, strength, self.dithering_mode
        )
        self.worker_thread.processing_finished.connect(
            lambda img, s=strength: self.on_preview_finished(img, silent, s)
        )
        self.worker_thread.processing_error.connect(self.on_process_error)
        self.worker_thread.start()

    def _ensure_preview_synced(self):
        """定时校验预览结果与滑条强度是否一致，不一致则触发预览"""
        # 未加载图片或LUT、强度控制未显示时跳过
        if not (self.source_image is not None and self.lut_table is not None):
            return
        if not self.strength_slider.isVisible():
            return

        # 正在处理时跳过，避免争抢线程
        if hasattr(self, 'worker_thread') and self.worker_thread and self.worker_thread.isRunning():
            return

        current_strength = self.lut_strength

        # 还未做过任何预览，或当前预览强度与滑条不一致时，触发一次静默预览
        if self.last_preview_strength is None or abs(current_strength - self.last_preview_strength) > 1e-4:
            self._apply_lut_preview(silent=True)

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
                self.last_preview_strength = None  # 重新加载图片后重置预览状态
                
                # 显示LUT强度滑块和抖动控制
                self.lbl_strength_title.setVisible(True)
                self.strength_slider.setVisible(True)
                self.lbl_strength_value.setVisible(True)
                self.lbl_dithering_title.setVisible(True)
                self.dithering_combo.setVisible(True)
                
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
                self.last_preview_strength = None  # 新 LUT 需重新预览
                
                # 如果已加载图像，自动预览
                if self.source_image is not None:
                    self._apply_lut_preview()
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
        strength = self.lut_strength
        self.worker_thread = ImageProcessingThread(
            self.source_image, self.lut_table, self.lut_size, strength, self.dithering_mode
        )
        self.worker_thread.processing_finished.connect(
            lambda img, s=strength: self.on_preview_finished(img, False, s)
        )
        self.worker_thread.processing_error.connect(self.on_process_error)
        self.worker_thread.start()
    
    @Slot(object)
    def on_preview_finished(self, result_image, silent=False, applied_strength=None):
        """预览完成"""
        self.processed_image = result_image
        self.lbl_result.set_image(self.processed_image)
        self.last_preview_strength = applied_strength if applied_strength is not None else self.lut_strength
        
        if not silent:
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
                self.image_paths, self.lut_table, self.lut_size, self.lut_strength, self.dithering_mode
            )
            self.worker_thread.progress_update.connect(self.on_batch_progress)
            self.worker_thread.processing_finished.connect(self.on_batch_finished)
            self.worker_thread.processing_error.connect(self.on_process_error)
            self.worker_thread.start()
        else:
            # 单张处理模式
            self.log("开始应用 3D LUT，请稍候...")
            self.worker_thread = ImageProcessingThread(
                self.source_image, self.lut_table, self.lut_size, self.lut_strength, self.dithering_mode
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
