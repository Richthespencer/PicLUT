# PicLUT

**PicLUT** 是一个基于 Python 的图像色彩处理工具，可以方便地将 3D LUT（Lookup Table）应用到图片上，实现电影级的色彩风格调整。项目采用 PySide6 构建图形界面，并使用 Pillow 和 OpenCV 进行高效图像处理。

---

## 功能特点

- 支持导入各种图片格式：PNG、JPEG、BMP、TIFF、WebP 等。
- 支持加载 `.cube` 格式的 3D LUT 文件。
- 实时预览处理前后的图像对比。
- 后台线程处理，避免界面卡顿。
- 支持将处理结果导出为 PNG、JPEG、TIFF 等格式。
- 暗色系 UI 风格，界面简洁美观。
- 自适应图像缩放控件，可根据窗口大小自动调整显示。

---

## 安装

1. 克隆或下载本项目：

```bash
git clone https://github.com/Richthespencer/PicLUT.git
cd PicLUT
````

2. 安装依赖：

```bash
pip install -r requirements.txt
```

> 依赖库主要包括：
>
> * PySide6
> * OpenCV (`opencv-python`)
> * Pillow
> * NumPy

如果没有 `requirements.txt`，可以手动安装：

```bash
pip install PySide6 opencv-python Pillow numpy
```

---

## 使用说明

1. 运行程序：

```bash
python PicLUT.py
```

2. 主界面操作流程：

   1. 点击 **打开图片** 选择待处理的图像。
   2. 点击 **导入 LUT (.cube)** 选择 3D LUT 文件。
   3. 点击 **应用处理**，程序会在后台线程应用 LUT。
   4. 处理完成后，点击 **导出结果** 保存图像。

3. 日志区域会显示操作信息和错误提示。

---

## 测试用例说明（Test Case）

本项目提供了一个用于验证 3D LUT 正确性的测试用例，说明如下：

### 测试输入

- **`test.png`**
   一张处于 **Apple Log 2（Apple Log 2 Color Space）** 色彩空间下的测试图像。
   该图像用于模拟来自 Apple 设备（如 iPhone / Apple Log 工作流）的 Log 素材输入。

- **`LUT/` 文件夹**
   该目录中包含多个 `.cube` 格式的 **3D LUT 文件**，其功能是：

  > **将 Apple Log 2 色彩空间映射到富士胶片模拟风格的 FLog2C 色彩空间**

  这些 LUT 可用于模拟富士相机胶片风格（Film Simulation）的色彩表现。

---

## 注意事项

* `.cube` 文件必须是标准 3D LUT 文件，程序会检查数据点数量与 LUT 尺寸是否匹配。
* 图像处理过程中可能消耗较多 CPU，推荐中高性能机器处理大尺寸图片。
* 支持处理带透明通道的图像，但 Alpha 通道会被移除，仅保留 RGB。

---

## 未来计划

* 支持批量处理图片。