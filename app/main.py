# -*- coding: utf-8 -*-
"""CBCT 三维重建结果查看器 (PyQt5 + VTK 桌面软件)。

功能:
    - 三维体渲染 (GPU ray casting, 骨骼 + 软组织传递函数)
    - 三正交切片 (轴向/冠状/矢状 MPR, 位置联动)
    - 重建 vs GT 切换
    - 窗宽窗位调节 / 切片滑块

运行:
    python main.py [时间戳]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import vtk
from vtk.util import numpy_support
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QSlider, QLabel, QPushButton, QComboBox, QFrame)
from PyQt5.QtCore import Qt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
VOL_N = 400  # 体数据边长


def get_data_dir():
    """定位数据目录: 优先 exe/脚本 同级的 data/, 其次 ../data/。"""
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    for d in [os.path.join(base, 'data'), os.path.join(base, '..', 'data')]:
        if os.path.exists(os.path.join(d, 'volume_1094408.bin')):
            return d
    return os.path.join(base, '..', 'data')


def load_volume(path):
    """读体数据并归一化到 [0,1]。"""
    arr = np.fromfile(path, dtype='<f4').reshape(VOL_N, VOL_N, VOL_N).astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)
    return np.ascontiguousarray(arr)


def numpy_to_vtk(arr):
    """numpy [z,y,x] float32 -> vtkImageData。"""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    N = arr.shape[0]
    img = vtk.vtkImageData()
    img.SetDimensions(N, N, N)
    img.SetSpacing(0.5, 0.5, 0.5)
    img.AllocateScalars(vtk.VTK_FLOAT, 1)
    vtk_arr = numpy_support.numpy_to_vtk(arr.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    img.GetPointData().SetScalars(vtk_arr)
    return img


class MPRView:
    """单个正交切片视图 (轴向/冠状/矢状)。"""

    def __init__(self, image_data, orientation):
        self.reslice = vtk.vtkImageReslice()
        self.reslice.SetInputData(image_data)
        self.reslice.SetOutputDimensionality(2)
        self.reslice.SetInterpolationModeToLinear()
        # 方向余弦: 轴向(固定z) / 冠状(固定y) / 矢状(固定x)
        if orientation == 'axial':
            self.axes = (1, 0, 0, 0, 1, 0, 0, 0, 1)   # 切面法向 z
            self.reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 1, 0, 0, 0, 1)
        elif orientation == 'coronal':
            self.reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 0, 1, 0, 1, 0)  # 法向 y
        else:  # sagittal
            self.reslice.SetResliceAxesDirectionCosines(0, 1, 0, 0, 0, 1, 1, 0, 0)  # 法向 x
        self.orientation = orientation
        self.N = image_data.GetDimensions()[0]
        self.set_slice(self.N // 2)

    def set_slice(self, idx):
        origin = list(self.reslice.GetResliceAxesOrigin())
        o = idx * 0.5
        if self.orientation == 'axial':
            self.reslice.SetResliceAxesOrigin(0, 0, o)
        elif self.orientation == 'coronal':
            self.reslice.SetResliceAxesOrigin(0, o, 0)
        else:
            self.reslice.SetResliceAxesOrigin(o, 0, 0)
        self.reslice.Update()


class CBCTViewer(QMainWindow):
    def __init__(self, ts='1094408'):
        super().__init__()
        self.setWindowTitle(f'CBCT 三维重建查看器 — {ts}')
        self.setGeometry(80, 60, 1500, 950)

        # 数据
        data_dir = get_data_dir()
        self.vol_recon = load_volume(f'{data_dir}/volume_{ts}.bin')
        self.vol_gt = load_volume(f'{data_dir}/gt_{ts}.bin')
        self.img_recon = numpy_to_vtk(self.vol_recon)
        self.img_gt = numpy_to_vtk(self.vol_gt)
        self.current_img = self.img_recon
        self.mpr_views = {}

        self.build_ui()

    # ---------- UI ----------
    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 工具栏
        bar = QHBoxLayout()
        bar.addWidget(QLabel('显示:'))
        self.combo_data = QComboBox()
        self.combo_data.addItems(['重建', 'GT'])
        self.combo_data.currentIndexChanged.connect(self.on_switch_data)
        bar.addWidget(self.combo_data)
        bar.addSpacing(20)
        bar.addWidget(QLabel('窗宽窗位:'))
        self.btn_bone = QPushButton('骨骼窗')
        self.btn_bone.clicked.connect(lambda: self.set_window(0.3, 0.7))
        self.btn_soft = QPushButton('软组织窗')
        self.btn_soft.clicked.connect(lambda: self.set_window(0.0, 1.0))
        self.btn_reset = QPushButton('重置')
        self.btn_reset.clicked.connect(self.on_reset)
        bar.addWidget(self.btn_bone)
        bar.addWidget(self.btn_soft)
        bar.addWidget(self.btn_reset)
        self.btn_volume = QPushButton('体渲染')
        self.btn_volume.setCheckable(True)
        self.btn_volume.clicked.connect(self.on_toggle_volume)
        bar.addWidget(self.btn_volume)
        bar.addStretch()
        bar.addWidget(QLabel('切片位置:'))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, VOL_N - 1)
        self.slider.setValue(VOL_N // 2)
        self.slider.setFixedWidth(300)
        self.slider.valueChanged.connect(self.on_slice)
        bar.addWidget(self.slider)
        root.addLayout(bar)

        # 视图网格: 左上轴向, 右上冠状, 左下矢状, 右下 3D 体渲染
        grid = QGridLayout()
        self.vtk_axial = QVTKRenderWindowInteractor(self)
        self.vtk_coronal = QVTKRenderWindowInteractor(self)
        self.vtk_sagittal = QVTKRenderWindowInteractor(self)
        self.vtk_3d = QVTKRenderWindowInteractor(self)
        for i, w in enumerate([self.vtk_axial, self.vtk_coronal, self.vtk_sagittal, self.vtk_3d]):
            grid.addWidget(w, i // 2, i % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        root.addLayout(grid, 1)

        self.setup_mpr(self.vtk_axial, 'axial')
        self.setup_mpr(self.vtk_coronal, 'coronal')
        self.setup_mpr(self.vtk_sagittal, 'sagittal')
        self.setup_volume_3d(self.vtk_3d)

    def setup_mpr(self, widget, orientation):
        ren = vtk.vtkRenderer()
        ren.SetBackground(0.1, 0.1, 0.12)
        widget.GetRenderWindow().AddRenderer(ren)
        self.mpr_views[orientation] = MPRView(self.current_img, orientation)
        actor = vtk.vtkImageActor()
        actor.GetMapper().SetInputConnection(self.mpr_views[orientation].reslice.GetOutputPort())
        actor.GetProperty().SetColorWindow(1.0)
        actor.GetProperty().SetColorLevel(0.5)
        ren.AddActor(actor)
        ren.ResetCamera()
        widget.GetRenderWindow().Render()

    def setup_volume_3d(self, widget):
        self.ren_3d = vtk.vtkRenderer()
        self.ren_3d.SetBackground(0.05, 0.05, 0.08)
        widget.GetRenderWindow().AddRenderer(self.ren_3d)

        self.volume_mapper = vtk.vtkSmartVolumeMapper()
        self.volume_mapper.SetInputData(self.current_img)
        self.volume_mapper.SetRequestedRenderModeToGPU()  # GPU ray casting (RTX 4090)

        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()

        # 不透明度传递函数 (软组织透明, 骨骼不透明)
        self.opacity_tf = vtk.vtkPiecewiseFunction()
        self.opacity_tf.AddPoint(0.0, 0.0)
        self.opacity_tf.AddPoint(0.25, 0.03)
        self.opacity_tf.AddPoint(0.55, 0.12)
        self.opacity_tf.AddPoint(0.8, 0.5)
        self.opacity_tf.AddPoint(1.0, 1.0)
        self.volume_property.SetScalarOpacity(self.opacity_tf)

        # 颜色传递函数 (骨骼暖色)
        self.color_tf = vtk.vtkColorTransferFunction()
        self.color_tf.AddRGBPoint(0.0, 0.1, 0.1, 0.1)
        self.color_tf.AddRGBPoint(0.4, 0.6, 0.55, 0.5)
        self.color_tf.AddRGBPoint(0.8, 1.0, 0.95, 0.85)
        self.color_tf.AddRGBPoint(1.0, 1.0, 1.0, 1.0)
        self.volume_property.SetColor(self.color_tf)

        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volume_mapper)
        self.volume.SetProperty(self.volume_property)
        self.ren_3d.AddVolume(self.volume)

        # 骨骼等值面 (不依赖 GPU ray casting, 可靠显示骨骼三维结构)
        self.contour = vtk.vtkContourFilter()
        self.contour.SetInputData(self.current_img)
        self.contour.SetValue(0, 0.7)
        self.contour_mapper = vtk.vtkPolyDataMapper()
        self.contour_mapper.SetInputConnection(self.contour.GetOutputPort())
        self.contour_mapper.ScalarVisibilityOff()
        self.contour_actor = vtk.vtkActor()
        self.contour_actor.SetMapper(self.contour_mapper)
        self.contour_actor.GetProperty().SetColor(1.0, 0.9, 0.75)
        self.contour_actor.GetProperty().SetOpacity(1.0)
        self.contour_actor.GetProperty().SetSpecular(0.4)
        self.contour_actor.GetProperty().SetSpecularPower(30)
        self.ren_3d.AddActor(self.contour_actor)

        # 光照 (让骨骼有立体感)
        self.light = vtk.vtkLight()
        self.light.SetPosition(1, 1, 1)
        self.light.SetFocalPoint(0, 0, 0)
        self.light.SetIntensity(1.0)
        self.ren_3d.AddLight(self.light)

        # 体渲染默认关闭(CPU FixedPoint 质量差且遮挡等值面), 可通过工具栏开启
        self.volume.SetVisibility(0)

        self.ren_3d.ResetCamera()
        widget.GetRenderWindow().Render()

    # ---------- 交互 ----------
    def on_switch_data(self, idx):
        self.current_img = self.img_recon if idx == 0 else self.img_gt
        for orientation, mpr in self.mpr_views.items():
            mpr.reslice.SetInputData(self.current_img)
            mpr.set_slice(self.slider.value())
        self.volume_mapper.SetInputData(self.current_img)
        self.contour.SetInputData(self.current_img)
        self.refresh()

    def on_slice(self, idx):
        for mpr in self.mpr_views.values():
            mpr.set_slice(idx)
        self.refresh_mpr()

    def set_window(self, lo, hi):
        self.opacity_tf.RemoveAllPoints()
        self.opacity_tf.AddPoint(lo, 0.0)
        self.opacity_tf.AddPoint((lo + hi) / 2, 0.3)
        self.opacity_tf.AddPoint(hi, 1.0)
        self.volume_property.SetScalarOpacity(self.opacity_tf)
        self.refresh_3d()

    def on_reset(self):
        self.set_window(0.0, 1.0)

    def on_toggle_volume(self, checked):
        self.volume.SetVisibility(1 if checked else 0)
        self.refresh_3d()

    def refresh(self):
        self.refresh_mpr()
        self.refresh_3d()

    def refresh_mpr(self):
        for w in [self.vtk_axial, self.vtk_coronal, self.vtk_sagittal]:
            w.GetRenderWindow().Render()

    def refresh_3d(self):
        self.vtk_3d.GetRenderWindow().Render()


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    ts = sys.argv[1] if len(sys.argv) > 1 else '1094408'
    win = CBCTViewer(ts)
    win.show()
    for w in [win.vtk_axial, win.vtk_coronal, win.vtk_sagittal, win.vtk_3d]:
        w.Initialize()
    sys.exit(app.exec_())
