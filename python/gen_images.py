# -*- coding: utf-8 -*-
"""生成 README 展示用的结果图片到 ../images/。

生成:
    images/slices_compare.png   三正交切片(重建 vs GT 并排)
    images/render_3d.png        三维骨骼等值面(VTK)
用法:
    python gen_images.py [时间戳]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import DATA_OUT, VOL_N

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(OUT_DIR, exist_ok=True)


def _norm(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def gen_slices(ts):
    vol = np.fromfile(f'{DATA_OUT}/volume_{ts}.bin', dtype='<f4').reshape(VOL_N, VOL_N, VOL_N)
    gt = np.fromfile(f'{DATA_OUT}/gt_{ts}.bin', dtype='<f4').reshape(VOL_N, VOL_N, VOL_N)
    k = VOL_N // 2
    sl = [('Axial', vol[k], gt[k]), ('Coronal', vol[:, k, :], gt[:, k, :]), ('Sagittal', vol[:, :, k], gt[:, :, k])]
    fig, axes = plt.subplots(3, 2, figsize=(7, 10))
    for r, (name, rv, gv) in enumerate(sl):
        axes[r, 0].imshow(_norm(rv), cmap='gray')
        axes[r, 0].set_title(f'Reconstruction — {name}')
        axes[r, 1].imshow(_norm(gv), cmap='gray')
        axes[r, 1].set_title(f'Ground truth — {name}')
        for c in range(2):
            axes[r, c].axis('off')
    plt.tight_layout()
    out = f'{OUT_DIR}/slices_compare.png'
    plt.savefig(out, dpi=110)
    plt.close()
    print(f'[write] {out}')


def gen_render_3d(ts):
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)

    from vtk.util import numpy_support
    vol = np.fromfile(f'{DATA_OUT}/volume_{ts}.bin', dtype='<f4').reshape(VOL_N, VOL_N, VOL_N)
    vol = _norm(vol)
    img = vtk.vtkImageData()
    img.SetDimensions(VOL_N, VOL_N, VOL_N)
    img.SetSpacing(0.5, 0.5, 0.5)
    img.AllocateScalars(vtk.VTK_FLOAT, 1)
    vtk_arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(vol.ravel(), dtype=np.float32),
                                        deep=True, array_type=vtk.VTK_FLOAT)
    img.GetPointData().SetScalars(vtk_arr)

    contour = vtk.vtkContourFilter()
    contour.SetInputData(img)
    contour.SetValue(0, 0.7)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 0.9, 0.75)
    actor.GetProperty().SetSpecular(0.4)
    actor.GetProperty().SetSpecularPower(30)

    ren = vtk.vtkRenderer()
    ren.SetBackground(0.05, 0.05, 0.08)
    ren.AddActor(actor)
    light = vtk.vtkLight()
    light.SetPosition(1, 1, 1)
    light.SetFocalPoint(0, 0, 0)
    light.SetIntensity(1.0)
    ren.AddLight(light)
    ren.ResetCamera()

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(700, 700)
    rw.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.Update()
    wr = vtk.vtkPNGWriter()
    out = f'{OUT_DIR}/render_3d.png'
    wr.SetFileName(out)
    wr.SetInputConnection(w2i.GetOutputPort())
    wr.Write()
    print(f'[write] {out}')


if __name__ == '__main__':
    ts = sys.argv[1] if len(sys.argv) > 1 else '1094408'
    gen_slices(ts)
    gen_render_3d(ts)
