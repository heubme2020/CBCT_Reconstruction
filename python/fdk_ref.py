# -*- coding: utf-8 -*-
"""CPU 参考 FDK 实现(numpy 向量化)。

用途:
    1. 阶段 1 几何反推: 用小体积快速重建, 与 GT 对比, 迭代几何参数
    2. 阶段 2 正确性基准: CUDA 实现跑完后, 用本实现核对结果

FDK 三步(圆轨迹锥束):
    余弦加权 → 斜坡滤波(R-L) → 加权反投影
几何约定(源 S(β) 绕 z 轴旋转, 半径 R, 源-探测器距离 D):
    t = x·cosβ + y·sinβ   体素沿"源-原点"方向坐标(带符号)
    s = -x·sinβ + y·cosβ  体素垂直方向坐标
    U = R - t             源到体素距离(源在 t=R, 体素在 t, 距离 R-t)
    u_det = D·s/U, v_det = D·z/U  探测器平面坐标(mm, 物理尺度)
    det_pitch = 探测器物理像素间距(0.304mm)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.ndimage import map_coordinates


def to_line_integral(proj, I0=None, frac=0.999):
    """强度域投影 → 线积分(负对数)。proj: [F,H,W] float。"""
    if I0 is None:
        I0 = np.percentile(proj, frac * 100)
    return -np.log(np.clip(proj, 1.0, None) / I0)


def ramp_filter(proj_w, det_pitch):
    """斜坡滤波(R-L): 对每行 FFT × |ω| → IFFT。proj_w: [F,H,W]。"""
    W = proj_w.shape[2]
    freq = np.fft.fftfreq(W, d=det_pitch)          # 频率轴, 单位 1/mm
    ramp = np.abs(freq)
    spec = np.fft.fft(proj_w, axis=2)
    return np.fft.ifft(spec * ramp[None, None, :], axis=2).real


def fdk_recon(proj_li, angles, R, D, det_pitch, u0, v0, vol_size, vol_pitch):
    """完整 FDK 重建。

    proj_li: [F,H,W] 线积分投影(负对数后)
    angles:  [F] 投影角(弧度)
    R:       源-旋转中心距离(mm)   D: 源-探测器距离(mm)
    det_pitch: 探测器像素间距(mm)   u0,v0: 探测器中心(像素坐标)
    vol_size: 体数据边长(体素)      vol_pitch: 体素间距(mm)
    """
    F, H, W = proj_li.shape
    # 1. 余弦加权
    uu = (np.arange(W) - u0) * det_pitch
    vv = (np.arange(H) - v0) * det_pitch
    w = D / np.sqrt(D ** 2 + uu[None, :] ** 2 + vv[:, None] ** 2)
    proj_w = proj_li * w[None, :, :]

    # 2. 斜坡滤波
    proj_f = ramp_filter(proj_w, det_pitch)

    # 3. 加权反投影
    coords = (np.arange(vol_size) - (vol_size - 1) / 2.0) * vol_pitch
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing='ij')  # vol[i,j,k]=(x=xx,y=yy,z=zz)
    vol = np.zeros((vol_size,) * 3, np.float64)
    for i, beta in enumerate(angles):
        cb, sb = np.cos(beta), np.sin(beta)
        t = xx * cb + yy * sb
        s = -xx * sb + yy * cb
        U = R - t
        u_det = D * s / U
        v_det = D * zz / U
        u_px = u_det / det_pitch + u0
        v_px = v_det / det_pitch + v0
        samp = map_coordinates(proj_f[i], [v_px, u_px], order=1, mode='constant', cval=0.0)
        vol += samp * (R * R / (U * U))
    return vol.astype(np.float32)
