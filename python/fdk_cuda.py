# -*- coding: utf-8 -*-
"""FDK 重建 CUDA 加速核心 (numba.cuda, 无需 MSVC)。

反投影(计算量最大的步骤)在 GPU 上并行; 加权/斜坡滤波在 host 用 numpy。
几何约定与 fdk_ref.py 完全一致。

用法:
    from fdk_cuda import fdk_cuda
    vol = fdk_cuda(proj_li, angles_rad, R, D, det_pitch, u0, v0, vol_size, vol_pitch, filter=0)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import numpy as np
from numba import cuda

from fdk_ref import ramp_filter  # 斜坡滤波复用 CPU 参考实现(仅 F×H×W 行 FFT, 代价小)


def ramp_filter_pad(proj_w, pad):
    """截断外推 + 斜坡滤波: 投影 u 方向左右各外推 pad 像素(线性衰减到0), 补零后滤波。

    用于横向截断(FOV 小于物体)场景, 避免斜坡滤波的 circular wrap-around 伪影。
    proj_w: [F,H,W] 余弦加权后的投影。
    """
    F, H, W = proj_w.shape
    Wp = W + 2 * pad
    out = np.zeros((F, H, Wp), dtype=np.float64)
    out[:, :, pad:pad + W] = proj_w
    w = np.arange(pad, dtype=np.float64) / pad  # [0, 1/pad, ..., (pad-1)/pad]
    out[:, :, :pad] = proj_w[:, :, :1] * w[None, None, :]          # 左外推: 近边界大, 远处衰减到0
    out[:, :, pad + W:] = proj_w[:, :, -1:] * w[::-1][None, None, :]  # 右外推: 近边界大, 远处衰减到0
    ramp = np.abs(np.fft.fftfreq(Wp))
    spec = np.fft.fft(out, axis=2)
    f = np.fft.ifft(spec * ramp[None, None, :], axis=2).real
    return f[:, :, pad:pad + W]


@cuda.jit
def _backproject_kernel(proj, cosb, sinb, vol, R, D, det_pitch, u0, v0, vol_pitch, N):
    """每个线程重建一个体素 (ix, iy, iz), 遍历所有投影帧累加。"""
    ix, iy, iz = cuda.grid(3)
    if ix >= N or iy >= N or iz >= N:
        return
    F = proj.shape[0]
    H = proj.shape[1]
    W = proj.shape[2]
    x = (ix - (N - 1) / 2.0) * vol_pitch
    y = (iy - (N - 1) / 2.0) * vol_pitch
    z = (iz - (N - 1) / 2.0) * vol_pitch
    acc = 0.0
    for f in range(F):
        cb = cosb[f]
        sb = sinb[f]
        t = x * cb + y * sb
        s = -x * sb + y * cb
        U = R - t
        u_det = D * s / U
        v_det = D * z / U
        u_px = u_det / det_pitch + u0
        v_px = v_det / det_pitch + v0
        ui = int(math.floor(u_px))
        vi = int(math.floor(v_px))
        du = u_px - ui
        dv = v_px - vi
        val = 0.0
        if ui >= 0 and ui + 1 < W and vi >= 0 and vi + 1 < H:
            v00 = proj[f, vi, ui]
            v01 = proj[f, vi, ui + 1]
            v10 = proj[f, vi + 1, ui]
            v11 = proj[f, vi + 1, ui + 1]
            val = v00 * (1 - du) * (1 - dv) + v01 * du * (1 - dv) + v10 * (1 - du) * dv + v11 * du * dv
        acc += val * (R * R / (U * U))
    vol[iz, iy, ix] = acc


def fdk_cuda(proj_li, angles_rad, R, D, det_pitch, u0, v0, vol_size, vol_pitch, filter=0):
    """完整 FDK。proj_li: [F,H,W] 线积分(负对数后); angles_rad: [F] 弧度。"""
    F, H, W = proj_li.shape
    proj_li = np.ascontiguousarray(proj_li, dtype=np.float32)

    # 1. 余弦加权 (host)
    uu = (np.arange(W) - u0) * det_pitch
    vv = (np.arange(H) - v0) * det_pitch
    w = D / np.sqrt(D ** 2 + uu[None, :] ** 2 + vv[:, None] ** 2)
    proj_w = (proj_li * w[None, :, :]).astype(np.float32)

    # 2. 斜坡滤波 (可选: 0=无, 1=R-L, 2=截断外推+R-L)
    if filter == 1:
        proj_w = ramp_filter(proj_w.astype(np.float64), det_pitch).astype(np.float32)
    elif filter == 2:
        proj_w = ramp_filter_pad(proj_w.astype(np.float64), pad=300).astype(np.float32)

    # 3. 反投影 (GPU)
    cosb = np.cos(angles_rad).astype(np.float32)
    sinb = np.sin(angles_rad).astype(np.float32)
    d_proj = cuda.to_device(proj_w)
    d_cos = cuda.to_device(cosb)
    d_sin = cuda.to_device(sinb)
    d_vol = cuda.device_array((vol_size, vol_size, vol_size), dtype=np.float32)

    threads = (8, 8, 8)
    blocks = ((vol_size + 7) // 8, (vol_size + 7) // 8, (vol_size + 7) // 8)
    _backproject_kernel[blocks, threads](
        d_proj, d_cos, d_sin, d_vol,
        np.float32(R), np.float32(D), np.float32(det_pitch),
        np.float32(u0), np.float32(v0), np.float32(vol_pitch), np.int32(vol_size))
    cuda.synchronize()
    return d_vol.copy_to_host()


if __name__ == '__main__':
    # 快速自检: 读投影, 负对数, 小体积重建
    ts = sys.argv[1] if len(sys.argv) > 1 else '1094408'
    from config import DATA_OUT, PROJ_N, VOL_N
    from fdk_ref import to_line_integral
    proj = np.fromfile(f'{DATA_OUT}/projections_{ts}.bin', dtype='<f4').reshape(-1, PROJ_N, PROJ_N)
    pli = to_line_integral(proj)
    F = pli.shape[0]
    R, M = 800.0, 1.5
    angles = np.deg2rad(np.linspace(0, 200, F, endpoint=False))
    vol = fdk_cuda(pli, angles, R, R * M, 0.304, (PROJ_N - 1) / 2 - 34, (PROJ_N - 1) / 2, 200, 1.0)
    print(f'[fdk_cuda] 重建 {vol.shape} 范围[{vol.min():.1f},{vol.max():.1f}]')
