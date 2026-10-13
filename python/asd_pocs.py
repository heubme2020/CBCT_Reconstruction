# -*- coding: utf-8 -*-
"""ASD-POCS 迭代重建 (SART + TV 最小化, numba.cuda)。

处理横向截断: 数据一致性(SART)只约束探测器内的投影, 探测器外的缺失数据
由 TV 正则化(图像的梯度稀疏)填补。

算法(论文 P1 5.2.2/5.3, Sidky & Pan 2008):
    主循环: SART(数据一致性) + 非负约束 -> TV 最小化(最速下降) -> 更新松弛因子
    几何与 fdk_cuda 一致。

用法:
    from asd_pocs import asd_pocs
    vol = asd_pocs(g, angles, R, D, det_pitch, u0, v0, vol_size, vol_pitch, ...)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import numpy as np
from numba import cuda


@cuda.jit
def _forward_kernel(vol, proj, cosb, sinb, R, D, det_pitch, u0, v0, vol_pitch, N):
    """前向投影(splat): 每个体素投影到所有角度, 双线性 + 原子加。"""
    ix, iy, iz = cuda.grid(3)
    if ix >= N or iy >= N or iz >= N:
        return
    F = proj.shape[0]
    H = proj.shape[1]
    W = proj.shape[2]
    x = (ix - (N - 1) / 2.0) * vol_pitch
    y = (iy - (N - 1) / 2.0) * vol_pitch
    z = (iz - (N - 1) / 2.0) * vol_pitch
    val = vol[iz, iy, ix]
    if val == 0.0:
        return
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
        if ui >= 0 and ui + 1 < W and vi >= 0 and vi + 1 < H:
            cuda.atomic.add(proj, (f, vi, ui), val * (1 - du) * (1 - dv))
            cuda.atomic.add(proj, (f, vi, ui + 1), val * du * (1 - dv))
            cuda.atomic.add(proj, (f, vi + 1, ui), val * (1 - du) * dv)
            cuda.atomic.add(proj, (f, vi + 1, ui + 1), val * du * dv)


@cuda.jit
def _backproject_kernel(proj, vol, cosb, sinb, R, D, det_pitch, u0, v0, vol_pitch, N):
    """反投影(采样): 每个体素从所有角度双线性采样累加。"""
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
        if ui >= 0 and ui + 1 < W and vi >= 0 and vi + 1 < H:
            v00 = proj[f, vi, ui]
            v01 = proj[f, vi, ui + 1]
            v10 = proj[f, vi + 1, ui]
            v11 = proj[f, vi + 1, ui + 1]
            acc += v00 * (1 - du) * (1 - dv) + v01 * du * (1 - dv) + v10 * (1 - du) * dv + v11 * du * dv
    vol[iz, iy, ix] = acc


@cuda.jit
def _tv_gradient_kernel(vol, grad, N):
    """3D 各向异性 TV 梯度: TV=Σ|f_x|+|f_y|+|f_z|。

    ∂TV/∂f_i = -sign(f_{i+1}-f_i) + sign(f_i-f_{i-1})  (y,z 同理)
    """
    ix, iy, iz = cuda.grid(3)
    if ix >= N or iy >= N or iz >= N:
        return
    v = vol[iz, iy, ix]
    g = 0.0
    if ix + 1 < N:
        d = vol[iz, iy, ix + 1] - v
        g -= 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
    if ix - 1 >= 0:
        d = v - vol[iz, iy, ix - 1]
        g += 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
    if iy + 1 < N:
        d = vol[iz, iy + 1, ix] - v
        g -= 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
    if iy - 1 >= 0:
        d = v - vol[iz, iy - 1, ix]
        g += 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
    if iz + 1 < N:
        d = vol[iz + 1, iy, ix] - v
        g -= 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
    if iz - 1 >= 0:
        d = v - vol[iz - 1, iy, ix]
        g += 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
    grad[iz, iy, ix] = g


def _launch(kernel, N, *args):
    threads = (8, 8, 8)
    blocks = ((N + 7) // 8, (N + 7) // 8, (N + 7) // 8)
    kernel[blocks, threads](*args)


def asd_pocs(g, angles_rad, R, D, det_pitch, u0, v0, vol_size, vol_pitch,
             n_iter=30, n_tv=10, beta0=0.5, beta_red=0.99, alpha0=0.2, alpha_red=0.95,
             f_init=None, verbose=True):
    """ASD-POCS 迭代重建。

    g: [F,H,W] 线积分投影(负对数后); angles_rad: [F] 弧度。
    返回 [N,N,N] float32 重建体数据。
    """
    F, H, W = g.shape
    N = vol_size
    g = np.ascontiguousarray(g, dtype=np.float32)
    cosb = np.cos(angles_rad).astype(np.float32)
    sinb = np.sin(angles_rad).astype(np.float32)

    d_g = cuda.to_device(g)
    d_cos = cuda.to_device(cosb)
    d_sin = cuda.to_device(sinb)

    # 初始化: FDK 结果(若给) 或 0
    if f_init is not None:
        f = np.ascontiguousarray(f_init, dtype=np.float32)
    else:
        f = np.zeros((N, N, N), dtype=np.float32)
    d_f = cuda.to_device(f)
    d_proj = cuda.device_array((F, H, W), dtype=np.float32)
    d_grad = cuda.device_array((N, N, N), dtype=np.float32)

    beta = beta0
    alpha = alpha0
    for n in range(n_iter):
        # 1. 前向投影
        d_proj.copy_to_device(np.zeros((F, H, W), np.float32))
        _launch(_forward_kernel, N, d_f, d_proj, d_cos, d_sin,
                np.float32(R), np.float32(D), np.float32(det_pitch),
                np.float32(u0), np.float32(v0), np.float32(vol_pitch), np.int32(N))
        cuda.synchronize()
        # 2. 误差 + 反投影误差 (SART 归一化: 每体素被 F 条射线穿过)
        err = (g - d_proj.copy_to_host()).astype(np.float32)
        d_err = cuda.to_device(err)
        d_update = cuda.device_array((N, N, N), dtype=np.float32)
        _launch(_backproject_kernel, N, d_err, d_update, d_cos, d_sin,
                np.float32(R), np.float32(D), np.float32(det_pitch),
                np.float32(u0), np.float32(v0), np.float32(vol_pitch), np.int32(N))
        cuda.synchronize()
        # 3. SART 更新 + 非负约束
        f_prev = f
        f = f + (beta / F) * d_update.copy_to_host()
        f = np.maximum(f, 0.0)
        d_f = cuda.to_device(f)
        # 4. TV 最小化 (自适应步长 dα = α·d_pocs, 论文 5.2.2)
        if n_tv > 0:
            d_pocs = float(np.sqrt(((f - f_prev) ** 2).sum()))
            for k in range(n_tv):
                d_f = cuda.to_device(f)
                _launch(_tv_gradient_kernel, N, d_f, d_grad, np.int32(N))
                cuda.synchronize()
                grad = d_grad.copy_to_host()
                norm = float(np.sqrt((grad ** 2).sum()) + 1e-9)
                f = f - (alpha * d_pocs) * (grad / norm)
            f = np.maximum(f, 0.0)
        beta *= beta_red
        alpha *= alpha_red
        if verbose and (n % 5 == 0 or n == n_iter - 1):
            print(f'  iter {n+1}/{n_iter}: f范围[{f.min():.2f},{f.max():.2f}] β={beta:.4f} α={alpha:.4f}')
    return f.astype(np.float32)


if __name__ == '__main__':
    ts = sys.argv[1] if len(sys.argv) > 1 else '1094408'
    from config import DATA_OUT, PROJ_N, VOL_N
    from fdk_ref import to_line_integral
    from fdk_cuda import fdk_cuda
    proj = np.fromfile(f'{DATA_OUT}/projections_{ts}.bin', dtype='<f4').reshape(-1, PROJ_N, PROJ_N)
    pli = to_line_integral(proj)[::2]
    F = pli.shape[0]
    R, M = 800.0, 1.6
    c0 = (PROJ_N - 1) / 2
    angles = np.deg2rad(np.linspace(0, 200, F, endpoint=False))
    # FDK 初始化 + ASD-POCS 小体积验证
    f0 = fdk_cuda(pli, angles, R, R * M, 0.304, c0 - 24, c0, 128, 200 / 128, filter=2)
    vol = asd_pocs(pli, angles, R, R * M, 0.304, c0 - 24, c0, 128, 200 / 128, f_init=f0)
    print(f'[asd_pocs] 重建 {vol.shape} 范围[{vol.min():.2f},{vol.max():.2f}]')
