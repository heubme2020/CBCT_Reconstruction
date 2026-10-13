# -*- coding: utf-8 -*-
"""例子: 球体体模自洽验证 FDK 重建算法 (无需真实数据)。

生成球体体模 -> 锥束前向投影 -> FDK 重建 -> 评估相关性。
正确实现下球体自洽重建的相关系数应 > 0.9 (本实现约 0.96)。

用法:
    python generate_phantom.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
from scipy.ndimage import map_coordinates
from fdk_ref import fdk_recon, ramp_filter


def forward_project(vol, angles, R, D, det_pitch, u0, v0, det_H, det_W, vol_pitch):
    """锥束前向投影(splat, 最近邻)。vol: [N,N,N]。"""
    N = vol.shape[0]
    coords = (np.arange(N) - (N - 1) / 2) * vol_pitch
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing='ij')
    proj = np.zeros((len(angles), det_H, det_W))
    for i, beta in enumerate(angles):
        cb, sb = np.cos(beta), np.sin(beta)
        t = xx * cb + yy * sb
        s = -xx * sb + yy * cb
        U = R - t
        u_px = (D * s / U) / det_pitch + u0
        v_px = (D * zz / U) / det_pitch + v0
        ui = np.round(u_px).astype(int)
        vi = np.round(v_px).astype(int)
        valid = (ui >= 0) & (ui < det_W) & (vi >= 0) & (vi < det_H)
        np.add.at(proj[i], (vi[valid], ui[valid]), vol[valid])
    return proj


def main():
    # 几何 (与真实数据标定一致)
    R, D = 800.0, 1280.0       # M = D/R = 1.6
    det_pitch, det = 0.304, 978
    u0 = v0 = (det - 1) / 2 - 0
    N = 64
    vol_pitch = 200.0 / N

    # 球体体模 (偏离旋转中心, 更接近真实场景)
    vol = np.zeros((N,) * 3)
    c = (N - 1) / 2
    rr = np.sqrt((np.arange(N) - (c + 10))[:, None, None] ** 2 +
                 (np.arange(N) - (c + 6))[None, :, None] ** 2 +
                 (np.arange(N) - c)[None, None, :] ** 2)
    vol[rr < 16] = 1.0

    # 200° 短扫描前向投影
    angles = np.deg2rad(np.linspace(0, 200, 96, endpoint=False))
    proj = forward_project(vol, angles, R, D, det_pitch, u0, v0, det, det, vol_pitch)

    # FDK 重建 (含斜坡滤波)
    recon = fdk_recon(proj, angles, R, D, det_pitch, u0, v0, N, vol_pitch)

    # 评估相关系数
    a = recon.ravel() - recon.mean()
    b = vol.ravel() - vol.mean()
    corr = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
    print(f'球体体模自洽重建: 相关系数 corr = {corr:.4f}  (正确实现应 > 0.9)')
    assert corr > 0.8, '重建结果异常, 检查几何/滤波实现'
    print('验证通过 [OK]')


if __name__ == '__main__':
    main()
