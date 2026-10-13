# -*- coding: utf-8 -*-
"""数据形态预览: 投影(强度/负对数) + GT 三正交切片, 保存 PNG 到 data/。

用途: 阶段 0 确认灰度域与数据形态, 为几何反推做准备。

用法:
    python preview.py [时间戳]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import DATA_OUT, PROJ_N, VOL_N


def main(ts):
    proj = np.fromfile(f'{DATA_OUT}/projections_{ts}.bin', dtype='<f4').reshape(-1, PROJ_N, PROJ_N)
    gt = np.fromfile(f'{DATA_OUT}/gt_{ts}.bin', dtype='<f4').reshape(VOL_N, VOL_N, VOL_N)
    F = proj.shape[0]

    fr = proj[F // 2].astype(np.float64)
    I0 = np.percentile(fr, 99.9)                       # 空气区强度估计
    p = -np.log(np.clip(fr, 1, None) / I0)             # 负对数 → 线积分

    print(f'帧数={F}  I0(99.9分位)={I0:.0f}  负对数后范围=[{p.min():.2f}, {p.max():.2f}]')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(fr, cmap='gray', vmin=0, vmax=I0); axes[0, 0].set_title('投影 强度(空气亮)')
    axes[0, 1].imshow(p, cmap='gray'); axes[0, 1].set_title('投影 负对数(物体亮)')
    axes[0, 2].hist(fr.ravel(), bins=200); axes[0, 2].set_title('投影灰度直方图')
    axes[1, 0].imshow(gt[VOL_N // 2], cmap='gray'); axes[1, 0].set_title('GT 轴向 z=200')
    axes[1, 1].imshow(gt[:, VOL_N // 2, :], cmap='gray'); axes[1, 1].set_title('GT 冠状 y=200')
    axes[1, 2].imshow(gt[:, :, VOL_N // 2], cmap='gray'); axes[1, 2].set_title('GT 矢状 x=200')
    for ax in axes.ravel():
        ax.axis('off')

    out = f'{DATA_OUT}/preview_{ts}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=90)
    print(f'[write] {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '1094408')
