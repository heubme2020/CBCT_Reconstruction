# -*- coding: utf-8 -*-
"""评估 + 可视化: 重建体数据 vs GT 的定量指标(d/r/e) + 三正交切片 + MIP。

用法:
    python evaluate.py [时间戳]
输出:
    data/eval_<ts>.png   三正交切片并排对比 + MIP
    控制台: d/r/e 指标 + 相关系数
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import DATA_OUT, VOL_N

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def metrics(recon, gt):
    """论文 1.4.3 三个指标 (min-max 归一化到 [0,1] 后计算, 尺度无关)。"""
    t = (recon - recon.min()) / (recon.max() - recon.min() + 1e-9)
    r = (gt - gt.min()) / (gt.max() - gt.min() + 1e-9)
    t = t.ravel().astype(np.float64)
    r = r.ravel().astype(np.float64)
    d = np.sqrt(((t - r) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    r_metric = np.abs(t - r).sum() / (np.abs(t).sum() + 1e-9)
    e = np.abs(t - r).max() / (t.max() - t.min() + 1e-9)
    return d, r_metric, e


def dice(a, b, frac=0.15):
    """高值(骨骼)体素的重叠系数 Dice = 2|A∩B| / (|A|+|B|)。"""
    ta = np.percentile(a, 100 * (1 - frac))
    tb = np.percentile(b, 100 * (1 - frac))
    ma, mb = a > ta, b > tb
    return float(2 * (ma & mb).sum() / (ma.sum() + mb.sum() + 1e-9))


def corr(a, b):
    a = a.ravel().astype(float) - a.mean()
    b = b.ravel().astype(float) - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def mip(vol, axis):
    return vol.max(axis=axis)


def main(ts):
    vol = np.fromfile(f'{DATA_OUT}/volume_{ts}.bin', dtype='<f4').reshape(VOL_N, VOL_N, VOL_N)
    gt = np.fromfile(f'{DATA_OUT}/gt_{ts}.bin', dtype='<f4').reshape(VOL_N, VOL_N, VOL_N)
    k = VOL_N // 2

    d, r, e = metrics(vol, gt)
    c = corr(vol, gt)
    dc = dice(vol, gt)
    print(f'[eval] 相关系数={c:.4f}  骨骼Dice={dc:.4f}  d={d:.4f}  r={r:.4f}  e={e:.4f}')

    # 三正交切片 (轴向 z, 冠状 y, 矢状 x)
    sl = [
        ('轴向 z', vol[k], gt[k]),
        ('冠状 y', vol[:, k, :], gt[:, k, :]),
        ('矢状 x', vol[:, :, k], gt[:, :, k]),
    ]
    vmax = np.percentile(vol, 99.5)
    gmax = np.percentile(gt, 99.5)

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    for row, (name, rv, gv) in enumerate(sl):
        axes[row, 0].imshow(rv, cmap='gray', vmax=vmax); axes[row, 0].set_title(f'重建 {name}')
        axes[row, 1].imshow(gv, cmap='gray', vmax=gmax); axes[row, 1].set_title(f'GT {name}')
        axes[row, 2].imshow(gv - rv * (gmax / vmax), cmap='RdBu_r', vmin=-gmax, vmax=gmax)
        axes[row, 2].set_title('差值')
    # MIP 行
    for j, (name, axis) in enumerate([('轴向', 0), ('冠状', 1), ('矢状', 2)]):
        axes[3, j].imshow(mip(vol, axis), cmap='gray', vmax=vmax)
        axes[3, j].set_title(f'MIP {name}')
    for ax in axes.ravel():
        ax.axis('off')
    plt.suptitle(f'CBCT 重建 vs GT  (相关系数 {c:.3f})', fontsize=13)
    plt.tight_layout()
    out = f'{DATA_OUT}/eval_{ts}.png'
    plt.savefig(out, dpi=100)
    plt.close()
    print(f'[write] {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '1094408')
