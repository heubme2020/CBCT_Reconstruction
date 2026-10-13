# -*- coding: utf-8 -*-
"""读取多帧投影 .dic → projections.bin (float32) + 灰度统计。

用法:
    python read_projections.py [时间戳]     # 默认 1094408

输出:
    ../data/projections_<ts>.bin   投影栈 [F, 978, 978] float32 (小端)
    控制台打印: 尺寸 / 灰度统计 / 空气区 vs 物体区(辅助判断是否线积分域)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 兼容 python -I 隔离模式
import numpy as np
from pydicom import dcmread

from config import DATA_ROOT, DATA_OUT, PREFIX, PROJ_N

import warnings
warnings.filterwarnings('ignore')  # 该数据 XRayTubeCurrent VR 不合规, 屏蔽无关告警


def main(ts):
    dic_path = f'{DATA_ROOT}/{PREFIX}.{ts}.dic'
    ds = dcmread(dic_path)                      # 读全部多帧像素
    pa = ds.pixel_array                          # (F, 978, 978) uint16
    F = pa.shape[0]
    print(f'[read] {dic_path}')
    print(f'[read] 帧数={F} 尺寸={pa.shape[1]}x{pa.shape[2]} dtype={pa.dtype}')

    # 灰度统计
    p = pa.astype(np.float64)
    print(f'[stat] 全局 min={p.min():.1f} max={p.max():.1f} mean={p.mean():.1f}')

    # 空气区(探测器四周 20px 边缘, 射线直射) vs 中心物体区
    border = np.concatenate([
        p[:, :20, :].ravel(), p[:, -20:, :].ravel(),
        p[:, :, :20].ravel(), p[:, :, -20:].ravel(),
    ])
    center = p[:, 300:700, 300:700].ravel()
    print(f'[stat] 空气区(边缘) mean={border.mean():.1f} 中位={np.median(border):.1f}')
    print(f'[stat] 物体区(中心) mean={center.mean():.1f} 中位={np.median(center):.1f}')
    # 若空气区<<物体区 => 线积分域(空气≈0); 若空气区>>物体区 => 强度域(需负对数)
    print(f'[judge] {"线积分域(可直接FDK)" if border.mean() < center.mean() else "强度域(需负对数)"}')

    out = f'{DATA_OUT}/projections_{ts}.bin'
    pa.astype('<f4').tofile(out)
    print(f'[write] {out}  ({pa.nbytes/1e6:.1f} MB float32)')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '1094408')
