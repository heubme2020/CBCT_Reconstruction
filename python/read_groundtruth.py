# -*- coding: utf-8 -*-
"""读取现成体数据卷 _3D.raw → gt.bin (float32) + 统计。

这是设备方已重建好的 ground truth, 用于几何反推与定量验证。

用法:
    python read_groundtruth.py [时间戳]     # 默认 1094408

输出:
    ../data/gt_<ts>.bin   体数据 [400, 400, 400] float32 (小端)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 兼容 python -I 隔离模式
import numpy as np

from config import DATA_ROOT, DATA_OUT, PREFIX, VOL_N


def main(ts):
    raw_path = f'{DATA_ROOT}/CTData/{PREFIX}.{ts}_3D.raw'
    v = np.fromfile(raw_path, dtype='<u2').reshape(VOL_N, VOL_N, VOL_N)
    print(f'[read] {raw_path}')
    print(f'[stat] 体数据 {v.shape} min={v.min()} max={v.max()} mean={v.mean():.1f} 中位={np.median(v):.0f}')

    out = f'{DATA_OUT}/gt_{ts}.bin'
    v.astype('<f4').tofile(out)
    print(f'[write] {out}  ({v.nbytes/1e6:.0f} MB float32)')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '1094408')
