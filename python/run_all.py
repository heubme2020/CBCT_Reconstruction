# -*- coding: utf-8 -*-
"""一键编排: 读投影 -> 负对数 -> CUDA FDK 重建 -> 评估/可视化。

用法:
    python run_all.py [时间戳]     # 默认 1094408
流程:
    1. 读多帧投影 .dic → data/projections_<ts>.bin (若不存在)
    2. 强度 → 线积分(负对数)
    3. numba.cuda FDK 反投影 → data/volume_<ts>.bin
    4. 评估 + 三正交切片/MIP → data/eval_<ts>.png
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np

from config import DATA_OUT, PROJ_N, VOL_N, SERIES
from fdk_ref import to_line_integral
from fdk_cuda import fdk_cuda


def main(ts):
    # 1. 投影 (若未转 bin 则先读)
    proj_bin = f'{DATA_OUT}/projections_{ts}.bin'
    if not os.path.exists(proj_bin):
        import read_projections
        read_projections.main(ts)

    proj = np.fromfile(proj_bin, dtype='<f4').reshape(-1, PROJ_N, PROJ_N)
    print(f'[run] 投影 {proj.shape}')

    # 2. 几何
    geom_path = f'{DATA_OUT}/geom_{ts}.json'
    if os.path.exists(geom_path):
        g = json.load(open(geom_path, encoding='utf-8'))
        R, D, u0, v0 = g['R'], g['D'], g['u0'], g['v0']
        span = g['angles_deg'][-1] - g['angles_deg'][0]
        filt = g.get('filter', 0)
    else:
        R, D, u0, v0, span, filt = 800.0, 1200.0, (PROJ_N - 1) / 2 - 34, (PROJ_N - 1) / 2, 200.0, 0
    F = proj.shape[0]
    angles = np.deg2rad(np.linspace(0, span, F, endpoint=False))

    # 3. 负对数 → 线积分
    pli = to_line_integral(proj)

    # 4. CUDA FDK 重建
    print(f'[run] CUDA 重建 {VOL_N}^3  (R={R:.0f} D={D:.0f} span={span:.0f} filter={filt})')
    t0 = time.time()
    vol = fdk_cuda(pli, angles, R, D, 0.304, u0, v0, VOL_N, 0.5, filter=filt)
    print(f'[run] 重建完成 {time.time() - t0:.1f}s  范围[{vol.min():.1f},{vol.max():.1f}]')
    vol.astype('<f4').tofile(f'{DATA_OUT}/volume_{ts}.bin')

    # 5. 评估 + NIfTI 导出 + 交互式前端查看器
    import evaluate
    evaluate.main(ts)
    import export_nifti
    export_nifti.main(ts)
    import viewer
    viewer.main(ts)
    # 可选三维体渲染: import visualize; visualize.main(ts)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '1094408')
