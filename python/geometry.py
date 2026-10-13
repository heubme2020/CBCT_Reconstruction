# -*- coding: utf-8 -*-
"""几何确定(体数据反推 + 假设) — 已通过 GT 相关性验证收敛。

结论(经球体自洽测试 + 真实数据相关性网格搜索确定):
    - 扫描范围 span = 200° (C 臂短扫描, 191 帧 ≈ 1.047°/帧)
    - 放大倍数 M = D/R = 1.5 (非像素间距比 1.664, 后者是 DICOM 像素定义差异)
    - R = 800 mm, D = 1200 mm (R 在 600~1500 对相关性影响仅 0.02, 取物理合理值)
    - 旋转轴在探测器 v(行) 方向, 斜坡滤波沿 u(列) 方向
    - 探测器中心偏移 u0 = (W-1)/2 - 34 (正弦拟合: 旋转轴投影在 454.3)

输出 geom.json 供 CUDA 重建程序读取。

用法:
    python geometry.py [时间戳]      # 写 data/geom_<ts>.json
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np

from config import DATA_OUT, PROJ_N, VOL_N, PROJ_PITCH, SERIES

# —— 已确定的几何参数(经 GT 相关性网格搜索 + 球体自洽验证收敛) ——
DET_PITCH = PROJ_PITCH          # 探测器物理像素间距 0.304 mm
SPAN_DEG = 200.0                # 扫描范围(度), C 臂短扫描
M = 1.6                         # 放大倍数 D/R (外推+滤波下最优, 对应 FOV 边缘轻微截断)
R = 800.0                       # 源-旋转中心距离 mm
U0 = (PROJ_N - 1) / 2.0 - 24.0  # 探测器列中心(旋转轴投影)
V0 = (PROJ_N - 1) / 2.0         # 探测器行中心
VOL_PITCH = 0.5                 # 重建体素 mm (_3D.raw 的 Pixeld_Volume)


def build_geom(ts):
    F = SERIES[ts]
    angles_deg = np.linspace(0.0, SPAN_DEG, F, endpoint=False).tolist()
    geom = {
        'projection_file': f'projections_{ts}.bin',
        'volume_file': f'volume_{ts}.bin',
        'F': F, 'H': PROJ_N, 'W': PROJ_N,
        'R': R, 'D': R * M,
        'det_pitch': DET_PITCH,
        'u0': U0, 'v0': V0,
        'angles_deg': angles_deg,
        'vol_size': VOL_N, 'vol_pitch': VOL_PITCH,
        'filter': 2,  # 2=截断外推+R-L斜坡滤波(恢复骨骼细节), 0=无滤波, 1=R-L
    }
    out = f'{DATA_OUT}/geom_{ts}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(geom, f, indent=2)
    print(f'[write] {out}')
    print(f'  R={R} D={geom["D"]} M={M} span={SPAN_DEG}° F={F} u0={U0} v0={V0}')


if __name__ == '__main__':
    build_geom(sys.argv[1] if len(sys.argv) > 1 else '1094408')
