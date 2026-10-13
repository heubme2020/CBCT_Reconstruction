# -*- coding: utf-8 -*-
"""共享路径与数据清单配置。

目录约定:
    CBCT/
    ├── 100000027/           原始检查数据(只读, 不修改)
    └── recon/               本项目
        ├── python/          数据 I/O、几何、评估、可视化
        ├── cuda/            FDK 重建核心(CUDA C++)
        └── data/            中间产物(bin / geom.json)
"""
import os
import sys

# Windows 控制台默认 GBK, 统一 UTF-8 输出避免中文乱码
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

RECON_PY = os.path.dirname(os.path.abspath(__file__))       # CBCT/recon/python
PROJECT_DIR = os.path.dirname(RECON_PY)                     # CBCT/recon
DATA_ROOT = os.path.join(os.path.dirname(PROJECT_DIR), '100000027')  # CBCT/100000027
DATA_OUT = os.path.join(PROJECT_DIR, 'data')                # CBCT/recon/data

# 文件名前缀: <机构UID>.<accession>.<date>.<timestamp>
# 机构 UID 前缀已脱敏, 请按实际数据文件名替换
PREFIX = '<机构UID>.100000027.20231018'

# 4 组原始投影序列(时间戳 -> 帧数)
SERIES = {
    '1094408': 191,
    '1102941': 383,
    '1105919': 168,
    '1113544': 191,
}

# 5 个重建体数据卷(时间戳), 其中 1111709 无对应投影
VOLUMES = ['1094408', '1102941', '1105919', '1111709', '1113544']

# 投影 / 体数据固定尺寸(已探明)
PROJ_N = 978        # 投影像素边长
VOL_N = 400         # 体数据边长
PROJ_PITCH = 0.304  # 探测器像素间距 mm
VOL_PITCH = 0.5058  # 重建等中心像素间距 mm

os.makedirs(DATA_OUT, exist_ok=True)
