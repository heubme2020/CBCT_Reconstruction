# -*- coding: utf-8 -*-
"""三维体渲染 (plotly, 导出交互式 HTML)。

对重建体数据做等值面(骨骼) + 体渲染, 输出可在浏览器旋转/缩放的 HTML。

用法:
    python visualize.py [时间戳]
输出:
    data/render_<ts>.html   交互式三维体渲染
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.ndimage import zoom
import plotly.graph_objects as go

from config import DATA_OUT, VOL_N


def main(ts):
    vol = np.fromfile(f'{DATA_OUT}/volume_{ts}.bin', dtype='<f4').reshape(VOL_N, VOL_N, VOL_N)
    # 降采样到 128³ 供体渲染
    v = zoom(vol.astype(np.float32), 128 / VOL_N, order=1)

    # 骨骼等值面(高值), 软组织体渲染(中值)
    lo = np.percentile(v, 60)
    hi = np.percentile(v, 95)
    fig = go.Figure()
    # 软组织体渲染
    fig.add_trace(go.Volume(
        x=np.linspace(0, 128, 128), y=np.linspace(0, 128, 128), z=np.linspace(0, 128, 128),
        value=v.flatten(),
        isomin=lo, isomax=hi,
        opacity=0.08,
        surface_count=15,
        colorscale='Gray',
        showscale=False,
    ))
    # 骨骼等值面
    fig.add_trace(go.Isosurface(
        x=np.linspace(0, 128, 128), y=np.linspace(0, 128, 128), z=np.linspace(0, 128, 128),
        value=v.flatten(),
        isomin=hi, isomax=v.max(),
        surface_count=1,
        colorscale='YlOrRd',
        opacity=0.6,
        showscale=False,
    ))
    fig.update_layout(
        scene=dict(aspectmode='data', xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        title=f'CBCT 重建三维体渲染 (时间戳 {ts})',
        width=800, height=700,
    )
    out = f'{DATA_OUT}/render_{ts}.html'
    fig.write_html(out)
    print(f'[write] {out}  (浏览器打开, 可旋转/缩放)')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '1094408')
