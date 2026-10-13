# -*- coding: utf-8 -*-
"""交互式前端查看器 (自包含 HTML, 无需联网/服务器, 浏览器双击打开)。

三正交切片(重建 vs GT 并排) + 切片位置滑块联动 + MIP 投影。
数据以 uint8 量化 + base64 内嵌, 单文件几 MB。

用法:
    python viewer.py [时间戳]
输出:
    data/viewer_<ts>.html
"""
import os
import sys
import base64
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.ndimage import zoom

from config import DATA_OUT, VOL_N


def _to_u8(a, N):
    a = zoom(a.astype(np.float32), N / VOL_N, order=1)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)


def main(ts):
    vol = np.fromfile(f'{DATA_OUT}/volume_{ts}.bin', dtype='<f4').reshape(VOL_N, VOL_N, VOL_N)
    gt = np.fromfile(f'{DATA_OUT}/gt_{ts}.bin', dtype='<f4').reshape(VOL_N, VOL_N, VOL_N)
    N = 96
    v = _to_u8(vol, N)
    g = _to_u8(gt, N)

    def b64(arr):
        return base64.b64encode(arr.tobytes()).decode()

    html = _HTML.format(
        ts=ts, N=N, half=N // 2,
        vol_b64=b64(v), gt_b64=b64(g),
    )
    out = f'{DATA_OUT}/viewer_{ts}.html'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'[write] {out}  ({os.path.getsize(out)/1e6:.1f} MB)')
    print('  浏览器双击打开, 拖动滑块浏览三正交切片(上排重建 / 下排 GT)')


_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>CBCT 重建结果查看器 {ts}</title>
<style>
body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; margin: 0; background: #1a1a1f; color: #eee; }}
header {{ padding: 12px 20px; background: #23232a; border-bottom: 1px solid #333; }}
header h1 {{ font-size: 16px; margin: 0; }}
header p {{ font-size: 12px; color: #999; margin: 4px 0 0; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 12px 20px; }}
.cell {{ text-align: center; }}
.cell canvas {{ width: 100%; image-rendering: pixelated; border: 1px solid #333; background: #000; }}
.cell .cap {{ font-size: 12px; color: #aaa; margin-top: 4px; }}
.controls {{ padding: 8px 20px 24px; }}
.controls label {{ font-size: 12px; color: #bbb; }}
input[type=range] {{ width: 100%; margin-top: 6px; }}
</style>
</head>
<body>
<header>
  <h1>CBCT 三维重建结果查看器 — 检查 {ts}</h1>
  <p>上排：重建体数据；下排：厂商 ground truth。拖动滑块浏览各层切片。</p>
</header>
<div class="grid">
  <div class="cell"><canvas id="ax0" width="{N}" height="{N}"></canvas><div class="cap">轴向 重建</div></div>
  <div class="cell"><canvas id="cor0" width="{N}" height="{N}"></canvas><div class="cap">冠状 重建</div></div>
  <div class="cell"><canvas id="sag0" width="{N}" height="{N}"></canvas><div class="cap">矢状 重建</div></div>
  <div class="cell"><canvas id="ax1" width="{N}" height="{N}"></canvas><div class="cap">轴向 GT</div></div>
  <div class="cell"><canvas id="cor1" width="{N}" height="{N}"></canvas><div class="cap">冠状 GT</div></div>
  <div class="cell"><canvas id="sag1" width="{N}" height="{N}"></canvas><div class="cap">矢状 GT</div></div>
</div>
<div class="controls">
  <label>切片位置：<span id="pos"></span></label>
  <input type="range" id="slider" min="0" max="{N}" value="{half}">
</div>
<script>
const N = {N};
const vol = new Uint8Array(atob('{vol_b64}').split('').map(c => c.charCodeAt(0)));
const gt  = new Uint8Array(atob('{gt_b64}').split('').map(c => c.charCodeAt(0)));
const canvases = {{
  ax0: document.getElementById('ax0'), cor0: document.getElementById('cor0'), sag0: document.getElementById('sag0'),
  ax1: document.getElementById('ax1'), cor1: document.getElementById('cor1'), sag1: document.getElementById('sag1'),
}};
function drawSlice(canvas, data) {{
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(N, N);
  for (let k = 0; k < N * N; k++) {{
    img.data[k*4] = img.data[k*4+1] = img.data[k*4+2] = data[k];
    img.data[k*4+3] = 255;
  }}
  ctx.putImageData(img, 0, 0);
}}
function render(i) {{
  let ax = new Uint8Array(N*N), cor = new Uint8Array(N*N), sag = new Uint8Array(N*N);
  // 轴向: 固定 z=i, row=y, col=x
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    ax[y*N + x] = vol[i*N*N + y*N + x];
  // 冠状: 固定 y=i, row=z, col=x
  for (let z = 0; z < N; z++) for (let x = 0; x < N; x++)
    cor[z*N + x] = vol[z*N*N + i*N + x];
  // 矢状: 固定 x=i, row=z, col=y
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++)
    sag[z*N + y] = vol[z*N*N + y*N + i];
  drawSlice(canvases.ax0, ax); drawSlice(canvases.cor0, cor); drawSlice(canvases.sag0, sag);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    ax[y*N + x] = gt[i*N*N + y*N + x];
  for (let z = 0; z < N; z++) for (let x = 0; x < N; x++)
    cor[z*N + x] = gt[z*N*N + i*N + x];
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++)
    sag[z*N + y] = gt[z*N*N + y*N + i];
  drawSlice(canvases.ax1, ax); drawSlice(canvases.cor1, cor); drawSlice(canvases.sag1, sag);
  document.getElementById('pos').textContent = i + ' / ' + (N - 1);
}}
const slider = document.getElementById('slider');
slider.addEventListener('input', () => render(parseInt(slider.value)));
render(parseInt(slider.value));
</script>
</body>
</html>
"""

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '1094408')
