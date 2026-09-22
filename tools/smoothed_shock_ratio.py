"""区分"激波带优势"来自光滑包络还是格点噪声。

带内/带外比值若直接用逐单元 max，会被 2Δ（棋盘格）分量骗过去。这里把 |u'| 排回
(n, s) 网格后做 s/n 两个方向的 3 点平均（对 2Δ 分量响应为零），再算同样的比值。

用法: uv run python tools/smoothed_shock_ratio.py <case>:<label> ...
"""

import json
import sys
from pathlib import Path

import numpy as np

REFERENCE = Path('runs/naca0012-ma076-re1e7-c5-a32-nsc160-a02')


def load_topology():
  parameters = json.loads((REFERENCE / 'input/parameters.json').read_text(encoding='utf-8'))
  data = np.genfromtxt(REFERENCE / 'ransdata.txt', names=True, encoding='utf-8')
  data = np.sort(data, order=['n', 's'])
  wall = data['x'][data['n'] == 1] / parameters['D_m']
  xc = (data['x'] / parameters['D_m'] - wall.min()) / (wall.max() - wall.min())
  yc = data['y'] / parameters['D_m']
  station, ring = data['s'].astype(int), data['n'].astype(int)
  rows, columns = ring.max(), station.max()
  return xc, yc, station, ring, rows, columns


def smooth(field, rows, columns):
  """在 (n, s) 网格上做两个方向的 3 点平均；s 方向周期回绕，n 方向端点复制。"""
  grid = field.reshape(rows, columns)
  along_s = (grid + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1)) / 3.0
  padded = np.vstack([along_s[:1], along_s, along_s[-1:]])
  along_n = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
  return along_n.ravel()


def main():
  xc, yc = load_topology()[:2]
  rows, columns = load_topology()[-2:]
  band = (xc > 0.35) & (xc < 0.62) & (yc > -0.05) & (yc < 0.15)
  outside = (xc < 1.4) & ~band
  print(f'{"label":>18} {"逐单元":>8} {"3点平均":>8} {"平滑峰值 x/c":>12}')
  for spec in sys.argv[1:]:
    case, _, label = spec.rpartition(':')
    stored = np.load(Path(case).expanduser() / f'{label}_modes.npz', allow_pickle=True)
    best_raw, best_smooth, peaks, indices = [], [], [], []
    for index in range(stored['modes'].shape[1]):
      amplitude = np.abs(stored['modes'][1::5, index])
      if amplitude.max() == 0:
        continue
      smoothed = smooth(amplitude, rows, columns)
      best_raw.append(amplitude[band].max() / amplitude[outside].max())
      best_smooth.append(smoothed[band].max() / smoothed[outside].max())
      peaks.append(xc[int(np.argmax(smoothed))])
      indices.append(index)
    order = np.argsort(-np.array(best_smooth))
    print(
      f'{label:>18} {max(best_raw):>8.2f} {best_smooth[order[0]]:>8.2f} {peaks[order[0]]:>12.3f}'
    )
    print(
      '  平滑后前 5（模态,比值,x/c）: '
      + ' '.join(f'({indices[i]},{best_smooth[i]:.2f},{peaks[i]:.3f})' for i in order[:5])
    )
    print(
      f'{"":>18} 平滑后前 5 名带内/带外: ' + ' '.join(f'{best_smooth[i]:.2f}' for i in order[:5])
    )
    print(f'{"":>18} 平滑后峰值 x/c: ' + ' '.join(f'{peaks[i]:.3f}' for i in order[:5]))


if __name__ == '__main__':
  main()
