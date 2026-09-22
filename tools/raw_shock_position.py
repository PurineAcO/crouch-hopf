"""用网格几何把交付的原始基流（``baseflow.dat``，9 列）对回 (s, n)，并定位激波。

原始文件的列是 ``x y rho u v T p Ma nu_tilde``，没有 s/n；同一算例的
``ransdata.txt`` 有相同的几何，用 (x, y) 精确匹配即可恢复拓扑。
"""

import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

VARIABLES = ('x', 'y', 'rho', 'u', 'v', 'T', 'p', 'Ma', 'nu_tilde')


def load_raw(path):
  rows = []
  with path.open(encoding='utf-8') as stream:
    for line in stream:
      if line.startswith(('TITLE', 'VARIABLES', '#', 'UTF8')):
        continue
      parts = line.split()
      if len(parts) == 9:
        rows.append([float(value) for value in parts])
  return np.asarray(rows)


def main():
  case = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('runs/naca0012-ma076-re1e7-c5-a32')
  raw = load_raw(case / 'baseflow.dat')
  mesh = np.genfromtxt(
    Path('runs/naca0012-ma076-re1e7-c5-a32-nsc160/ransdata.txt'), names=True, encoding='utf-8'
  )
  mesh = np.sort(mesh, order=['n', 's'])
  key = {(round(x, 6), round(y, 6)): i for i, (x, y) in enumerate(zip(mesh['x'], mesh['y']))}
  index = np.array([key.get((round(x, 6), round(y, 6)), -1) for x, y in zip(raw[:, 0], raw[:, 1])])
  missing = int((index < 0).sum())
  print(f'{case.name}: baseflow.dat {len(raw)} 行，几何失配 {missing} 行')
  good = index >= 0
  normal = np.zeros(len(raw), dtype=int)
  station = np.zeros(len(raw), dtype=int)
  normal[good] = mesh['n'][index[good]].astype(int)
  station[good] = mesh['s'][index[good]].astype(int)
  wall = normal == 1
  x, y, p, ss = raw[wall, 0], raw[wall, 1], raw[wall, 6], station[wall]
  order = np.argsort(ss)
  x, y, p, ss = x[order], y[order], p[order], ss[order]
  arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
  gradient = np.gradient(p, arc)
  chord = x + 0.5
  window = np.flatnonzero((chord > 0.25) & (chord < 0.80) & (ss <= 190))
  top = window[np.argsort(-np.abs(gradient[window]))[:4]]
  listed = '  '.join(f'x/c={chord[i]:.4f}({abs(gradient[i]) / 1e3:.0f} kPa/m)' for i in top)
  print(f'  原始（未平滑）基流上表面最强压力梯度: {listed}')
  print(
    f'  p 范围 {p.min():.0f} .. {p.max():.0f} Pa，nu_tilde 范围 '
    f'{raw[:, 8].min():.3e} .. {raw[:, 8].max():.3e}'
  )


if __name__ == '__main__':
  main()
