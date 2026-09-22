"""定位各基流上表面的激波位置（最强壁面压力梯度处）。"""

import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CASES = [
  'naca0012-ma076-re1e7-c5-a32-nsc160',
  'naca0012-ma076-re1e7-c5-a32-nsc160-a02',
  'naca0012-ma076-re1e7-ref-nsc160',
]


def shock(case):
  data = np.genfromtxt(case / 'ransdata.txt', names=True, encoding='utf-8')
  data = np.sort(data, order=['n', 's'])
  normal, station = data['n'].astype(int), data['s'].astype(int)
  wall = normal == 1
  x, y, ss = data['x'][wall], data['y'][wall], station[wall]
  p = (data['rho'] * 287.05 * data['T'])[wall]
  order = np.argsort(ss)
  x, y, p, ss = x[order], y[order], p[order], ss[order]
  arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
  gradient = np.gradient(p, arc)
  # 只看上表面激波可能所在的中弦段，避开后缘角点与前缘驻点
  chord = x + 0.5
  window = np.flatnonzero((chord > 0.25) & (chord < 0.80) & (ss <= 190))
  top = window[np.argsort(-np.abs(gradient[window]))[:4]]
  listed = '  '.join(f'x/c={chord[i]:.4f}({abs(gradient[i]) / 1e3:.0f} kPa/m)' for i in top)
  print(f'{case.name}: {listed}')


def main():
  for name in CASES:
    case = Path('runs') / name
    if (case / 'ransdata.txt').exists():
      shock(case)
    else:
      print(f'{name}: 缺少 ransdata.txt')


if __name__ == '__main__':
  main()
