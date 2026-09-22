"""同口径比较不同 ``alpha_H`` 在同一个移位窗口里的特征值与模态落点。

每个算例都只用一个移位（σ = 0.30i）与 k=20，模态一律用 ``u`` 分量。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TEMP = Path(os.environ['TEMP'])
REFERENCE = Path('runs/naca0012-ma076-re1e7-c5-a32-nsc160-a02')
GROUPS = [
  ('0.0', Path('runs/naca0012-ma076-re1e7-c5-a32-nsc160'), 'k20'),
  ('0.2', REFERENCE, 'k20'),
  ('0.3', TEMP / 'ah03', 'scan0300'),
  ('0.5', TEMP / 'ah05', 'scan0300'),
  ('0.7', TEMP / 'ah07', 'scan0300'),
  ('1.0', TEMP / 'ah1', 'scan0300'),
]


def main():
  parameters = json.loads((REFERENCE / 'input/parameters.json').read_text(encoding='utf-8'))
  data = np.genfromtxt(REFERENCE / 'ransdata.txt', names=True, encoding='utf-8')
  data = np.sort(data, order=['n', 's'])
  wall = data['x'][data['n'] == 1] / parameters['D_m']
  xc = (data['x'] / parameters['D_m'] - wall.min()) / (wall.max() - wall.min())
  near = xc < 1.4
  print(
    f'{"aH":>4} {"n":>3} {"Re 范围":>22} {"Im 范围":>20} '
    f'{"近场能量占比":>14} {"峰值 x/c":>16} {"模态间重叠中位":>14}'
  )
  for name, root, label in GROUPS:
    stored = np.load(root / f'{label}_modes.npz', allow_pickle=True)
    vectors = []
    rows = []
    for index, value in enumerate(stored['eigenvalues']):
      field = stored['modes'][1::5, index]
      norm = np.linalg.norm(field)
      if norm == 0:
        continue
      vectors.append(field / norm)
      amplitude = np.abs(field)
      rows.append(
        (
          value.real,
          value.imag,
          float((amplitude[near] ** 2).sum() / (amplitude**2).sum()),
          xc[int(np.argmax(amplitude))],
        )
      )
    rows = np.asarray(rows)
    gram = np.abs(np.asarray(vectors).conj() @ np.asarray(vectors).T)
    overlap = gram[~np.eye(len(gram), dtype=bool)]
    peaks = np.unique(np.round(rows[:, 3], 2))
    print(
      f'{name:>4} {len(rows):>3} {rows[:, 0].min():+.4f}..{rows[:, 0].max():+.4f}   '
      f'{rows[:, 1].min():+.4f}..{rows[:, 1].max():+.4f} '
      f'{rows[:, 2].min():>7.2e}..{rows[:, 2].max():<7.2e} '
      f'{peaks.min():>6.2f}..{peaks.max():<6.2f} ({len(peaks)} 处) '
      f'{np.median(overlap):>14.3f}'
    )


if __name__ == '__main__':
  main()
