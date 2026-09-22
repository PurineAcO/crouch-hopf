"""给一个算例里所有已求出的模态按"空间局部化"排序。

判据（全部在 x/c-y/c 归一化坐标里）：

* ``band``：激波根部带 ``x/c in [0.35, 0.62]``、``y/c in [-0.05, 0.15]``；
* ``up``：上游区 ``x/c < 0.30``；``wake``：尾迹 ``x/c > 0.75``；
* 每个区取 ``max |u'|``，再除以该模态全局 ``max |u'|``（模态已缩放，但只做比值，
  所以整体比例不影响结论）。

论文 Fig. 8 的 buffet 模态应当 ``band`` 接近 1 而 ``up``/``wake`` 远小于 1；
若 ``wake`` 接近 1 则是尾迹模态，若 ``band`` 很小则是纯远场/尾迹模态。

用法::

  uv run python tools/rank_mode_localization.py <case> --labels scan0300 [--variable u]
"""

import argparse
import pathlib
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
VARIABLES = ('rho', 'u', 'v', 'T', 'nu~')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('case', type=pathlib.Path)
  parser.add_argument('--labels', nargs='+', required=True)
  parser.add_argument('--variable', default='u', choices=VARIABLES)
  parser.add_argument('--top', type=int, default=10)
  args = parser.parse_args()

  import json

  parameters = json.loads((args.case / 'input/parameters.json').read_text(encoding='utf-8'))
  data = np.genfromtxt(args.case / 'ransdata.txt', names=True, encoding='utf-8')
  data = np.sort(data, order=['n', 's'])
  wall_x = data['x'][data['n'] == 1] / parameters['D_m']
  xc = (data['x'] / parameters['D_m'] - wall_x.min()) / (wall_x.max() - wall_x.min())
  yc = data['y'] / parameters['D_m']
  band = (xc > 0.35) & (xc < 0.62) & (yc > -0.05) & (yc < 0.15)
  upstream = xc < 0.30
  wake = xc > 0.75
  rows = []
  for label in args.labels:
    stored = np.load(args.case / f'{label}_modes.npz', allow_pickle=True)
    for index, value in enumerate(stored['eigenvalues']):
      field = np.abs(stored['modes'][VARIABLES.index(args.variable) :: 5, index])
      scale = field.max()
      if scale <= 0:
        continue
      rows.append(
        (
          label,
          index,
          value.real,
          value.imag,
          field[band].max() / scale,
          field[upstream].max() / scale,
          field[wake].max() / scale,
          int(np.argmax(field)),
        )
      )
  rows.sort(key=lambda row: -row[4])
  print(
    f'{"label":>9} {"mode":>4} {"Re":>9} {"Im":>8} {"St":>7} '
    f'{"band":>7} {"up":>7} {"wake":>7}  peak node (x/c, y/c)'
  )
  for label, index, re, im, b, u, w, node in rows[: args.top]:
    peak = f'({xc[node]:.3f}, {yc[node]:+.3f})'
    print(
      f'{label:>9} {index:>4} {re:>+9.5f} {im:>8.4f} {im / (2 * np.pi):>7.4f} '
      f'{b:>7.4f} {u:>7.4f} {w:>7.4f}  {peak}'
    )
  print(f'\n共 {len(rows)} 个模态；band = 激波根部带幅值 / 全局峰值')


if __name__ == '__main__':
  main()
