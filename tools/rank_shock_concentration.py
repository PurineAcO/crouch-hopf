"""按"激波带集中度"给一次装配的所有模态排序。

激波带取 ``x/c in [0.35, 0.62]``、``y/c in [-0.05, 0.15]``（贴体薄带），
其余区域为带外；比值为带内 ``max |u'|`` 除以带外 ``max |u'|``。
论文 Fig. 8 的 buffet 模态这一比值约为 100。

用法::

  uv run python tools/rank_shock_concentration.py <case>:<label> [<case>:<label> ...]
  uv run python tools/rank_shock_concentration.py runs/a:k20 %TEMP%/face02:face0300 --top 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REFERENCE = Path('runs/naca0012-ma076-re1e7-c5-a32-nsc160-a02')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('runs', nargs='+', metavar='CASE:LABEL', help='算例目录:模态标签')
  parser.add_argument('--top', type=int, default=5)
  args = parser.parse_args()
  groups = []
  for item in args.runs:
    case, _, label = item.rpartition(':')
    if not case or not label:
      raise SystemExit(f'需要 CASE:LABEL 形式：{item}')
    groups.append((label, Path(case).expanduser(), label))
  parameters = json.loads((REFERENCE / 'input/parameters.json').read_text(encoding='utf-8'))
  data = np.genfromtxt(REFERENCE / 'ransdata.txt', names=True, encoding='utf-8')
  data = np.sort(data, order=['n', 's'])
  wall = data['x'][data['n'] == 1] / parameters['D_m']
  xc = (data['x'] / parameters['D_m'] - wall.min()) / (wall.max() - wall.min())
  yc = data['y'] / parameters['D_m']
  band = (xc > 0.35) & (xc < 0.62) & (yc > -0.05) & (yc < 0.15)
  outside = (xc < 1.4) & ~band
  for name, root, label in groups:
    stored = np.load(root / f'{label}_modes.npz', allow_pickle=True)
    rows = []
    for index, value in enumerate(stored['eigenvalues']):
      amplitude = np.abs(stored['modes'][1::5, index])
      if amplitude.max() == 0:
        continue
      peak = int(np.argmax(amplitude))
      rows.append(
        (
          amplitude[band].max() / amplitude[outside].max(),
          value.real,
          value.imag,
          xc[peak],
          yc[peak],
        )
      )
    rows.sort(key=lambda row: -row[0])
    print(f'--- {name}（{len(rows)} 个模态，按激波带集中度排序）---')
    for ratio, re, im, px, py in rows[: args.top]:
      print(
        f'   带内/带外 = {ratio:>8.2f}   λD/U = {re:+.4f}{im:+.4f}i  St={im / (2 * np.pi):.4f}  '
        f'峰值 (x/c={px:.3f}, y/c={py:+.3f})'
      )
    print(f'   带内/带外 中位数 {np.median([row[0] for row in rows]):.3f}，最大值 {rows[0][0]:.2f}')


if __name__ == '__main__':
  main()
