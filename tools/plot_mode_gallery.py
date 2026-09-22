"""把一个谱里的所有模态画成网格总览图（每个模态一栏）。

模态清单来自 tools/scan_spectrum.py 产生的 scan_merged.csv（按窗口去重）。
每栏画指定扰动分量的实部/虚部/幅值，并自动缩放到该模态能量集中的区域，
以便看清它是"沿激波延展"还是"贴在壁面/尾迹里"。
"""

import argparse
import csv
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path as PlotPath

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
plt.rcParams.update(
  {
    'font.family': 'serif',
    'font.serif': ['Noto Serif SC', 'Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.9,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.size': 8,
  }
)

VARIABLES = ('rho', 'u', 'v', 'T', 'nu~')
LABELS = {'rho': "\\rho'", 'u': "u'", 'v': "v'", 'T': "T'", 'nu~': "\\tilde{\\nu}'"}

parser = argparse.ArgumentParser()
parser.add_argument('case', type=pathlib.Path)
parser.add_argument('--variable', default='u', choices=VARIABLES)
parser.add_argument('--part', default='real', choices=('real', 'imag', 'abs'))
parser.add_argument(
  '--scale',
  default='saturated',
  choices=('saturated', 'log', 'full'),
  help='色标：saturated=按峰值若干比例饱和（看结构），log=|幅值| 对数（看尾部），full=完整 ±1',
)
parser.add_argument('--saturate', type=float, default=0.05, help='saturated 时相对峰值的上限')
parser.add_argument('--decades', type=float, default=4.0, help='log 时显示的数量级范围')
parser.add_argument('--pdf', action='store_true', help='同时输出光栅化 PDF（矢量 PDF 会极大）')
parser.add_argument('--dpi', type=int, default=150)
parser.add_argument('--columns', type=int, default=8)
parser.add_argument(
  '--window',
  default=None,
  help='不读 scan_merged.csv，直接画该标签（如 stage1_0300）的全部模态',
)
parser.add_argument('--out', type=pathlib.Path, required=True)
parser.add_argument('--title', default='')
parser.add_argument(
  '--view',
  default='auto',
  choices=('auto', 'airfoil', 'wide'),
  help='每栏视野：自动收拢 / 固定全翼型 / 固定翼型加近尾迹',
)
parser.add_argument(
  '--min-span',
  type=float,
  nargs=2,
  default=(0.55, 0.22),
  metavar=('DX', 'DY'),
  help='自动视野的最小弦向/法向跨度',
)
args = parser.parse_args()

parameters = json.loads((args.case / 'input/parameters.json').read_text(encoding='utf-8'))
data = np.genfromtxt(args.case / 'ransdata.txt', names=True)
data = np.sort(data, order=['n', 's'])
wall_x = data['x'][data['n'] == 1] / parameters['D_m']
x0, chord = wall_x.min(), wall_x.max() - wall_x.min()
xc = (data['x'] / parameters['D_m'] - x0) / chord
yc = data['y'] / parameters['D_m']
triangulation = mtri.Triangulation(xc, yc)
wall = np.column_stack([xc[data['n'] == 1], yc[data['n'] == 1]])
centers = np.column_stack([triangulation.x, triangulation.y])[triangulation.triangles].mean(axis=1)
triangulation.set_mask(PlotPath(wall).contains_points(centers))

if args.window:
  values = np.load(args.case / f'{args.window}_modes.npz', allow_pickle=True)['eigenvalues']
  table = np.genfromtxt(
    args.case / f'{args.window}_eigenvalues.csv', delimiter=',', names=True, encoding='utf-8'
  )
  records = [
    {
      'window': args.window,
      'mode': str(index),
      'growth_D_U': repr(float(table['growth_D_U'][index])),
      'omega_D_U': repr(float(table['omega_D_U'][index])),
      'St': repr(float(table['St'][index])),
    }
    for index in range(len(values))
  ]
else:
  records = list(
    csv.DictReader((args.case / 'scan_merged.csv').read_text(encoding='utf-8').splitlines())
  )
entries, seen = [], set()
for item in records:
  key = (item['window'], int(item['mode']))
  if key in seen:
    continue
  seen.add(key)
  entries.append(
    {
      'window': item['window'],
      'mode': int(item['mode']),
      'growth': float(item['growth_D_U']),
      'omega': float(item['omega_D_U']),
      'St': float(item['St']),
    }
  )
entries.sort(key=lambda entry: -entry['growth'])

cache = {}
columns = min(args.columns, len(entries))
rows = -(-len(entries) // columns)
figure, axes = plt.subplots(
  rows, columns, figsize=(2.15 * columns, 2.05 * rows), constrained_layout=True, squeeze=False
)
filled = None

for index, entry in enumerate(entries):
  axis = axes[index // columns][index % columns]
  window = entry['window']
  if window not in cache:
    cache[window] = np.load(args.case / f'{window}_modes.npz', allow_pickle=True)['modes']
  field = cache[window][VARIABLES.index(args.variable) :: 5, entry['mode']]
  weight = np.abs(field) ** 2
  peak = int(np.argmax(weight))
  field = field * np.exp(-1j * np.angle(field[peak]))
  if args.part == 'abs':
    values = np.abs(field) / max(np.abs(field).max(), 1e-300)
  else:
    values = (field.real if args.part == 'real' else field.imag) / max(np.abs(field).max(), 1e-300)
  entry['peak'] = peak

  if args.view == 'airfoil':
    xlim, ylim = (-0.20, 1.20), (-0.45, 0.80)
  elif args.view == 'wide':
    xlim, ylim = (-0.30, 8.50), (-1.60, 1.60)
  else:
    order = np.argsort(-weight)
    cumulative = np.cumsum(weight[order]) / weight.sum()
    keep = order[cumulative <= 0.995]
    if keep.size < 8:
      keep = order[:8]
    bx0, bx1 = xc[keep].min(), xc[keep].max()
    by0, by1 = yc[keep].min(), yc[keep].max()
    if bx1 - bx0 < args.min_span[0]:
      middle = 0.5 * (bx0 + bx1)
      bx0, bx1 = middle - 0.5 * args.min_span[0], middle + 0.5 * args.min_span[0]
    if by1 - by0 < args.min_span[1]:
      middle = 0.5 * (by0 + by1)
      by0, by1 = middle - 0.5 * args.min_span[1], middle + 0.5 * args.min_span[1]
    pad_x, pad_y = 0.08 * (bx1 - bx0), 0.12 * (by1 - by0)
    xlim, ylim = (bx0 - pad_x, bx1 + pad_x), (by0 - pad_y, by1 + pad_y)

  if args.scale == 'log':
    magnitude = np.log10(np.maximum(np.abs(field), 1e-300) / max(np.abs(field).max(), 1e-300))
    filled = axis.tricontourf(
      triangulation,
      magnitude,
      levels=np.linspace(-args.decades, 0.0, 41),
      cmap='inferno',
      extend='min',
    )
  elif args.scale == 'saturated':
    limit = args.saturate
    filled = axis.tricontourf(
      triangulation,
      values,
      levels=np.linspace(-limit, limit, 41),
      cmap='RdBu_r',
      extend='both',
    )
  else:
    filled = axis.tricontourf(
      triangulation, values, levels=np.linspace(-1, 1, 41), cmap='RdBu_r', extend='both'
    )
  axis.fill(wall[:, 0], wall[:, 1], color='0.8', edgecolor='0.35', lw=0.5, zorder=3)
  axis.plot(xc[peak], yc[peak], marker='x', ms=4, mew=1.1, color='k', zorder=4)
  axis.set(xlim=xlim, ylim=ylim, aspect='equal')
  axis.tick_params(labelsize=6.5, length=2, pad=1.5)
  colour = '#c00000' if entry['growth'] > 0 else '#1f4e79'
  axis.set_title(
    f'#{index + 1}  $\\lambda_r$={entry["growth"]:+.4f}\n'
    f'$\\omega$={entry["omega"]:.4f}  St={entry["St"]:.4f}',
    fontsize=6.8,
    color=colour,
    pad=2.5,
  )

for index in range(len(entries), rows * columns):
  axes[index // columns][index % columns].axis('off')

if filled is not None:
  label = f'$\\mathrm{{{LABELS[args.variable]}}}$'
  if args.scale == 'log':
    label += ' 归一化幅值的 $\\log_{10}$'
  else:
    label += {'real': '（实部）', 'imag': '（虚部）', 'abs': '（幅值）'}[args.part]
    if args.scale == 'saturated':
      label += f'（色标 $\\pm${args.saturate:g} 倍峰值，超出部分饱和）'
  figure.colorbar(filled, ax=axes, fraction=0.012, pad=0.01).set_label(label, fontsize=8)

figure.suptitle(
  f'{args.title}  共 {len(entries)} 个唯一模态，按增长率降序；'
  f'红标题为增长（不稳），蓝标题为衰减；黑叉为峰值单元'
  + ('' if args.view != 'auto' else '；每栏视野按该模态 99.5% 能量范围自动收拢'),
  fontsize=10,
)
figure.savefig(args.out, dpi=args.dpi)
if args.pdf:
  figure.savefig(pathlib.Path(args.out).with_suffix('.pdf'), dpi=args.dpi, rasterized=True)
print(f'写出 {args.out}')

print(f'{"#":>3} {"window":>8} {"mode":>4} {"growth":>9} {"omega":>8} {"St":>7}  峰值(x/c,y/c)')
for index, entry in enumerate(entries):
  peak = entry['peak']
  print(
    f'{index + 1:>3} {entry["window"]:>8} {entry["mode"]:>4} {entry["growth"]:>+9.4f} '
    f'{entry["omega"]:>8.4f} {entry["St"]:>7.4f}  ({xc[peak]:.3f},{yc[peak]:+.3f})'
  )
