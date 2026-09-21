"""模态的 u-速度"幅值 + 相位"图（论文 Fig. 8 的形式）。

左列 |u′|，右列固定幅值以上的相位；上排全翼型 + 近尾迹，下排激波附近放大。
相位是逐点定义的，幅值低于阈值处不画（否则全是噪声）。
"""

import argparse
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
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.size': 9,
  }
)

VARIABLES = ('rho', 'u', 'v', 'T', 'nu~')

parser = argparse.ArgumentParser()
parser.add_argument('case', type=pathlib.Path)
parser.add_argument('--label', required=True)
parser.add_argument('--mode', type=int, required=True)
parser.add_argument('--variable', default='u', choices=VARIABLES)
parser.add_argument('--out', type=pathlib.Path, required=True)
parser.add_argument('--title', default='')
parser.add_argument('--floor', type=float, default=1e-3, help='相位阈值（相对峰值）')
parser.add_argument(
  '--decades', type=float, default=4.0, help='幅值面板的对数数量级范围（0 表示线性）'
)
parser.add_argument('--zoom', type=float, nargs=4, default=(0.25, 0.75, -0.06, 0.30))
parser.add_argument('--full', type=float, nargs=4, default=(-0.15, 1.60, -0.45, 0.45))
args = parser.parse_args()

parameters = json.loads((args.case / 'input/parameters.json').read_text(encoding='utf-8'))
ns = parameters['nt']
data = np.genfromtxt(args.case / 'ransdata.txt', names=True)
data = np.sort(data, order=['n', 's'])
wall_x = data['x'][data['n'] == 1] / parameters['D_m']
x0, chord = wall_x.min(), wall_x.max() - wall_x.min()
xc = (data['x'] / parameters['D_m'] - x0) / chord
yc = data['y'] / parameters['D_m']
stored = np.load(args.case / f'{args.label}_modes.npz', allow_pickle=True)
value = stored['eigenvalues'][args.mode]
field = stored['modes'][VARIABLES.index(args.variable) :: 5, args.mode]
peak = int(np.argmax(np.abs(field)))
field = field * np.exp(-1j * np.angle(field[peak]))
magnitude = np.abs(field) / np.abs(field).max()
phase = np.degrees(np.angle(field))

triangulation = mtri.Triangulation(xc, yc)
wall = np.column_stack([xc[data['n'] == 1], yc[data['n'] == 1]])
centers = np.column_stack([triangulation.x, triangulation.y])[triangulation.triangles].mean(axis=1)
inside = PlotPath(wall).contains_points(centers)
value_per_triangle = magnitude[triangulation.triangles].mean(axis=1)

figure, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), constrained_layout=True)
panels = (
  (
    "(a) 幅值 $\\log_{{10}}(|u'|/\\max)$",
    args.full,
    np.log10(np.maximum(magnitude, 1e-4)),
    'inferno',
    (-args.decades, 0.0),
    None,
  ),
  ("(b) 相位 $\\arg u'$（度）", args.full, phase, 'twilight_shifted', None, (-180, 180)),
  (
    '(c) 激波附近幅值',
    args.zoom,
    np.log10(np.maximum(magnitude, 1e-4)),
    'inferno',
    (-args.decades, 0.0),
    None,
  ),
  ('(d) 激波附近相位', args.zoom, phase, 'twilight_shifted', None, (-180, 180)),
)
filtered = triangulation
for axis, (name, limits, values, cmap, magnitude_range, phase_range) in zip(axes.ravel(), panels):
  if phase_range is not None:
    # 幅值过低的单元不画相位：三角形级掩膜，与翼型内部掩膜合并
    triangulation.set_mask(inside | (value_per_triangle < args.floor))
  else:
    triangulation.set_mask(inside)
  filled = axis.tricontourf(
    triangulation,
    values,
    levels=41 if phase_range else 21,
    cmap=cmap,
    vmin=None if phase_range else magnitude_range[0],
    vmax=None if phase_range else magnitude_range[1],
  )
  axis.fill(wall[:, 0], wall[:, 1], color='0.85', edgecolor='0.3', lw=0.5, zorder=3)
  axis.set(
    xlim=limits[:2],
    ylim=limits[2:],
    aspect='equal',
    xlabel='$x/c$',
    ylabel='$y/c$',
    title=name,
  )
  figure.colorbar(filled, ax=axis, fraction=0.046, pad=0.02)

print(
  f'{args.case.name}/{args.label} mode {args.mode}: '
  f'$\\lambda D/U$={value.real:+.4f}{value.imag:+.4f}i, '
  f'ω={value.imag:.4f}, St={value.imag / (2 * np.pi):.4f}'
)
print(f'  峰值单元 x/c={xc[peak]:.4f} y/c={yc[peak]:+.4f}')
shock_band = (xc > 0.36) & (xc < 0.60) & (yc > 0.0) & (yc < 0.35)
interior = np.ones(len(xc), dtype=bool)
interior[:ns] = False
interior[(parameters['nr'] - 1) * ns :] = False
ratio = magnitude[shock_band].max() / magnitude[interior & ~shock_band].max()
print(f'  激波带/带外幅值比 {ratio:.2f}（论文约 100）')
print(f'  幅值 >{args.floor:g} 峰值的单元占比 {float((magnitude > args.floor).mean()):.3%}')
figure.suptitle(
  f'{args.title}  $\\lambda D/U={value.real:+.4f}{value.imag:+.4f}\\,\\mathrm{{i}}$'
  f"  (St={value.imag / (2 * np.pi):.4f})  相位阈值 $|u'|>{args.floor:g}\\,\\max$",
  fontsize=11,
)
figure.savefig(args.out, dpi=150)
print(f'写出 {args.out}')
