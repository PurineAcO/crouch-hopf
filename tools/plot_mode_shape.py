"""模态形状图：指定模态某个扰动分量的实部与虚部（全翼型 + 激波附近放大）。"""

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
    'axes.linewidth': 1.1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.size': 10.5,
  }
)

parser = argparse.ArgumentParser()
parser.add_argument('case', type=pathlib.Path)
parser.add_argument('--label', default='scan0300')
parser.add_argument('--mode', type=int, required=True)
parser.add_argument('--variable', default='u', choices=('rho', 'u', 'v', 'T', 'nu~'))
parser.add_argument('--out', type=pathlib.Path, required=True)
parser.add_argument('--title', default='')
parser.add_argument(
  '--zoom', type=float, nargs=4, metavar=('X0', 'X1', 'Y0', 'Y1'), default=(0.28, 0.72, -0.06, 0.26)
)
args = parser.parse_args()

parameters = json.loads((args.case / 'input/parameters.json').read_text(encoding='utf-8'))
ns = parameters['nt']
data = np.genfromtxt(args.case / 'ransdata.txt', names=True)
data = np.sort(data, order=['n', 's'])
stored = np.load(args.case / f'{args.label}_modes.npz', allow_pickle=True)
scales = stored['scales']
value = stored['eigenvalues'][args.mode]
index = ('rho', 'u', 'v', 'T', 'nu~').index(args.variable)
# 物理扰动 = 存下来的模态 × 该变量参考量
field = stored['modes'][index::5, args.mode] * scales[index]

wall_x = data['x'][data['n'] == 1] / parameters['D_m']
x0, chord = wall_x.min(), wall_x.max() - wall_x.min()
x = (data['x'] / parameters['D_m'] - x0) / chord
y = data['y'] / parameters['D_m']
triangulation = mtri.Triangulation(x, y)
wall = np.column_stack([x[data['n'] == 1], y[data['n'] == 1]])
centers = np.column_stack([triangulation.x, triangulation.y])[triangulation.triangles].mean(axis=1)
triangulation.set_mask(PlotPath(wall).contains_points(centers))

# 用该分量幅值最大处对齐相位，使实部集中
phase = np.angle(field[np.argmax(np.abs(field))])
field = field * np.exp(-1j * phase)
peak = int(np.argmax(np.abs(field)))
scale = np.abs(field).max()
levels = np.linspace(-1, 1, 41)
labels = {'rho': "\\rho'", 'u': "u'", 'v': "v'", 'T': "T'", 'nu~': "\\tilde{\\nu}'"}

figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
views = (
  ('(a) 全翼型：实部', (-0.15, 1.15), (-0.35, 0.35), field.real),
  ('(b) 全翼型：虚部', (-0.15, 1.15), (-0.35, 0.35), field.imag),
  ('(c) 激波附近：实部', args.zoom[:2], args.zoom[2:], field.real),
  ('(d) 激波附近：虚部', args.zoom[:2], args.zoom[2:], field.imag),
)
for axis, (name, xlim, ylim, values) in zip(axes.ravel(), views):
  filled = axis.tricontourf(
    triangulation, values / scale, levels=levels, cmap='RdBu_r', extend='both'
  )
  axis.fill(wall[:, 0], wall[:, 1], color='0.85', edgecolor='0.3', lw=0.6, zorder=3)
  axis.set(xlim=xlim, ylim=ylim, xlabel='$x/c$', ylabel='$y/c$', title=name)
  figure.colorbar(filled, ax=axis, fraction=0.05, pad=0.02).set_label(
    f'$\\mathrm{{{labels[args.variable]}}}$ (归一化)'
  )

print(
  f'峰值单元 x/c={x[peak]:.4f} y/c={y[peak]:+.4f}，|{args.variable}| 峰 {np.abs(field).max():.4e}'
)
figure.suptitle(
  f'{args.title}  $\\lambda D/U={value.real:+.4f}{value.imag:+.4f}\\,\\mathrm{{i}}$'
  f'  ($\\omega={value.imag:.4f}$, $St={value.imag / (2 * np.pi):.4f}$)'
  f'   峰值 $x/c$={x[peak]:.4f}, $y/c$={y[peak]:+.4f}',
  fontsize=11.5,
)
figure.savefig(args.out, dpi=150)
figure.savefig(pathlib.Path(args.out).with_suffix('.pdf'))
print(f'写出 {args.out}')
print(
  f'实部峰值 {np.abs(field.real).max() / scale:.4f}，虚部峰值 {np.abs(field.imag).max() / scale:.4f}'
)
