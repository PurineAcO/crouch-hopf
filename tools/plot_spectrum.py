"""Scatter the eigenvalues of one or more cases in a complex plane.

``--plane lambda``（默认）画 **横轴实部 Re(λD/U) – 纵轴虚部 Im(λD/U)**，即特征值本身所在
的复平面，并用 × 标出各移位 σ；``--plane strouhal`` 画横轴 St、纵轴增长率。

用法::

  uv run python tools/plot_spectrum.py "label=path" [--plane lambda] [--out figure.png]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WINDOWS = {'s05': 0.5, 's10': 1.0, 'wake': 2.0, 's30': 3.0, 's40': 4.0}
STYLES = [
  {'marker': 'o', 'color': '#1f4e79', 'label': 'Ma 0.76, Re 1e7, $\\alpha_H=0$'},
  {'marker': '^', 'color': '#c00000', 'label': 'Ma 0.70, Re 3e6, $\\alpha_H=0.2$'},
  {'marker': 's', 'color': '#2e7d32', 'label': 'Ma 0.70, Re 3e6, $\\alpha_H=0$'},
  {'marker': 'D', 'color': '#7b1fa2', 'label': 'Ma 0.70, Re 3e6, $\\alpha_H=1$'},
]
plt.rcParams.update(
  {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.frameon': False,
  }
)


def load(root, windows):
  """Return (St, growth, sigma) for every window present under ``root``."""
  records = []
  for label, sigma in windows.items():
    path = root / f'{label}_eigenvalues.csv'
    if not path.is_file():
      continue
    data = np.genfromtxt(path, delimiter=',', names=True)
    for index in range(len(data)):
      records.append((data['St'][index], data['growth_D_U'][index], sigma))
  return np.asarray(records)


def parse_spec(spec):
  label, separator, path = spec.partition('=')
  if not separator:
    raise SystemExit(f'case spec must look like "label=path": {spec}')
  return label, Path(path)


def draw_shifts(ax, plane):
  """Mark the five shift points of the plane in use."""
  for sigma in sorted(WINDOWS.values()):
    if plane == 'lambda':
      ax.scatter([0.0], [sigma], marker='x', s=64, linewidths=1.5, color='0.30', zorder=4)
      ax.annotate(
        f'$\\sigma={sigma:g}i$',
        (0.0, sigma),
        textcoords='offset points',
        xytext=(7, -3),
        fontsize=9,
        color='0.30',
      )
    else:
      ax.axvline(sigma / (2 * np.pi), color='0.7', linestyle=':', linewidth=1.0)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('cases', nargs='+', help='one or more "label=path" specs')
  parser.add_argument('--plane', choices=['lambda', 'strouhal'], default='lambda')
  parser.add_argument('--out', type=Path, default=None, help='output figure stem')
  args = parser.parse_args()
  cases = [parse_spec(spec) for spec in args.cases]
  fig, ax = plt.subplots(figsize=(7.6, 5.8), constrained_layout=True)
  if args.plane == 'lambda':
    ax.axvspan(0.0, 1.0, color='#c00000', alpha=0.05)
    ax.axvline(0.0, color='0.35', linewidth=1.1)
    xlabel = '$\\mathrm{Re}(\\lambda D/U)$  (growth)'
    ylabel = '$\\mathrm{Im}(\\lambda D/U)$  (frequency)'
  else:
    ax.axhline(0.0, color='0.35', linewidth=1.1)
    xlabel = '$St = \\omega_r/2\\pi$  (frequency)'
    ylabel = '$\\mathrm{Re}(\\lambda D/U)$  (growth)'
  draw_shifts(ax, args.plane)
  for position, (label, root) in enumerate(cases):
    values = load(root, WINDOWS)
    if values.size == 0:
      raise SystemExit(f'no eigenvalue files found under {root}')
    growth = values[:, 1]
    if args.plane == 'lambda':
      x, y = growth, 2 * np.pi * values[:, 0]
    else:
      x, y = values[:, 0], growth
    style = dict(STYLES[position % len(STYLES)])
    style['label'] = label
    ax.scatter(x, y, s=46, linewidths=0.7, edgecolors='white', zorder=3, **style)
    best = int(np.argmax(growth))
    ax.annotate(
      f'{growth[best]:+.4f}',
      (x[best], y[best]),
      textcoords='offset points',
      xytext=(7, 5),
      ha='left',
      fontsize=9,
      color=style['color'],
    )
  ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
  ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
  ax.tick_params(labelsize=11)
  for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight('bold')
  if args.plane == 'lambda':
    note = 'red band: growing side ($\\mathrm{Re}>0$);  $\\times$ = shift $\\sigma$'
  else:
    note = 'dotted lines: shift $\\sigma$'
  ax.text(
    0.99, 0.02, note, transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color='0.4'
  )
  if len(cases) > 1:
    ax.legend(loc='upper right', fontsize=10)
  ax.grid(alpha=0.15, linewidth=0.8)
  stem = args.out or (cases[0][1] / 'spectrum')
  for extension in ('png', 'pdf'):
    fig.savefig(stem.with_suffix(f'.{extension}'), dpi=180)
    print(f'wrote {stem.with_suffix(f".{extension}")}')
  plt.close(fig)


if __name__ == '__main__':
  main()
