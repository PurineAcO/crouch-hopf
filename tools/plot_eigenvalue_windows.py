"""把一个或多个移位窗口的特征值画成复平面分布图，便于对照不同装配版本。

每个算例用一个 ``标签=目录:窗口`` 指定；窗口对应 ``<窗口>_eigenvalues.csv`` 与
``<窗口>_solve.json``，后者用来标出移位 σ 并记录装配指纹。

用法::

  uv run python tools/plot_eigenvalue_windows.py "centre=runs/<case>:k20" \\
      "face-algebraic=%TEMP%/face00:face0300" --out compare
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
plt.rcParams.update(
  {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': False,
  }
)

STYLES = [
  {'marker': 'o', 'color': '#1f4e79'},
  {'marker': '^', 'color': '#c00000'},
  {'marker': 's', 'color': '#2e7d32'},
  {'marker': 'D', 'color': '#7b1fa2'},
  {'marker': 'v', 'color': '#ef6c00'},
]


def parse_spec(spec):
  label, separator, rest = spec.partition('=')
  case, colon, window = rest.rpartition(':')
  if not separator or not colon:
    raise SystemExit(f'需要 "标签=目录:窗口" 形式：{spec}')
  return label, Path(case).expanduser(), window


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('runs', nargs='+', metavar='LABEL=CASE:WINDOW')
  parser.add_argument('--out', type=Path, required=True, help='输出图文件名（不含后缀）')
  parser.add_argument('--strouhal', action='store_true', help='横轴改用 St=ω_r/2π')
  parser.add_argument('--pdf', action='store_true', help='同时输出矢量 PDF')
  args = parser.parse_args()

  loaded = []
  for spec in args.runs:
    label, case, window = parse_spec(spec)
    table = np.genfromtxt(
      case / f'{window}_eigenvalues.csv', delimiter=',', names=True, encoding='utf-8'
    )
    info = json.loads((case / f'{window}_solve.json').read_text(encoding='utf-8'))
    loaded.append((label, table, complex(*info['sigma']), info.get('assembly', {})))
  horizontal = [table['St'] if args.strouhal else table['growth_D_U'] for _, table, _, _ in loaded]
  vertical = [
    table['growth_D_U'] if args.strouhal else table['omega_D_U'] for _, table, _, _ in loaded
  ]
  x_values, y_values = np.concatenate(horizontal), np.concatenate(vertical)

  figure, axis = plt.subplots(figsize=(7.4, 5.8), constrained_layout=True)
  axis.set_xlim(
    x_values.min() - max(0.08 * (x_values.max() - x_values.min()), 0.01),
    x_values.max() + max(0.08 * (x_values.max() - x_values.min()), 0.01),
  )
  axis.set_ylim(
    y_values.min() - max(0.08 * (y_values.max() - y_values.min()), 0.01),
    y_values.max() + max(0.08 * (y_values.max() - y_values.min()), 0.01),
  )
  if args.strouhal:
    axis.axhline(0.0, color='0.35', linewidth=1.1)
    axis.set_xlabel(r'$St = \omega_r/2\pi$', fontsize=12, fontweight='bold')
    axis.set_ylabel(r'$\mathrm{Re}(\lambda D/U)$  (growth)', fontsize=12, fontweight='bold')
  else:
    axis.axvline(0.0, color='0.35', linewidth=1.1)
    axis.axvspan(0.0, axis.get_xlim()[1], color='#c00000', alpha=0.05, zorder=0)
    axis.set_xlabel(r'$\mathrm{Re}(\lambda D/U)$  (growth)', fontsize=12, fontweight='bold')
    axis.set_ylabel(r'$\mathrm{Im}(\lambda D/U)$  (frequency)', fontsize=12, fontweight='bold')

  for position, (label, table, sigma, assembly) in enumerate(loaded):
    growth, omega = table['growth_D_U'], table['omega_D_U']
    x = table['St'] if args.strouhal else growth
    y = growth if args.strouhal else omega
    style = STYLES[position % len(STYLES)]
    axis.scatter(
      x,
      y,
      s=54,
      linewidths=0.7,
      edgecolors='white',
      zorder=3,
      **style,
      label=(
        f'{label}: $\\alpha_H$={assembly.get("alpha_H")}, {assembly.get("boundary_derivative")}'
      ),
    )
    best = int(np.argmax(growth))
    axis.annotate(
      f'{growth[best]:+.4f} @ St={table["St"][best]:.4f}',
      (x[best], y[best]),
      textcoords='offset points',
      xytext=(-9, -4) if x[best] > 0.5 * np.mean(x_values) else (9, -4),
      ha='right' if x[best] > 0.5 * np.mean(x_values) else 'left',
      fontsize=9,
      color=style['color'],
    )
    if not args.strouhal:
      axis.scatter([0.0], [sigma.imag], marker='x', s=70, linewidths=1.6, color=style['color'])
      axis.axhline(sigma.imag, color=style['color'], linestyle=':', linewidth=0.9, alpha=0.5)
  if not args.strouhal:
    axis.annotate(
      'shift $\\sigma=0.30i$',
      (0.0, 0.30),
      textcoords='offset points',
      xytext=(6, 4),
      fontsize=9,
      color='0.30',
    )
  axis.set_title('NACA0012, Re=1e7, M=0.76: $\\alpha_H=0$ window at $\\sigma=0.30i$', fontsize=12)
  axis.legend(fontsize=9, loc='lower right')
  axis.grid(alpha=0.18, linewidth=0.6)
  figure.savefig(args.out.with_suffix('.png'), dpi=170)
  if args.pdf:
    figure.savefig(args.out.with_suffix('.pdf'))
  print(f'写出 {args.out.with_suffix(".png")}')


if __name__ == '__main__':
  main()
