"""Plot eigenvalues in the (omega_r, omega_i) plane with the journal style of Crouch (2007).

横轴 ``omega_r = Im(lambda D/U)``、纵轴 ``omega_i = Re(lambda D/U)``，与代码约定
``lambda = sigma + i*omega``（``exp(lambda t)``）一致。每个算例（一个基流）画一组空心符号，
便于像论文那样把不同攻角的结果叠在同一张图上。

用法::

  uv run python tools/plot_paper_spectrum.py "alpha=3.0:runs/case" \
      --labels s015,s025,s035 --xlim 0.1 0.45 --ylim -0.1 0.05 --out fig
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']
COLORS = ['#c00000', '#1f4e79', '#2e7d32', '#7b1fa2', '#ef6c00', '#00838f']


def parse_spec(spec):
  """Parse ``label:case_dir``; the label may itself contain ``=`` (e.g. ``alpha=3.0``)."""
  label, separator, path = spec.partition(':')
  if not separator:
    raise SystemExit(f'case spec must look like "label:path": {spec}')
  return label, Path(path)


def collect(root, labels):
  values = []
  for label in labels:
    path = root / f'{label}_eigenvalues.csv'
    if not path.is_file():
      raise SystemExit(f'missing eigenvalue file: {path}')
    data = np.genfromtxt(path, delimiter=',', names=True)
    for index in range(len(data)):
      values.append((data['omega_D_U'][index], data['growth_D_U'][index]))
  points = np.asarray(values)
  order = np.lexsort((points[:, 1], points[:, 0]))
  points = points[order]
  keep = np.ones(len(points), dtype=bool)
  keep[1:] = (np.abs(np.diff(points[:, 0])) > 1e-6) | (np.abs(np.diff(points[:, 1])) > 1e-6)
  return points[keep]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('cases', nargs='+', help='one or more "label:case_dir" specs')
  parser.add_argument(
    '--labels',
    action='append',
    required=True,
    help='comma separated window prefixes; repeat once per case to give each its own list',
  )
  parser.add_argument('--xlim', type=float, nargs=2, default=None)
  parser.add_argument('--ylim', type=float, nargs=2, default=None)
  parser.add_argument('--annotate-max', action='store_true', help='arrow at the fastest growth')
  parser.add_argument('--out', type=Path, required=True, help='output stem without extension')
  args = parser.parse_args()
  label_sets = [[name.strip() for name in spec.split(',') if name.strip()] for spec in args.labels]
  plt.rcParams.update(
    {
      'font.family': 'serif',
      'font.serif': ['Times New Roman', 'DejaVu Serif'],
      'mathtext.fontset': 'stix',
      'font.size': 12,
      'axes.linewidth': 1.1,
      'xtick.direction': 'in',
      'ytick.direction': 'in',
      'xtick.top': True,
      'ytick.right': True,
    }
  )
  fig, ax = plt.subplots(figsize=(6.4, 5.4), layout='constrained')
  ax.axhline(0.0, color='black', linewidth=1.2)
  ax.axvline(0.0, color='black', linewidth=1.2)
  for position, spec in enumerate(args.cases):
    label, root = parse_spec(spec)
    labels = label_sets[min(position, len(label_sets) - 1)]
    points = collect(root, labels)
    ax.plot(
      points[:, 0],
      points[:, 1],
      linestyle='none',
      marker=MARKERS[position % len(MARKERS)],
      markersize=7.5,
      markerfacecolor='white',
      markeredgecolor=COLORS[position % len(COLORS)],
      markeredgewidth=1.3,
      label=label,
    )
    print(
      f'{label}: {len(points)} unique eigenvalues, '
      f'omega_i in [{points[:, 1].min():+.4f}, {points[:, 1].max():+.4f}]'
    )
    if args.annotate_max:
      best = int(np.argmax(points[:, 1]))
      x, y = points[best]
      ax.annotate(
        '',
        xy=(x, y),
        xytext=(x, y - 0.012),
        arrowprops={'arrowstyle': '-|>', 'color': 'black', 'linewidth': 1.0},
      )
      ax.annotate(
        f'{y:+.4f}',
        (x, y),
        textcoords='offset points',
        xytext=(6, 3),
        fontsize=10,
        color=COLORS[position % len(COLORS)],
      )
  if args.xlim:
    ax.set_xlim(*args.xlim)
  if args.ylim:
    ax.set_ylim(*args.ylim)
  ax.set_xlabel(r'$\omega_r$', fontsize=14)
  ax.set_ylabel(r'$\omega_i$', fontsize=14)
  ax.tick_params(labelsize=12, width=1.1, length=4.5)
  ax.legend(loc='upper left', frameon=False, fontsize=12, handletextpad=0.6)
  stem = args.out.with_suffix('')
  for extension in ('png', 'pdf'):
    fig.savefig(stem.with_suffix(f'.{extension}'), dpi=200)
    print(f'wrote {stem.with_suffix(f".{extension}")}')
  plt.close(fig)


if __name__ == '__main__':
  main()
