"""Plot a selected velocity eigenmode; mode index must be chosen from the eigenvalue CSV."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('case', type=Path)
  parser.add_argument('--label', default='wake')
  parser.add_argument('--mode', type=int, required=True)
  args = parser.parse_args()
  root = args.case
  p = json.loads((root / 'input/parameters.json').read_text())
  data = np.genfromtxt(root / 'ransdata.txt', names=True)
  modes = np.load(root / f'{args.label}_modes.npz')
  if not 0 <= args.mode < len(modes['eigenvalues']):
    parser.error('Mode index is outside the saved eigenvalue list')
  # Sparse assembly uses radial-major ordering; input row order need not match it.
  data = np.sort(data, order=['n', 's'])
  q = modes['modes'][:, args.mode].reshape(len(data), len(modes['scales']))
  phase = np.angle(q[np.argmax(abs(q[:, 2])), 2])
  velocity = (q[:, 1:3] * np.exp(-1j * phase)).real
  velocity /= max(np.max(abs(velocity)), 1e-300)
  triangulation = mtri.Triangulation(data['x'] / p['D_m'], data['y'] / p['D_m'])
  # Exclude triangles spanning the solid interior using the wall-ring polygon.
  from matplotlib.path import Path as PlotPath

  wall = np.column_stack([data['x'][data['n'] == 1], data['y'][data['n'] == 1]]) / p['D_m']
  centers = np.column_stack([triangulation.x, triangulation.y])[triangulation.triangles].mean(
    axis=1
  )
  triangulation.set_mask(PlotPath(wall).contains_points(centers))
  fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
  for i, ax in enumerate(axes):
    contour = ax.tricontourf(
      triangulation, velocity[:, i], levels=np.linspace(-1, 1, 41), cmap='RdBu_r'
    )
    ax.fill(wall[:, 0], wall[:, 1], color='0.85', edgecolor='0.2')
    ax.set(xlim=(-1, 18), ylim=(-4, 4), xlabel='x/D', ylabel='y/D', title=["Re(u')", "Re(v')"][i])
    ax.set_aspect('equal')
    fig.colorbar(contour, ax=ax)
  value = modes['eigenvalues'][args.mode]
  fig.suptitle(
    f'{p["model"]} | mode {args.mode} | lambda D/U = {value.real:.6f} {value.imag:+.6f}i'
  )
  for extension in ['png', 'pdf']:
    fig.savefig(root / f'{args.label}_mode_{args.mode}.{extension}', dpi=180)
  plt.close(fig)


if __name__ == '__main__':
  main()
