"""Plot real(u-hat) in the physical window and style of Crouch (2007), Fig. 2."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.path import Path as Polygon


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('case', type=Path)
  parser.add_argument('--mode', type=int, required=True)
  parser.add_argument('--label', default='wake')
  parser.add_argument(
    '--output', type=Path, required=True, help='Output basename without extension'
  )
  args = parser.parse_args()
  case = args.case.resolve()
  parameters = json.loads((case / 'input/parameters.json').read_text())
  if parameters['model'] != 'laminar':
    parser.error('The paper cylinder comparison requires a four-variable laminar mode')
  data = np.sort(np.genfromtxt(case / 'ransdata.txt', names=True), order=['n', 's'])
  modes = np.load(case / f'{args.label}_modes.npz')
  if not 0 <= args.mode < len(modes['eigenvalues']):
    parser.error('Mode index is outside the saved eigenvalue list')
  q = modes['modes'][:, args.mode].reshape(len(data), len(modes['scales']))
  x, y = data['x'] / parameters['D_m'], data['y'] / parameters['D_m']
  u = q[:, 1] * modes['scales'][1]
  # Only the arbitrary global phase is fixed in this window, not the mode itself.
  upper_wake = (x > 0) & (x < 15) & (y > 0) & (y < 5)
  if not upper_wake.any() or not np.isfinite(u).all():
    parser.error('No finite upper-wake mode data')
  anchor = np.flatnonzero(upper_wake)[np.argmax(abs(u[upper_wake]))]
  phase = float(np.angle(u[anchor]))
  rotated = u * np.exp(-1j * phase)
  normalization = float(np.max(abs(rotated.real)))
  if not np.isfinite(normalization) or normalization <= 0:
    parser.error('The u component must have nonzero finite amplitude')
  values = rotated.real / normalization
  wall = np.column_stack([x[data['n'] == 1], y[data['n'] == 1]])
  triangulation = mtri.Triangulation(x, y)
  centers = np.column_stack([x, y])[triangulation.triangles].mean(axis=1)
  triangulation.set_mask(Polygon(wall).contains_points(centers))
  plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11})
  fig, ax = plt.subplots(figsize=(5.8, 5.2), layout='constrained')
  # A bin centered on zero avoids drawing roundoff sign changes as two colors.
  field = ax.tricontourf(triangulation, values, levels=np.linspace(-1, 1, 82), cmap='jet')
  ax.add_patch(plt.Circle((0, 0), 0.5, facecolor='white', edgecolor='white', zorder=3))
  ax.set(xlim=(-5, 15), ylim=(-10, 10), xlabel=r'$x/D$', ylabel=r'$y/D$')
  ax.set_aspect('equal')
  ax.set_xticks([-5, 0, 5, 10, 15])
  ax.set_yticks([-10, -5, 0, 5, 10])
  ax.tick_params(direction='in', top=True, right=True)
  colorbar = fig.colorbar(field, ax=ax, fraction=0.045, pad=0.03, ticks=np.linspace(-1, 1, 11))
  colorbar.set_label(r'$\mathrm{Re}(\hat{u})/\max|\mathrm{Re}(\hat{u})|$')
  value = complex(modes['eigenvalues'][args.mode])
  assembly = json.loads((case / 'assembly.json').read_text())
  ax.set_title(
    f'Re={parameters["Re"]:g}, Ma={parameters["Ma"]:g}, '
    f'{parameters["nt"]}×{parameters["nr"]} cells\n'
    f'αH={assembly["alpha_H"]:g}, growth={value.real:.6f}, |ω|={abs(value.imag):.6f}',
    fontsize=10,
  )
  output = args.output.resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  for extension in ['png', 'pdf', 'svg']:
    fig.savefig(output.with_suffix(f'.{extension}'), dpi=220)
  plt.close(fig)
  np.savez_compressed(
    output.with_suffix('.npz'),
    x_D=x,
    y_D=y,
    u_real_normalized=values,
    u_complex_normalized=rotated / normalization,
    triangles=triangulation.triangles,
    triangle_mask=triangulation.mask,
  )
  info = {
    'source_case': str(case),
    'source_modes': f'{args.label}_modes.npz',
    'mode': args.mode,
    'parameters': parameters,
    'alpha_H': assembly['alpha_H'],
    'eigenvalue_D_U': [value.real, value.imag],
    'phase_rotation_radians': -phase,
    'phase_anchor_x_y_D': [float(x[anchor]), float(y[anchor])],
    'normalization': 'global maximum of absolute real u after phase rotation',
    'normalization_value': normalization,
    'window_D': [-5, 15, -10, 10],
    'field': 'real u perturbation only; no mean-flow addition',
  }
  output.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n')
  print(output)


if __name__ == '__main__':
  main()
