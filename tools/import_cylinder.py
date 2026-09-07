"""Import a converged cylinder run produced by PurineCFD-R2 cases/cylinder/run.py."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crouch'))
from models import validate_thermodynamics


def convert(source, root, symmetrize=False):
  source, root = source.resolve(), root.resolve()
  p = json.loads((source / 'parameters.json').read_text())
  if p['model'] not in ('laminar', 'sa'):
    raise ValueError('Unknown base-flow model')
  if p['model'] == 'sa' and p.get('sa_formulation') != 'crouch-2007':
    raise ValueError('SA base flow must record sa_formulation=crouch-2007')
  if p['converged'] is not True:
    raise ValueError('Base flow has not converged; rerun the steady solver with more steps')
  with (source / 'baseflow.dat').open() as stream:
    title = stream.readline()
  if f'model={p["model"]}' not in title:
    raise ValueError('Field title and parameters disagree on model')
  if p['model'] == 'sa' and 'sa_formulation=crouch-2007' not in title:
    raise ValueError('SA field must record sa_formulation=crouch-2007')
  ns, nn = p['nt'], p['nr']
  q = np.loadtxt(source / 'baseflow.dat', skiprows=2).reshape(nn, ns, 9)
  if not np.isfinite(q).all() or np.any(q[:, :, [2, 5, 6]] <= 0):
    raise ValueError('Base flow must be finite with positive density, temperature and pressure')
  if np.any(q[:, :, 8] < 0) or (p['model'] == 'laminar' and np.any(q[:, :, 8] != 0)):
    raise ValueError('Invalid nu-tilde for the selected model')
  validate_thermodynamics(p)
  if 'thermo=ideal-air-cv717625-v1' not in title:
    raise ValueError('Field must record the corrected thermodynamics version')
  if root.exists():
    raise ValueError(f'Output exists; choose a new directory: {root}')
  change = 0.0
  if symmetrize:
    if ns % 2:
      raise ValueError('Reflection requires an even azimuthal count')
    columns = [2, 3, 4, 5, 8]
    original = q[:, :, columns].copy()
    symmetric = (original + original[:, ::-1] * [1, 1, -1, 1, 1]) / 2
    scales = [p['rho_kg_m3'], p['U_m_s'], p['U_m_s'], p['T_K'], 1.0]
    change = float(np.max(abs(symmetric - original) / scales))
    if change > 1e-10:
      raise ValueError(f'Base flow is not reflection symmetric to roundoff: {change}')
    q[:, :, columns] = symmetric
  lines = (source / 'mesh.txt').read_text().splitlines()
  start = lines.index('(node)') + 1
  xy = np.array(
    [list(map(float, line.split()[1:])) for line in lines[start : start + (nn + 1) * ns]]
  ).reshape(nn + 1, ns, 2)
  if not np.allclose(np.linalg.norm(xy[0], axis=1), p['D_m'] / 2, rtol=1e-10):
    raise ValueError('This importer supports the bundled origin-centered cylinder O-grid only')
  (root / 'input').mkdir(parents=True)
  (root / 'input/parameters.json').write_text(json.dumps(p, indent=2) + '\n')
  (root / 'import.json').write_text(
    json.dumps({'source': str(source), 'symmetry_change': change}, indent=2) + '\n'
  )
  nextxy = np.roll(xy, -1, axis=1)

  def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

  area = 0.5 * np.abs(
    cross(xy[:-1], xy[1:])
    + cross(xy[1:], nextxy[1:])
    + cross(nextxy[1:], nextxy[:-1])
    + cross(nextxy[:-1], xy[:-1])
  )
  wallmid = 0.5 * (xy[0] + nextxy[0])
  distance = np.linalg.norm(q[:, :, :2] - wallmid[None], axis=2)
  rows = []
  for n in range(nn):
    for s in range(ns):
      rows.append(
        [s + 1, n + 1, *q[n, s, :2], distance[n, s], area[n, s], *q[n, s, [2, 3, 4, 5, 8]]]
      )
  np.savetxt(
    root / 'ransdata.txt', rows, header='s n x y sad vol rho u v T miubl', comments='', fmt='%.16e'
  )
  with (root / 'edge.txt').open('w') as f:
    f.write('type s n id c1_s c1_n c1_id c2_s c2_n c2_id nx ny mx my\n')
    idx = 0
    for n in range(nn + 1):
      for s in range(ns):
        idx += 1
        a, b = xy[n, s], nextxy[n, s]
        delta = b - a
        normal = np.array([delta[1], -delta[0]])
        mid = (a + b) / 2
        vals = [s + 1, n + 1, idx, s + 1, n, 0, s + 1, n + 1 if n < nn else 0, 0, *normal, *mid]
        f.write('NS ' + ' '.join(map(str, vals)) + '\n')
    for n in range(nn):
      for s in range(ns):
        idx += 1
        a, b = xy[n, s], xy[n + 1, s]
        delta = b - a
        normal = np.array([-delta[1], delta[0]])
        vals = [
          s + 1,
          n + 1,
          idx,
          (s - 1) % ns + 1,
          n + 1,
          0,
          s + 1,
          n + 1,
          0,
          *normal,
          *(a + b) / 2,
        ]
        f.write('WE ' + ' '.join(map(str, vals)) + '\n')
  print(f'Converted {ns} x {nn} cells to {root}; model={p["model"]}')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('source', type=Path)
  parser.add_argument('output', type=Path)
  parser.add_argument('--symmetrize', action='store_true', help='Remove reflection roundoff only')
  args = parser.parse_args()
  convert(args.source, args.output, args.symmetrize)


if __name__ == '__main__':
  main()
