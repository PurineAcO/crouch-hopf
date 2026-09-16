"""Import a converged single-block O-grid base flow delivered by a steady RANS run.

用例目录须包含 ``parameters.json``、``baseflow.dat``（或 ``field/step_*.dat``）与 ``mesh.txt``。
网格必须是单块 O 型网格，且面编号符合导入格式约定：环向面 ``i*ns + j + 1``
连接节点 ``(i, j)`` 与 ``(i, j+1)``，径向面 ``(nr+1)*ns + i*ns + j + 1`` 连接
节点 ``(i, j)`` 与 ``(i+1, j)``。导入器逐条核对这一约定，不假设几何形状。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crouch'))
from models import validate_thermodynamics


def read_mesh(path, ns, nr):
  """Return (xy, wall_midpoints) after checking the O-grid numbering convention."""
  lines = path.read_text().splitlines()
  nodes, faces, cells, groups = (int(value) for value in lines[0].split())
  if groups != 3 or (nodes, faces, cells) != ((nr + 1) * ns, (2 * nr + 1) * ns, nr * ns):
    raise ValueError('Mesh counts disagree with the O-grid nt/nr in parameters.json')
  names, kinds = [], []
  for index in range(1, 1 + groups):
    name, equal, kind = lines[index].split()
    if equal != '=':
      raise ValueError(f'Invalid boundary group header: {lines[index]}')
    names.append(name)
    kinds.append(kind)
  if kinds != ['INTER', 'WALL', 'FAR']:
    raise ValueError('Mesh must declare the INTER, WALL and FAR groups in this order')
  start = lines.index('(node)') + 1
  node_rows = lines[start : start + nodes]
  if any(int(line.split()[0]) != position for position, line in enumerate(node_rows, 1)):
    raise ValueError('Node block must be sequentially numbered')
  xy = np.array([list(map(float, line.split()[1:])) for line in node_rows])
  if xy.shape != (nodes, 2) or not np.isfinite(xy).all():
    raise ValueError('Node block must contain two finite coordinates per node')
  xy = xy.reshape(nr + 1, ns, 2)
  blocks, position = {}, lines.index('(edge)') + 1
  for name in names:
    if lines[position] != name:
      raise ValueError(f'Edge block {name} does not match the declared group order')
    position += 1
    rows = []
    while lines[position] != '(end)':
      fields = lines[position].split()
      if len(fields) != 5:
        raise ValueError(f'Invalid face row: {lines[position]}')
      rows.append([int(value) for value in fields])
      position += 1
    position += 1
    blocks[name] = np.array(rows, dtype=np.int64)
  if sorted(np.concatenate([block[:, 0] for block in blocks.values()])) != list(
    range(1, faces + 1)
  ):
    raise ValueError('Face indices must be unique and cover 1..face_count')
  interior, wall, far = (blocks[name] for name in names)
  if interior.shape[0] != (2 * nr - 1) * ns or wall.shape[0] != ns or far.shape[0] != ns:
    raise ValueError('Face groups must contain the O-grid interior, wall and farfield counts')
  if np.any((interior[:, 3] == 0) | (interior[:, 4] == 0)):
    raise ValueError('Interior faces must have two adjacent cells')
  for block, label in ((wall, 'wall'), (far, 'farfield')):
    if np.any((block[:, 3] == 0) == (block[:, 4] == 0)):
      raise ValueError(f'{label} faces must have exactly one adjacent cell')
  wall_edges = {frozenset(pair) for pair in wall[:, 1:3].tolist()}
  far_edges = {frozenset(pair) for pair in far[:, 1:3].tolist()}
  if wall_edges != {frozenset((j + 1, (j + 1) % ns + 1)) for j in range(ns)}:
    raise ValueError('Wall faces must connect the i=0 node ring')
  if far_edges != {frozenset((nr * ns + j + 1, nr * ns + (j + 1) % ns + 1)) for j in range(ns)}:
    raise ValueError('Farfield faces must connect the i=nr node ring')
  ring_edges = {
    frozenset((i * ns + j + 1, i * ns + (j + 1) % ns + 1)) for i in range(nr + 1) for j in range(ns)
  }
  radial_edges = {
    frozenset((i * ns + j + 1, (i + 1) * ns + j + 1)) for i in range(nr) for j in range(ns)
  }
  if not {frozenset(pair) for pair in interior[:, 1:3].tolist()} <= (ring_edges | radial_edges):
    raise ValueError('Interior faces must follow the ring/radial O-grid connectivity')
  if sorted(wall[:, 3] + wall[:, 4]) != list(range(1, ns + 1)):
    raise ValueError('Wall ring faces must connect the i=0 cells')
  if sorted(far[:, 3] + far[:, 4]) != list(range(nr * ns - ns + 1, nr * ns + 1)):
    raise ValueError('Farfield faces must connect the i=nr-1 cells')
  first_cell = lines.index('(cell)') + 1
  cell_rows = np.array(
    [list(map(int, line.split())) for line in lines[first_cell : first_cell + cells]]
  )
  if cell_rows.shape != (cells, 5) or not np.array_equal(cell_rows[:, 0], np.arange(1, cells + 1)):
    raise ValueError('Cell block must have five columns and sequential indices')
  if np.any(cell_rows[:, 1:] < 1) or np.any(cell_rows[:, 1:] > faces):
    raise ValueError('Cell face indices out of range')
  adjacency = np.zeros(faces + 1, dtype=np.int64)
  for block in blocks.values():
    if np.any(block[:, 1:3] < 1) or np.any(block[:, 1:3] > nodes):
      raise ValueError('Face nodes out of range')
    adjacency[block[:, 0]] = (block[:, 3] != 0).astype(np.int64) + (block[:, 4] != 0)
  members = np.bincount(cell_rows[:, 1:].ravel(), minlength=faces + 1)
  if not np.array_equal(adjacency[1:], members[1:]):
    raise ValueError('Face adjacency and cell face lists disagree')
  return xy, 0.5 * (xy[0] + np.roll(xy[0], -1, axis=0))


def find_field(source, parameters):
  """Return the field recorded by the case runner, or the last numbered dump."""
  recorded = source / 'baseflow.dat'
  if recorded.is_file():
    return recorded
  dumps = sorted((source / 'field').glob('step_*.dat'), key=lambda path: int(path.stem[6:]))
  if not dumps:
    raise ValueError(f'Missing baseflow.dat and field dumps under {source}')
  steps = parameters.get('achieved_steps')
  matching = [path for path in dumps if steps and int(path.stem[6:]) == steps]
  return matching[0] if matching else dumps[-1]


def convert(source, root, symmetrize=False, field=None, in_place=False):
  source, root = source.resolve(), root.resolve()
  parameters_path = source / 'parameters.json'
  if not parameters_path.is_file():
    raise ValueError('Missing parameters.json; run cases/<case>/run.py to export the base flow')
  p = json.loads(parameters_path.read_text())
  if p['model'] not in ('laminar', 'sa'):
    raise ValueError('Unknown base-flow model')
  if p['model'] == 'sa' and p.get('sa_formulation') != 'crouch-2007':
    raise ValueError('SA base flow must record sa_formulation=crouch-2007')
  if p['converged'] is not True:
    raise ValueError('Base flow has not converged; rerun the steady solver with more steps')
  path = Path(field).resolve() if field else find_field(source, p)
  with path.open() as stream:
    title = stream.readline()
  if f'model={p["model"]}' not in title:
    raise ValueError('Field title and parameters disagree on model')
  if p['model'] == 'sa' and 'sa_formulation=crouch-2007' not in title:
    raise ValueError('SA field must record sa_formulation=crouch-2007')
  if 'thermo=ideal-air-cv717625-v1' not in title:
    raise ValueError('Field must record the corrected thermodynamics version')
  ns, nn = p['nt'], p['nr']
  q = np.loadtxt(path, skiprows=2).reshape(nn, ns, 9)
  if not np.isfinite(q).all() or np.any(q[:, :, [2, 5, 6]] <= 0):
    raise ValueError('Base flow must be finite with positive density, temperature and pressure')
  if np.any(q[:, :, 8] < 0) or (p['model'] == 'laminar' and np.any(q[:, :, 8] != 0)):
    raise ValueError('Invalid nu-tilde for the selected model')
  validate_thermodynamics(p)
  if in_place:
    if any((root / name).exists() for name in ('ransdata.txt', 'edge.txt', 'input')):
      raise ValueError(f'In-place conversion refuses to overwrite converted input in {root}')
  elif root.exists():
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
  xy, wall_midpoints = read_mesh(source / 'mesh.txt', ns, nn)
  nextxy = np.roll(xy, -1, axis=1)

  def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

  area = 0.5 * np.abs(
    cross(xy[:-1], xy[1:])
    + cross(xy[1:], nextxy[1:])
    + cross(nextxy[1:], nextxy[:-1])
    + cross(nextxy[:-1], xy[:-1])
  )
  distance = np.min(
    np.linalg.norm(q[:, :, None, :2] - wall_midpoints[None, None, :, :], axis=3), axis=2
  )
  (root / 'input').mkdir(parents=True)
  (root / 'input/parameters.json').write_text(json.dumps(p, indent=2) + '\n')
  (root / 'import.json').write_text(
    json.dumps(
      {
        'source': str(source),
        'field': path.name,
        'field_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'mesh_sha256': hashlib.sha256((source / 'mesh.txt').read_bytes()).hexdigest(),
        'wall_distance': 'minimum distance to the wall face midpoints',
        'symmetry_change': change,
      },
      indent=2,
    )
    + '\n'
  )
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
  print(f'Converted {ns} x {nn} cells to {root}; model={p["model"]}; field={path.name}')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('source', type=Path)
  parser.add_argument('output', type=Path)
  parser.add_argument('--field', type=Path, help='override the converged field file')
  parser.add_argument('--symmetrize', action='store_true', help='Remove reflection roundoff only')
  parser.add_argument(
    '--in-place',
    action='store_true',
    help='Write into the case directory itself; refuses to overwrite converted input',
  )
  args = parser.parse_args()
  convert(args.source, args.output, args.symmetrize, args.field, args.in_place)


if __name__ == '__main__':
  main()
