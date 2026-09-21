"""Export a delivered solver-output case directory into the layout import_case.py expects.

用例目录里的原始交付物是求解器直接写出的结果：``config.json``、``run.log``、
``mesh.txt`` 与 ``field/step_*.dat``。其中场文件的 TITLE 只有步数，没有模型与热力学
标记，也没有 ``baseflow.dat`` 与 ``parameters.json``。本脚本把这三件事补齐：

1. 从 ``run.log`` 找出收敛步与收敛判据数值；
2. 从 ``config.json`` 与场文件实测来流反算 Ma、Re、U、rho、mu 与迎角，并逐项核对；
3. 写出 ``parameters.json``（含来源说明与校验记录）和 ``baseflow.dat``
   （仅重写 TITLE 行，数据行逐字节保留）。

用法：``uv run python tools/export_case.py runs/<case>``。SA 版本信息按算例提供者的
说明记录为 Crouch 2007（含 C5 应变率项、Cv=717.625）；原始场文件没有标记，因此来源写在
``parameters.json`` 的 ``provenance`` 里，同时在 ``provenance.baseflow_note`` 记录基流端
自带的物理版本约定。
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

GAMMA, R, CP = 1.4, 287.05, 1004.675
MU0, T0, TS = 1.716e-5, 273.15, 110.4
FIELDS = 9
TITLE = (
  'TITLE="step {step} model=sa sa_model=crouch2007-eq2.1.5-v1 '
  'sa_formulation=crouch-2007 thermo=ideal-air-cv717625-v1"'
)
# 求解器有两种收敛判据措辞：相对更新量，或各流动量下降的量级数。
CONVERGED = re.compile(
  r'Converged at step (\d+), (?:relative_update=([0-9.eE+-]+)'
  r'|every field dropped at least ([0-9.]+) decades)'
)
BASE_FLOW_NOTE = (
  '基流端记录 Cv=717.645，其 SA 源项不含 C5 项；本算例的 sa_formulation 与 Cv 按基流记录采用'
)


def sound_speed(temperature):
  return math.sqrt(GAMMA * R * temperature)


def dynamic_viscosity(temperature):
  return MU0 * (temperature / T0) ** 1.5 * (T0 + TS) / (temperature + TS)


def read_wall_faces(mesh):
  """Return the wall ring face count, ring coordinates and the declared counts."""
  lines = mesh.read_text(encoding='utf-8').splitlines()
  nodes, faces, cells, groups = (int(value) for value in lines[0].split())
  if nodes < 4 or groups != 3:
    raise ValueError('Expected a single-block mesh with three boundary groups')
  names, kinds = [], []
  for index in range(1, 1 + groups):
    name, equal, kind = lines[index].split()
    if equal != '=':
      raise ValueError(f'Invalid boundary group header: {lines[index]}')
    names.append(name)
    kinds.append(kind)
  if kinds != ['INTER', 'WALL', 'FAR']:
    raise ValueError('Expected the INTER, WALL and FAR groups in this order')
  position, wall = lines.index('(edge)') + 1, None
  for name, kind in zip(names, kinds):
    if lines[position] != name:
      raise ValueError(f'Edge block {name} does not match the declared group order')
    position += 1
    rows = []
    while lines[position] != '(end)':
      rows.append([int(value) for value in lines[position].split()])
      position += 1
    position += 1
    if kind == 'WALL':
      wall = rows
  if not wall or cells % len(wall):
    raise ValueError('The wall ring faces must divide the cell count evenly')
  first = lines.index('(node)') + 1
  coordinates = {}
  for line in lines[first : first + nodes]:
    index, x, y = line.split()
    coordinates[int(index)] = (float(x), float(y))
  return len(wall), [coordinates[row[1]] for row in wall], (nodes, faces, cells)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('case', type=Path, help='delivered case directory containing config.json')
  args = parser.parse_args()
  root = args.case.resolve()
  config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
  farfield = config['farfield']
  log = (root / 'run.log').read_text(encoding='utf-8')
  match = CONVERGED.search(log)
  if not match:
    raise SystemExit('run.log does not record a converged state')
  step = int(match.group(1))
  relative_update = float(match.group(2)) if match.group(2) else None
  decades = float(match.group(3)) if match.group(3) else None
  dump = root / 'field' / f'step_{step:06d}.dat'
  if not dump.is_file():
    raise SystemExit(f'missing converged dump: {dump}')
  nt, ring, counts = read_wall_faces(root / 'mesh.txt')
  rows = dump.read_text(encoding='utf-8').splitlines()
  q = [[float(value) for value in line.split()] for line in rows[2:]]
  if any(len(row) != FIELDS for row in q):
    raise SystemExit('field rows must all hold nine values')
  cells = len(q)
  if cells % nt or counts != ((cells // nt + 1) * nt, (2 * (cells // nt) + 1) * nt, cells):
    raise SystemExit(f'field cell count {cells} disagrees with the mesh counts {counts}')
  nr = cells // nt
  outer = q[(nr - 1) * nt : nr * nt]
  alpha = farfield['alpha']
  ma = farfield['Ma']
  temperature = farfield['T']
  pressure = farfield['p']
  speed = ma * sound_speed(temperature)
  density = pressure / (R * temperature)
  mu = dynamic_viscosity(temperature)
  outer_u = sum(row[3] for row in outer) / nt
  outer_v = sum(row[4] for row in outer) / nt
  outer_T = sum(row[5] for row in outer) / nt
  outer_p = sum(row[6] for row in outer) / nt
  checks = {
    'ring_u_m_s': outer_u,
    'ring_v_m_s': outer_v,
    'ring_T_K': outer_T,
    'ring_p_Pa': outer_p,
    'ring_alpha_deg': math.degrees(math.atan2(outer_v, outer_u)),
    'ring_Ma': math.hypot(outer_u, outer_v) / sound_speed(outer_T),
    'T_deviation_K': outer_T - temperature,
    'p_relative_deviation': outer_p / pressure - 1.0,
  }
  checks['alpha_deviation_deg'] = checks['ring_alpha_deg'] - alpha
  checks['note'] = 'outer ring mean; the airfoil still perturbs the 10-chord boundary'
  if (
    abs(checks['alpha_deviation_deg']) > 0.5
    or abs(checks['T_deviation_K']) > 1.0
    or abs(checks['p_relative_deviation']) > 0.01
    or abs(checks['ring_Ma'] - ma) > 0.01
  ):
    raise SystemExit(f'farfield ring disagrees with config.json: {checks}')
  chord = max(point[0] for point in ring) - min(point[0] for point in ring)
  reference_chord = 1.0
  parameters = {
    'Ma': ma,
    'Re': density * speed * reference_chord / mu,
    'T_K': temperature,
    'p_Pa': pressure,
    'rho_kg_m3': density,
    'U_m_s': speed,
    'mu_Pa_s': mu,
    'alpha_deg': alpha,
    'D_m': reference_chord,
    'chord_measured_m': chord,
    'nt': nt,
    'nr': nr,
    'node_count': (nr + 1) * nt,
    'face_count': (2 * nr + 1) * nt,
    'cell_count': cells,
    'model': 'sa',
    'converged': True,
    'thermodynamics': {
      'version': 'ideal-air-cv717625-v1',
      'gamma': GAMMA,
      'R': R,
      'Cp': CP,
      'Cv': CP - R,
    },
    'sa_formulation': 'crouch-2007',
    'sa_model': 'crouch2007-eq2.1.5-v1',
    'convergence': {
      'step': step,
      'relative_update': relative_update,
      'decades': decades,
      'criterion': f'solver run.log: {match.group(0)}',
    },
    'farfield_ring_check': checks,
    'provenance': {
      'case': f'steady SA-RANS on the bundled O-grid, delivered as {root.name}',
      'source_field': f'field/step_{step:06d}.dat',
      'source_title': rows[0],
      'metadata': 'config.json and run.log of this case; SA formulation and Cv per the case owner',
      'note': 'the delivered field carries no model/thermo markers, so provenance is recorded here',
      'baseflow_note': BASE_FLOW_NOTE,
      'field_sha256': hashlib.sha256(dump.read_bytes()).hexdigest(),
      'mesh_sha256': hashlib.sha256((root / 'mesh.txt').read_bytes()).hexdigest(),
    },
  }
  (root / 'parameters.json').write_text(json.dumps(parameters, indent=2) + '\n', encoding='utf-8')
  (root / 'baseflow.dat').write_text(TITLE.format(step=step) + '\n' + '\n'.join(rows[1:]) + '\n')
  print(f'parameters.json: Ma={ma:.9f} Re={parameters["Re"]:.6e} alpha={alpha:.6f} deg')
  print(f'  U={speed:.9f} m/s rho={density:.12e} mu={mu:.12e}')
  print(f'  nt={nt} nr={nr} cells={cells} chord={chord:.6f}')
  print(
    f'  ring check: alpha={checks["ring_alpha_deg"]:.4f} deg T={outer_T:.4f} K '
    f'p={outer_p:.4f} Pa Ma={checks["ring_Ma"]:.6f}'
  )
  print(f'  converged at step {step}: {match.group(0)}')
  print('baseflow.dat: title rewritten, data lines preserved')


if __name__ == '__main__':
  main()
