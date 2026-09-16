"""Importer contract tests: reject unusable base flows before writing any output."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from import_case import convert

THERMO = 'thermo=ideal-air-cv717625-v1'
PARAMETERS = {
  'nt': 4,
  'nr': 2,
  'D_m': 1.0,
  'U_m_s': 1.0,
  'rho_kg_m3': 1.0,
  'T_K': 300.0,
  'mu_Pa_s': 1e-5,
  'model': 'sa',
  'converged': True,
  'sa_formulation': 'crouch-2007',
  'thermodynamics': {'R': 287.05, 'Cp': 1004.675, 'Cv': 717.625, 'gamma': 1.4},
}


def write_case(root, nu=1e-5, title_model=None, thermo=THERMO, **overrides):
  """Write a minimal single-block O-grid case using the solver numbering."""
  root.mkdir(parents=True, exist_ok=True)
  parameters = dict(PARAMETERS, **overrides)
  nt, nr = parameters['nt'], parameters['nr']
  nodes, cells = (nr + 1) * nt, nr * nt

  def ring(i, j):
    return i * nt + (j % nt) + 1

  def radial(i, j):
    return nodes + i * nt + (j % nt) + 1

  def cell(i, j):
    return i * nt + (j % nt) + 1

  lines = [f'{nodes} {nodes + cells} {cells} 3']
  lines += ['interior = INTER', 'airfoil = WALL', 'farfield = FAR', '(node)']
  for i in range(nr + 1):
    radius = 0.5 + 0.75 * i
    for j in range(nt):
      angle = 2.0 * math.pi * j / nt
      lines.append(f'{ring(i, j)} {radius * math.cos(angle):.16e} {radius * math.sin(angle):.16e}')
  lines += ['(end)', '(edge)']
  wall = [(ring(0, j), ring(0, j), ring(0, j + 1), cell(0, j), 0) for j in range(nt)]
  far = [(ring(nr, j), ring(nr, j), ring(nr, j + 1), cell(nr - 1, j), 0) for j in range(nt)]
  interior = [
    (ring(i, j), ring(i, j), ring(i, j + 1), cell(i - 1, j), cell(i, j))
    for i in range(1, nr)
    for j in range(nt)
  ] + [
    (radial(i, j), ring(i, j), ring(i + 1, j), cell(i, j), cell(i, j - 1))
    for i in range(nr)
    for j in range(nt)
  ]
  for name, block in (('interior', interior), ('airfoil', wall), ('farfield', far)):
    lines.append(name)
    lines += [' '.join(str(value) for value in row) for row in block]
    lines.append('(end)')
  lines += ['(end)', '(cell)']
  lines += [
    ' '.join(
      str(value)
      for value in (cell(i, j), ring(i, j), ring(i + 1, j), radial(i, j), radial(i, j + 1))
    )
    for i in range(nr)
    for j in range(nt)
  ]
  (root / 'mesh.txt').write_text('\n'.join(lines) + '\n')
  (root / 'parameters.json').write_text(json.dumps(parameters) + '\n')
  rows = [
    [1.0, 0.1 * (i + 1), 0.2, 1.0, 0.0, 1.0, 300.0, 0.2, nu] for i in range(nr) for _ in range(nt)
  ]
  body = '\n'.join(' '.join(f'{value:.9e}' for value in row) for row in rows)
  (root / 'baseflow.dat').write_text(
    f'TITLE="step 200 model={title_model or parameters["model"]} '
    f'sa_formulation=crouch-2007 {thermo}"\nVARIABLES\n{body}\n'
  )
  return root


def test_converts_minimal_ogrid(tmp_path):
  source = write_case(tmp_path / 'source')
  target = tmp_path / 'out'
  convert(source, target)
  assert np.loadtxt(target / 'ransdata.txt', skiprows=1).shape == (8, 11)
  edge = (target / 'edge.txt').read_text().splitlines()
  assert len(edge) - 1 == (2 * 2 + 1) * 4
  assert {line.split()[0] for line in edge[1:]} == {'NS', 'WE'}
  recorded = json.loads((target / 'input/parameters.json').read_text())
  assert recorded['model'] == 'sa' and recorded['sa_formulation'] == 'crouch-2007'
  assert json.loads((target / 'import.json').read_text())['field'] == 'baseflow.dat'


@pytest.mark.parametrize(
  ('overrides', 'message'),
  [
    ({'converged': False}, 'has not converged'),
    ({'model': 'les'}, 'Unknown base-flow model'),
    ({'sa_formulation': 'legacy'}, 'sa_formulation=crouch-2007'),
    ({'title_model': 'laminar'}, 'disagree on model'),
    ({'thermo': 'thermo=ideal-air-cv717645-v1'}, 'corrected thermodynamics version'),
    ({'model': 'laminar', 'nu': 1.0}, 'Invalid nu-tilde'),
  ],
)
def test_rejects_unusable_baseflow(tmp_path, overrides, message):
  source = write_case(tmp_path / 'source', **overrides)
  target = tmp_path / 'out'
  with pytest.raises(ValueError, match=message):
    convert(source, target)
  assert not target.exists()


def test_rejects_broken_wall_ring(tmp_path):
  source = write_case(tmp_path / 'source')
  lines = (source / 'mesh.txt').read_text().splitlines()
  position = lines.index('airfoil') + 1
  fields = lines[position].split()
  fields[2] = '3'
  lines[position] = ' '.join(fields)
  (source / 'mesh.txt').write_text('\n'.join(lines) + '\n')
  with pytest.raises(ValueError, match='i=0 node ring'):
    convert(source, tmp_path / 'out')


def test_rejects_inconsistent_cell_section(tmp_path):
  source = write_case(tmp_path / 'source')
  lines = (source / 'mesh.txt').read_text().splitlines()
  position = lines.index('(cell)') + 1
  fields = lines[position].split()
  fields[4] = fields[3]
  lines[position] = ' '.join(fields)
  (source / 'mesh.txt').write_text('\n'.join(lines) + '\n')
  with pytest.raises(ValueError, match='adjacency and cell face lists disagree'):
    convert(source, tmp_path / 'out')


def test_in_place_refuses_existing_input(tmp_path):
  source = write_case(tmp_path / 'source')
  (source / 'ransdata.txt').write_text('existing\n')
  with pytest.raises(ValueError, match='refuses to overwrite'):
    convert(source, source, in_place=True)
