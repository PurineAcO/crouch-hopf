"""Independent analytic actions on skew grids; no physical-flux differentiation.

The nonuniform-grid oracle applies analytic Aq'+B grad(q') coefficients directly
using face averages and the discrete Gauss theorem. It does not reuse grad.py's
weight dictionaries or viscous.py's slot mappings. A separate physical quadratic
heat field checks the continuum diffusion sign, geometry, and magnitude on affine
skew grids, where this reconstruction reproduces quadratic derivatives exactly.
"""

import classconfig as cc
import numpy as np
import pytest
import viscous
from linearization import source_coefficients, viscous_coefficients
from models import FlowModel

# Original slots, independently enumerated in nn,nw,n,ne,ww,w,c,e,ee,sw,s,se,ss order.
OFFSETS = [
  (0, 2),
  (-1, 1),
  (0, 1),
  (1, 1),
  (-2, 0),
  (-1, 0),
  (0, 0),
  (1, 0),
  (2, 0),
  (-1, -1),
  (0, -1),
  (1, -1),
  (0, -2),
]
GRADIENT_FIELDS = ['rhograd', 'ugrad', 'vgrad', 'Tgrad', 'miublgrad']


def _base_state(x, y):
  """Physical manufactured base state and its closed-form Cartesian gradient."""
  q = np.array(
    [
      1.2 + 0.01 * x + 0.02 * y,
      30 + 0.2 * x - 0.1 * y,
      2 + 0.3 * x + 0.15 * y,
      300 + 0.5 * x + 0.2 * y,
      0.001 + 1e-6 * (x * x + y * y),
    ]
  )
  g = np.array([[0.01, 0.02], [0.2, -0.1], [0.3, 0.15], [0.5, 0.2], [2e-6 * x, 2e-6 * y]])
  return q, g


def _mesh(warp=0.025, angle=0.61, stretch=1.7, uniform=False):
  """Seven by seven genuine quadrilateral cells, with no artificial periodic seam."""
  rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

  def vertex(i, j):
    s, n = i - 3.5, j - 3.5
    return rotation @ [
      stretch * (s + 0.35 * n + warp * n * n),
      (n + 0.15 * s + warp * s * s) / stretch,
    ]

  cells, base, faces = {}, {}, []
  for j in range(7):
    for i in range(7):
      polygon = np.array([vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)])
      x, y = polygon.mean(axis=0)
      area = 0.5 * (
        polygon[:, 0] @ np.roll(polygon[:, 1], -1) - polygon[:, 1] @ np.roll(polygon[:, 0], -1)
      )
      assert area > 0
      q, g = _base_state(x, y)
      if uniform:
        q, g = np.array([1.2, 0.0, 0.0, 300.0, 0.0]), np.zeros((5, 2))
      c = cc.cell_class((i, j), x, y, *q, area, 0.3)
      for name, vector in zip(GRADIENT_FIELDS, g):
        setattr(c, name, vector.copy())
      cells[i, j], base[i, j] = c, (q, g)
  for (i, j), c in cells.items():
    if i < 6:
      low, high = vertex(i + 1, j), vertex(i + 1, j + 1)
      tangent = high - low
      normal = np.array([tangent[1], -tangent[0]])
      other = cells[i + 1, j]
      f = cc.face_class('WE', (low + high) / 2, [normal, tangent], other, c)
      c.east = other.west = f
      faces.append(f)
    if j < 6:
      low, high = vertex(i, j + 1), vertex(i + 1, j + 1)
      tangent = high - low
      normal = np.array([-tangent[1], tangent[0]])
      other = cells[i, j + 1]
      f = cc.face_class('NS', (low + high) / 2, [normal, tangent], other, c)
      c.north = other.south = f
      faces.append(f)
  for f in faces:
    viscous.prepare_face_diffusion(f)
  return cells, base


def _field(cells, kind):
  rng = np.random.default_rng(818)
  coefficients = rng.normal(size=(5, 6))
  values = {}
  for index, c in cells.items():
    x, y = c.x, c.y
    values[index] = (
      coefficients @ [1, x, y, x * x, x * y, y * y] if kind == 'polynomial' else rng.normal(size=5)
    )
  return values


def _faces(c):
  return [(c.east, 1), (c.west, -1), (c.north, 1), (c.south, -1)]


def _gauss_gradient(c, values):
  # Direct surface-integral definition, independent of all production weights.
  result = np.zeros((5, 2))
  for face, sign in _faces(c):
    average = (values[face.me.index] + values[face.nei.index]) / 2
    result += np.outer(average, sign * face.jacobian[0]) / c.vol
  return result


def _block_action(c, values):
  assert np.shape(c.influence) == (13, 5, 5)
  i, j = c.index
  return sum(block @ values[i + di, j + dj] for block, (di, dj) in zip(c.influence, OFFSETS))


@pytest.mark.parametrize('model', list(FlowModel))
@pytest.mark.parametrize('kind', ['polynomial', 'random'])
def test_skew_nonuniform_viscous_blocks_match_direct_analytic_action(monkeypatch, model, kind):
  monkeypatch.setattr(cc, 'flow_model', model)
  cells, base = _mesh()
  values = _field(cells, kind)
  c = cells[3, 3]
  assert abs(c.east.jacobian[0] @ c.north.jacobian[0]) > 0.01
  assert np.ptp([cell.vol for cell in cells.values()]) > 0.01
  expected = np.zeros(5)
  for face, sign in _faces(c):
    left, right = face.nei.index, face.me.index
    q = (base[left][0] + base[right][0]) / 2
    g = (base[left][1] + base[right][1]) / 2
    A, B = viscous_coefficients(q, g, face.jacobian[0])
    dq = (values[left] + values[right]) / 2
    dg = (_gauss_gradient(face.nei, values) + _gauss_gradient(face.me, values)) / 2
    flux = A @ dq + np.einsum('ija,ja->i', B, dg)
    expected += sign * flux / c.vol
  viscous.cell_diffusion(c)
  np.testing.assert_allclose(_block_action(c, values), expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('kind', ['polynomial', 'random'])
def test_skew_nonuniform_source_blocks_match_direct_analytic_action(kind):
  cells, base = _mesh()
  values = _field(cells, kind)
  c = cells[3, 3]
  q, g = base[c.index]
  A, B = source_coefficients(q, g, c.sad)
  expected = np.zeros(5)
  expected[4] = A @ values[c.index] + np.sum(B * _gauss_gradient(c, values))
  viscous.cell_source(c)
  np.testing.assert_allclose(_block_action(c, values), expected, rtol=2e-12, atol=2e-12)
  np.testing.assert_array_equal(np.asarray(c.influence)[:, :4, :], 0)


@pytest.mark.parametrize('angle', [0.1, 1.9])
@pytest.mark.parametrize('stretch', [1.0, 4.0])
def test_physical_quadratic_heat_diffusion_on_affine_skew_grid(monkeypatch, angle, stretch):
  monkeypatch.setattr(cc, 'flow_model', FlowModel.LAMINAR)
  cells, _ = _mesh(warp=0, angle=angle, stretch=stretch, uniform=True)
  # T' = x² + 2xy + 3y² + 4x - y + 2 has Laplacian 8 in physical coordinates.
  values = {
    index: np.array([0, 0, 0, c.x * c.x + 2 * c.x * c.y + 3 * c.y * c.y + 4 * c.x - c.y + 2, 0])
    for index, c in cells.items()
  }
  c = cells[3, 3]
  mu = cc.mu0 * (300 / cc.T0) ** 1.5 * (cc.T0 + cc.Ts) / (300 + cc.Ts)
  expected = np.array([0, 0, 0, 8 * cc.cp * mu / cc.Pr, 0])
  viscous.cell_diffusion(c)
  np.testing.assert_allclose(_block_action(c, values), expected, rtol=2e-12, atol=2e-12)
