import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crouch'))
import classconfig as cc
import linearization as lin


def test_mass_inverse():
  from formmat import _primitive_map

  c = cc.cell_class((1, 1), 1.0, 0.0, 1.2e-5, 40.0, 3.0, 300.0, 0.2, 1.0, 1.0)
  np.testing.assert_allclose(
    _primitive_map(c) @ lin.mass_jacobian(lin.state(c)), np.eye(5), atol=2e-8
  )


def test_zero_vorticity_source_is_finite():
  q = np.array([1.2e-5, 40.0, 0.0, 300.0, 0.2])
  g = np.zeros((5, 2))
  a, b = lin.source_coefficients(q, g, 0.4)
  assert np.isfinite(a).all() and np.isfinite(b).all()


def test_uniform_cartesian_advection_and_diffusion_have_correct_signs():
  import convect
  import viscous

  cells = {}
  count = 7
  for j in range(count):
    for i in range(count):
      cells[i, j] = cc.cell_class((i, j), i, j, 1.0, 40.0, 0.0, 300.0, 0.2, 1.0, 1.0)
  for c in cells.values():
    c.jacobian = np.eye(2)
  faces = []
  for (i, j), c in cells.items():
    east = cells[(i + 1) % count, j]
    north = cells[i, (j + 1) % count]
    f = cc.face_class('WE', (i + 0.5, j), [[1.0, 0.0], [0.0, 1.0]], east, c)
    c.east = east.west = f
    faces.append(f)
    f = cc.face_class('NS', (i, j + 0.5), [[0.0, 1.0], [-1.0, 0.0]], north, c)
    c.north = north.south = f
    faces.append(f)
  for f in faces:
    viscous.prepare_face_diffusion(f)
  c = cells[3, 3]
  convect.convect_hybrid(c)
  np.testing.assert_allclose(np.sum(c.influence, axis=0), 0, atol=1e-8)
  offsets = {
    'nn': 0,
    'n': 0,
    'nw': -1,
    'ne': 1,
    'ww': -2,
    'w': -1,
    'c': 0,
    'e': 1,
    'ee': 2,
    'sw': -1,
    's': 0,
    'se': 1,
    'ss': 0,
  }
  mode = np.array([0, 0, 1, 0, 0])

  def symbol():
    return sum(
      np.exp(0.3j * offsets[name]) * (c.influence[idx] @ mode)[2] for name, idx in cc.dic.items()
    )

  adv = symbol()
  assert adv.real < 0 and adv.imag < 0
  assert abs(adv.imag + 40 * 0.3) / (40 * 0.3) < 0.001
  c.influence = [np.zeros((5, 5)) for _ in range(13)]
  viscous.cell_diffusion(c)
  np.testing.assert_allclose(np.sum(c.influence, axis=0), 0, atol=1e-12)
  diff = symbol()
  assert diff.real < 0 and abs(diff.imag) < 1e-12
  c.influence = [np.zeros((5, 5)) for _ in range(13)]
  viscous.cell_source(c)
  assert not np.any(np.asarray(c.influence)[:, :4, :])


@pytest.mark.parametrize('velocity', [-20.0, 0.0, 1e-16, -1e-16, 20.0])
@pytest.mark.parametrize('alpha', [0.0, 0.2, 1.0])
def test_paper_reconstructs_matrix_and_mode_separately(velocity, alpha, monkeypatch):
  from types import SimpleNamespace

  import convect

  monkeypatch.setattr(cc, 'alpha_H', alpha)
  cells = [
    cc.cell_class((i, 1), i, 0, 1 + 0.1 * i, velocity, 3, 290 + 8 * i, 0.2, 1, 1) for i in range(4)
  ]
  for i, c in enumerate(cells):
    c.jacobian = np.array([[1.0, 0.0], [0.0, 1 + 0.1 * i]])
  cells[1].west = SimpleNamespace(west=cells[0])
  cells[2].east = SimpleNamespace(east=cells[3])
  face = SimpleNamespace(direction='WE', west=cells[1], east=cells[2])
  perturbation = np.random.default_rng(34).normal(size=(4, 5))
  matrices = np.array([(1 + 0.1 * i) * c.F for i, c in enumerate(cells)])
  for i, c in enumerate(cells):
    matrices[i, 4] = (1 + 0.1 * i) * c.sa_convect_vec()[0]
  qm = (-perturbation[0] + 5 * perturbation[1] + 2 * perturbation[2]) / 6
  qp = (2 * perturbation[1] + 5 * perturbation[2] - perturbation[3]) / 6
  am = (-matrices[0] + 5 * matrices[1] + 2 * matrices[2]) / 6
  ap = (2 * matrices[1] + 5 * matrices[2] - matrices[3]) / 6
  central = (am @ qm + ap @ qp) / 2
  direction = velocity if abs(velocity) > 1e-14 else 0.0
  upwind = am @ qm if direction > 0 else ap @ qp if direction < 0 else central
  expected = alpha * upwind + (1 - alpha) * central
  sa_left = matrices[1, 4] @ perturbation[1]
  sa_right = matrices[2, 4] @ perturbation[2]
  expected[4] = (
    sa_left if direction > 0 else sa_right if direction < 0 else (sa_left + sa_right) / 2
  )
  actual = sum(b @ q for b, q in zip(convect._face_stencil(face), perturbation))
  np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-9)


def test_curvilinear_metrics_are_cofactors():
  from types import SimpleNamespace

  import convect

  for ds, dn in [([2.0, 0.7], [-0.4, 3.0]), ([2.0, 0.7], [0.4, -3.0])]:
    c = SimpleNamespace(jacobian=np.array([ds, dn]))
    det = ds[0] * dn[1] - ds[1] * dn[0]
    ms, mn = convect._metric(c, 'WE'), convect._metric(c, 'NS')
    np.testing.assert_allclose([ms @ ds, mn @ dn], [abs(det), abs(det)])
    np.testing.assert_allclose([ms @ dn, mn @ ds], 0, atol=1e-14)


def test_convection_sound_speed_matches_thermodynamic_closure():
  from linearization import mass_jacobian

  np.testing.assert_allclose(cc.cp - cc.cv, cc.R, rtol=1e-14)
  np.testing.assert_allclose(cc.cp / cc.cv, cc.gamma, rtol=1e-14)
  q = np.array([1.2, 40.0, -3.0, 300.0, 0.002])
  cell = cc.cell_class((1, 1), 0, 0, *q, 1, 1)
  flux = cell.F.copy()
  flux[4] = cell.sa_convect_vec()[0]
  sound_speed = np.sqrt(cc.gamma * cc.R * q[3])
  expected = np.sort([q[1] - sound_speed, q[1], q[1], q[1], q[1] + sound_speed])
  np.testing.assert_allclose(
    np.sort(np.linalg.eigvals(np.linalg.solve(mass_jacobian(q), flux))),
    expected,
    rtol=1e-12,
  )
