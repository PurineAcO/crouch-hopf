"""Analytic covariance and zero-speed regressions; no physical-flux differentiation."""

from types import SimpleNamespace

import classconfig as cc
import convect
import numpy as np
import pytest
from models import FlowModel

PARITY = np.diag([1, 1, -1, 1, 1])
REFLECT_XY = np.diag([1, -1])


def _face(kind, states, metrics):
  cells = []
  for i, (q, metric) in enumerate(zip(states, metrics)):
    c = cc.cell_class((i + 1, 2), 0, 0, *q, 1.0, 1.0)
    c.jacobian = metric
    cells.append(c)
  if kind == 'WE':
    cells[1].west = SimpleNamespace(west=cells[0])
    cells[2].east = SimpleNamespace(east=cells[3])
    return SimpleNamespace(direction=kind, west=cells[1], east=cells[2]), cells
  cells[1].south = SimpleNamespace(south=cells[0])
  cells[2].north = SimpleNamespace(north=cells[3])
  return SimpleNamespace(direction=kind, south=cells[1], north=cells[2]), cells


def _closed_action(kind, cells, perturbations, alpha, sign):
  """Independent reconstructed-matrix/reconstructed-field flux definition.

  Dual coordinate vectors are obtained from the inverse 2D metric, rather than
  convect.py's cofactor helper. All physical Jacobians here are analytic.
  """
  matrices = []
  for c in cells:
    metric = np.asarray(c.jacobian)
    normal = abs(np.linalg.det(metric)) * np.linalg.inv(metric)[:, 0 if kind == 'WE' else 1]
    A = normal[0] * c.F + normal[1] * c.G
    if cc.active_model() is FlowModel.SA:
      fx, fy = c.sa_convect_vec()
      A[4] = normal[0] * fx + normal[1] * fy
    matrices.append(A)
  A0, A1, A2, A3 = matrices
  q0, q1, q2, q3 = perturbations
  minus = ((-A0 + 5 * A1 + 2 * A2) / 6) @ ((-q0 + 5 * q1 + 2 * q2) / 6)
  plus = ((2 * A1 + 5 * A2 - A3) / 6) @ ((2 * q1 + 5 * q2 - q3) / 6)
  result = (1 - alpha) * (minus + plus) / 2 + alpha * (minus if sign > 0 else plus)
  result[4] = (A1 @ q1)[4] if sign > 0 else (A2 @ q2)[4]
  return result


@pytest.mark.parametrize('model', [FlowModel.LAMINAR, FlowModel.SA])
@pytest.mark.parametrize('kind', ['WE', 'NS'])
@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('alpha', [0.0, 0.2, 1.0])
def test_reflected_skew_upwind_flux_and_resolved_transport(
  monkeypatch, model, kind, direction, alpha
):
  monkeypatch.setattr(cc, 'flow_model', model)
  monkeypatch.setattr(cc, 'alpha_H', alpha)
  states = [
    np.array([1.2 + 0.1 * i, direction * (10 + 0.1 * i), 2 + 0.05 * i, 280 + 3 * i, 0.002])
    for i in range(4)
  ]
  metrics = [np.array([[2 + 0.1 * i, 0.7], [0.4, 3 + 0.2 * i]]) for i in range(4)]
  reflected_states = [PARITY @ q for q in states]
  reflected_metrics = [np.array([-REFLECT_XY @ m[0], REFLECT_XY @ m[1]]) for m in metrics]
  if kind == 'WE':
    reflected_states = reflected_states[::-1]
    reflected_metrics = reflected_metrics[::-1]
  f, cells = _face(kind, states, metrics)
  rf, _ = _face(kind, reflected_states, reflected_metrics)
  blocks = np.array(convect._face_stencil(f))
  reflected = np.array(convect._face_stencil(rf))
  expected = np.array([PARITY @ b @ PARITY for b in (blocks[::-1] if kind == 'WE' else blocks)])
  if kind == 'WE':
    expected = -expected
  np.testing.assert_allclose(reflected, expected, rtol=2e-14, atol=2e-10)
  # Signs follow directly from these prescribed metrics and velocities:
  # m_WE~(3,-.4), m_NS~(-.7,2); both choices are far above the tie bound.
  sign = direction if kind == 'WE' else -direction
  perturbations = np.random.default_rng(2007).normal(size=(4, 5))
  actual = sum(block @ dq for block, dq in zip(blocks, perturbations))
  np.testing.assert_allclose(
    actual, _closed_action(kind, cells, perturbations, alpha, sign), rtol=3e-14, atol=2e-9
  )


@pytest.mark.parametrize('model', [FlowModel.LAMINAR, FlowModel.SA])
@pytest.mark.parametrize('alpha', [0.0, 0.2, 1.0])
def test_noisy_fixed_axis_ties_without_changing_central_flow_rows(monkeypatch, model, alpha):
  monkeypatch.setattr(cc, 'flow_model', model)
  monkeypatch.setattr(cc, 'alpha_H', alpha)
  noise = 1e-13
  states = [np.array([1.2, 0.02, v + noise, 300.0, 0.002]) for v in [0.02, 0.01, -0.01, -0.02]]
  metrics = [np.array([[0.0, 1.0], [1.0, 0.0]]) for _ in range(4)]
  f, _ = _face('WE', states, metrics)
  # This bias is above the former LOCAL-velocity bound but below the acoustic
  # bound. It is not an exactly symmetric fixture that would miss the bug.
  assert noise > 64 * np.finfo(float).eps * max(np.hypot(q[1], q[2]) for q in states[1:3])
  corrected = np.array(convect._face_stencil(f))
  # For this fixture the old bound selected the raw nonzero sign. A zero bound
  # reproduces that decision without copying the production flux implementation.
  with monkeypatch.context() as local:
    local.setattr(np, 'finfo', lambda dtype: SimpleNamespace(eps=0.0))
    raw_sign = np.array(convect._face_stencil(f))

  def defect(blocks):
    return np.linalg.norm(
      blocks + np.array([PARITY @ b @ PARITY for b in blocks[::-1]])
    ) / np.linalg.norm(blocks)

  assert defect(corrected) < 1e-12
  if alpha > 0:
    assert defect(raw_sign) > 0.1 * alpha
  else:
    assert raw_sign[:, :4, :].tobytes() == corrected[:, :4, :].tobytes()
    if model is FlowModel.LAMINAR:
      assert raw_sign.tobytes() == corrected.tobytes()
  if model is FlowModel.SA:
    # SA's first-order row changes its zero-sign decision even for alpha=0.
    assert not np.array_equal(raw_sign[:, 4, :], corrected[:, 4, :])


@pytest.mark.parametrize('tangential_speed', [0.0, 1e-15, 1e-8, 0.02])
@pytest.mark.parametrize('alpha', [0.2, 1.0])
def test_exact_zero_normal_speed_is_central_near_stagnation(monkeypatch, tangential_speed, alpha):
  monkeypatch.setattr(cc, 'flow_model', FlowModel.LAMINAR)
  states = [np.array([1.0, tangential_speed, v, 300.0, 0.0]) for v in [0.02, 0.01, -0.01, -0.02]]
  metrics = [np.array([[0.0, 1.0], [1.0, 0.0]]) for _ in range(4)]
  f, _ = _face('WE', states, metrics)
  monkeypatch.setattr(cc, 'alpha_H', 0.0)
  central = np.array(convect._face_stencil(f))
  monkeypatch.setattr(cc, 'alpha_H', alpha)
  np.testing.assert_array_equal(convect._face_stencil(f), central)
