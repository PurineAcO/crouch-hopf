from types import SimpleNamespace

import boundary as bc
import classconfig as cc
import numpy as np
import pytest

BASE = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, 2], [1, 1], [-1, 1.0]])


@pytest.mark.parametrize('angle', [0, 0.71, 2.3])
@pytest.mark.parametrize('stretch', [1, 1e3, 1e6])
@pytest.mark.parametrize('side', [-1, 1])
def test_quadratic_physical_normal(angle, stretch, side):
  rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
  mapping = np.array([[stretch, 0], [0.7 * stretch, side]]) @ rotation.T
  points = BASE @ mapping
  normal = rotation[:, 1]
  weights = bc.normal_derivative(points, normal)
  # Polynomials expressed in logical coordinates still span all physical quadratics.
  # Their physical derivatives use the independently computed affine chain rule.
  direction = np.linalg.solve(mapping.T, normal)
  x, y = BASE.T
  values = np.array([np.ones(7), x, y, x * x, x * y, y * y])
  np.testing.assert_allclose(values @ weights, [0, *direction, 0, 0, 0], atol=2e-8)


def test_tangential_linear_counterexample():
  points = BASE @ np.array([[1, 0], [0.8, 0.3]])
  weights = bc.normal_derivative(points, [0, 1])
  assert abs(weights @ points[:, 0]) < 1e-14
  assert abs(weights @ points[:, 1] - 1) < 1e-14
  assert abs(weights.sum()) < 1e-14
  # Projecting just the three radial cells falsely differentiates x as 0.8/0.3.
  assert abs((points[3, 0] - points[0, 0]) / points[3, 1]) > 2


@pytest.mark.parametrize(
  'points', [np.zeros((7, 2)), BASE[:3], np.array([[i, i * i] for i in range(7)]), BASE * np.nan]
)
def test_invalid_or_rank_deficient_stencil(points):
  with pytest.raises(ValueError):
    bc.normal_derivative(points, [0, 1])


def make_stencil(monkeypatch, far=False, mach=0.3):
  monkeypatch.setattr(cc, 'N_MAX', 4)
  normal = np.array([0.6, 0.8])
  tangent = np.array([0.8, -0.6])
  points = BASE @ np.array([2 * tangent, (-1 if far else 1) * normal + 0.7 * tangent])
  cells = []
  for i, (x, y) in enumerate(points):
    cells.append(
      SimpleNamespace(
        x=x, y=y, rho=1.1 + i * 0.05, T=290 + 3 * i, u=0.0, v=0.0, index=(1, 4 if far else 1)
      )
    )
  c, e, w, first, second, fe, fw = cells
  a = np.sqrt(cc.gamma * cc.R * c.T)
  for cell in cells:
    cell.u, cell.v = mach * a * normal
  side = 'south' if far else 'north'
  for owner, direction, neighbor in [
    (c, 'east', e),
    (c, 'west', w),
    (c, side, first),
    (first, side, second),
    (first, 'east', fe),
    (first, 'west', fw),
  ]:
    setattr(owner, direction, SimpleNamespace(**{direction: neighbor}))
  # 边界面：物面在环单元内侧半个单元高度处，远场面在外侧半个单元高度处；
  # 面状态按 face_class.form_physics 由两侧平均得到（物面镜像后 u=v=0）。
  ghost = SimpleNamespace(rho=c.rho, u=-c.u if not far else c.u, v=-c.v if not far else c.v, T=c.T)
  face = SimpleNamespace(
    mid=np.array([c.x, c.y]) + (0.5 if far else -0.5) * normal,
    jacobian=np.array([normal, tangent]),
    rho=(c.rho + ghost.rho) / 2,
    u=(c.u + ghost.u) / 2,
    v=(c.v + ghost.v) / 2,
    T=(c.T + ghost.T) / 2,
    miubl=0.0,
  )
  setattr(c, 'north' if far else 'south', face)
  c.influence = np.zeros((13, 5, 5))
  c.form_influence = lambda slot, block: c.influence[slot].__iadd__(block)
  return c, cells, normal


def make_uniform_stencil(monkeypatch, mach):
  """远场模板，但基流在整条模板上均匀，便于用精确线性扰动验证条件位置。"""
  c, cells, normal = make_stencil(monkeypatch, far=True, mach=mach)
  a = np.sqrt(cc.gamma * cc.R * 300.0)
  for cell in cells:
    cell.rho, cell.T = 0.7, 300.0
    cell.u, cell.v = mach * a * normal
  c.north.rho, c.north.T = 0.7, 300.0
  c.north.u, c.north.v = c.u, c.v
  return c, cells, normal


@pytest.mark.parametrize('rho,T', [(1.1, 290), (0.02, 130), (3.7, 900)])
def test_paper_invariant_closed_derivatives(monkeypatch, rho, T):
  c, _, normal = make_stencil(monkeypatch)
  c.rho, c.T = rho, T
  rows, speeds = bc.characteristics(c, 7 * normal)
  # Closed directional derivatives of the original nonlinear invariants:
  # I+/- = n.u +/- 2 sqrt(gamma R T)/(gamma-1), It = t.u,
  # E = R T rho**(1-gamma), and I_SA = nu. No numerical differentiation.
  for drho, du, dv, dT, dnu in np.eye(5):
    dun = normal @ [du, dv]
    da = 0.5 * np.sqrt(cc.gamma * cc.R / T) * dT
    dE = cc.R * (rho ** (1 - cc.gamma) * dT + T * (1 - cc.gamma) * rho ** (-cc.gamma) * drho)
    expected = [
      dun + 2 * da / (cc.gamma - 1),
      dun - 2 * da / (cc.gamma - 1),
      -normal[1] * du + normal[0] * dv,
      dE,
      dnu,
    ]
    np.testing.assert_allclose(rows @ [drho, du, dv, dT, dnu], expected, atol=1e-12)
  a = np.sqrt(cc.gamma * cc.R * T)
  un = normal @ [c.u, c.v]
  np.testing.assert_allclose(speeds, [un + a, un - a, un, un, un])


def test_paper_acoustic_relation_to_euler_characteristics(monkeypatch):
  c, _, normal = make_stencil(monkeypatch)
  rows, _ = bc.characteristics(c, normal)
  a = np.sqrt(cc.gamma * cc.R * c.T)
  beta = c.rho ** (cc.gamma - 1) / ((cc.gamma - 1) * a)
  dp_over_rhoa = np.array([cc.R * c.T / c.rho, 0, 0, cc.R, 0]) / a
  dun = np.r_[0, normal, 0, 0]
  acoustic = np.array([dun + dp_over_rhoa, dun - dp_over_rhoa])
  np.testing.assert_allclose(
    rows[:2] - acoustic, np.array([beta, -beta])[:, None] * rows[3], atol=1e-12
  )
  # An isentropic perturbation gives identical acoustic amplitudes.
  drho = 0.2
  dq = np.array([drho, 0.7, -0.4, (cc.gamma - 1) * c.T / c.rho * drho, 0.1])
  np.testing.assert_allclose(rows[3] @ dq, 0, atol=1e-11)
  np.testing.assert_allclose(rows[:2] @ dq, acoustic @ dq, atol=1e-12)
  # A density-only perturbation has nonzero entropy and distinguishes the BCs.
  assert abs(rows[3, 0]) > 0
  np.testing.assert_array_equal(rows[:2, 0], 0)
  assert np.all(np.abs(acoustic[:, 0]) > 0)


def test_outflow_prescribes_paper_minus_with_nonzero_entropy(monkeypatch):
  c, _, normal = make_stencil(monkeypatch, far=True, mach=0.3)
  bc.far_boundary(c)
  # With du=dT=0 and drho!=0, paper I- is zero but Euler C- is not.
  dq = np.array([1.0, 0, 0, 0, 0])
  assert c.influence[cc.dic['c'], 1] @ dq == 0
  a = np.sqrt(cc.gamma * cc.R * c.T)
  euler_minus = np.r_[-cc.R * c.T / (c.rho * a), normal, -cc.R / a, 0]
  assert abs(euler_minus @ dq) > 1
  # 入流族是“面处不变量为零”，不是“把扰动设为零”：常值 T̂ 必须给出非零残差。
  assert abs(c.influence[cc.dic['c'], 1] @ np.array([0, 0, 0, 1.0, 0])) > 0


@pytest.mark.parametrize(
  'mach,incoming',
  [(-2, [1, 1, 1, 1, 1]), (-0.3, [0, 1, 1, 1, 1]), (0.3, [0, 1, 0, 0, 0]), (2, [0, 0, 0, 0, 0])],
)
def test_farfield_modes_and_neighbor_states(monkeypatch, mach, incoming):
  c, cells, normal = make_stencil(monkeypatch, far=True, mach=mach)
  bc.far_boundary(c)
  names = ['c', 'e', 'w', 's', 'ss', 'se', 'sw']
  points = np.array([[q.x, q.y] for q in cells])
  mid = np.asarray(c.north.mid)
  value = bc.face_value(points, mid, normal, preferred_stencil=[0, 3, 4])
  gradient = bc.normal_derivative(points, normal, preferred_stencil=[0, 3, 4], origin=mid)
  face_rows, _ = bc.characteristics(c.north, normal)
  incoming = np.array(incoming, dtype=bool)
  for i, (name, cell, weight_value, weight_gradient) in enumerate(
    zip(names, cells, value, gradient)
  ):
    rows, _ = bc.characteristics(cell, normal)
    block = c.influence[cc.dic[name]]
    # 出流族：面处法向导数为零，用面法向导数算子乘各单元自身的系数。
    np.testing.assert_allclose(block[~incoming], weight_gradient * rows[~incoming])
    # 入流族：面处不变量为零，用面值算子乘面自身的系数。
    np.testing.assert_allclose(block[incoming], weight_value * face_rows[incoming])
  assert abs(value.sum() - 1) < 1e-11
  assert abs(gradient.sum()) < 1e-11


@pytest.mark.parametrize('mach', [-2, -0.3, 0.3, 2])
@pytest.mark.parametrize('perturbation', [np.array([0.5, -0.3, 0.2, 0.1, 0.05])])
def test_farfield_conditions_are_evaluated_on_the_face(monkeypatch, mach, perturbation):
  """均勺基流下用精确线性扰动验证：出流族=面处法向导数，入流族=面处不变量值。"""
  c, cells, normal = make_uniform_stencil(monkeypatch, mach)
  bc.far_boundary(c)
  names = ['c', 'e', 'w', 's', 'ss', 'se', 'sw']
  points = np.array([[q.x, q.y] for q in cells])
  mid = np.asarray(c.north.mid)
  rows, speeds = bc.characteristics(c.north, normal)
  incoming = speeds <= 0
  slope = np.array([0.03, -0.02, 0.01, 0.04, 0.02])
  fields = perturbation + ((points - mid) @ normal)[:, None] * slope
  action = np.array(
    [
      sum(c.influence[cc.dic[name]][f] @ fields[i] for i, name in enumerate(names))
      for f in range(5)
    ]
  )
  expected = np.where(incoming, rows @ perturbation, rows @ slope)
  np.testing.assert_allclose(action, expected, atol=1e-11)
  # 扰动没有被整族置零：出流族的残差就是面处真实斜率，入流族的是面处不变量值。
  assert np.any(np.abs(action) > 1e-6)


@pytest.mark.parametrize('angle', [0.1, 1.9])
def test_translated_physical_quadratic(angle):
  rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
  points = BASE @ np.array([[3, 0], [2, 0.2]]) @ rotation.T + [7, -2]
  normal = np.array([0.3, -0.8])
  weights = bc.normal_derivative(points, normal)
  x, y = points.T
  f = 2 + 3 * x - 4 * y + 0.5 * x * x + 2 * x * y - 3 * y * y
  gradient = [3 + x[0] + 2 * y[0], -4 + 2 * x[0] - 6 * y[0]]
  expected = np.dot(normal, gradient) / np.linalg.norm(normal)
  np.testing.assert_allclose(weights @ f, expected, atol=1e-11)


def test_invalid_boundary_coefficients_rejected(monkeypatch):
  c, cells, _ = make_stencil(monkeypatch, far=True)
  cells[3].rho = np.nan
  with pytest.raises(ValueError, match='base state'):
    bc.far_boundary(c)
  assert not c.influence.any()


@pytest.mark.parametrize('mach', [-0.3, 0.3])
@pytest.mark.parametrize('rho,T', [(1.1, 290.0)])
def test_extrapolated_paper_plus_with_spatial_entropy(monkeypatch, mach, rho, T):
  """纯密度线性剖面：声学行（论文 I+/- 不含密度）无响应，熵行给出面处真实斜率。"""
  c, cells, normal = make_uniform_stencil(monkeypatch, mach)
  for cell in cells:
    cell.rho, cell.T = rho, T
  c.north.rho, c.north.T = rho, T
  bc.far_boundary(c)
  names = ['c', 'e', 'w', 's', 'ss', 'se', 'sw']
  points = np.array([[q.x, q.y] for q in cells])
  mid = np.asarray(c.north.mid)
  rows, speeds = bc.characteristics(c.north, normal)
  slope = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
  fields = ((points - mid) @ normal)[:, None] * slope
  action = np.array(
    [
      sum(c.influence[cc.dic[name]][f] @ fields[i] for i, name in enumerate(names))
      for f in range(5)
    ]
  )
  outgoing = ~(speeds <= 0)
  np.testing.assert_allclose(action[outgoing], rows[outgoing] @ slope, atol=1e-11)
  # 两条声学行的密度列为零；熵行不为零。
  np.testing.assert_allclose(rows[:2, 0], 0)
  assert abs(rows[3, 0]) > 0
  assert abs(action[3]) > 0


@pytest.mark.parametrize('angle', [0.0, 0.7, 2.1])
@pytest.mark.parametrize('skew', [0.0, 0.8, 3.0])
def test_preferred_stencil_preserves_normal_limit_and_2d_quadratics(angle, skew):
  rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
  points = (
    np.array([[0, 0], [2, 0], [-2, 0], [skew, 1], [2.4 * skew, 2.4], [2 + skew, 1], [-2 + skew, 1]])
    @ rotation.T
  )
  normal = rotation @ [0, 1]
  weights = bc.normal_derivative(points, normal, preferred_stencil=[0, 3, 4])
  x, y = points.T
  for field, exact in [
    (np.ones(7), 0),
    (x, normal[0]),
    (y, normal[1]),
    (x * x, 0),
    (x * y, 0),
    (y * y, 0),
  ]:
    np.testing.assert_allclose(weights @ field, exact, atol=1e-12)
  if skew == 0:
    radial = np.array([[1, 1, 1], [0, 1, 2.4], [0, 1, 2.4**2]])
    expected = np.zeros(7)
    expected[[0, 3, 4]] = np.linalg.solve(radial, [0, 1, 0])
    np.testing.assert_allclose(weights, expected, atol=2e-14)


def test_collapsed_radial_projection_still_uses_full_2d_constraints():
  points = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, 2], [1, 1], [-1, 1]], dtype=float)
  weights = bc.normal_derivative(points, [1, 0], preferred_stencil=[0, 3, 4])
  np.testing.assert_allclose(weights @ points, [1, 0], atol=1e-14)
