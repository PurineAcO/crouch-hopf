"""物面/远场虚单元的镜像规则：值映射与梯度映射必须与面处边界条件一致。"""

from types import SimpleNamespace

import classconfig as cc
import formmat
import grad
import numpy as np
import pytest


def test_wall_value_map_gives_paper_face_conditions():
  """面平均值给出 û=v̂=ν̃̂=0；ρ̂/T̂ 的两点面法向导数为零。"""
  source = np.array([0.7, 30.0, -12.0, 300.0, 1.0e-3])
  ghost = formmat._WALL_MAP @ source
  np.testing.assert_allclose((source + ghost) / 2, [0.7, 0.0, 0.0, 300.0, 0.0])
  difference = source - ghost
  np.testing.assert_allclose(difference[[0, 3]], 0)
  # u/v/ν̃ 的面法向导数保留：壁面摩擦与壁面热流正是靠它进入首层守恒律。
  assert np.all(np.abs(difference[[1, 2, 4]]) > 0)


@pytest.mark.parametrize('sign', [1.0, -1.0])
def test_mirror_ghost_places_cell_and_reflects_gradient(sign):
  normal = np.array([0.6, 0.8])
  tangent = np.array([-0.8, 0.6])
  midpoint = np.array([1.5, -0.25])
  distance = 0.4
  gradient = np.array([0.07, -0.13])
  source = SimpleNamespace(
    x=midpoint[0] + distance * normal[0], y=midpoint[1] + distance * normal[1]
  )
  for attribute in grad.GRADIENT_FIELDS:
    setattr(source, attribute, gradient.copy())
  face = SimpleNamespace(mid=tuple(midpoint), jacobian=np.array([2.5 * normal, tangent]))
  value_map = np.diag([1.0, sign, sign, 1.0, sign])
  cell = SimpleNamespace()
  grad.mirror_ghost(cell, source, face, value_map)
  np.testing.assert_allclose([cell.x, cell.y], midpoint - distance * normal, atol=1e-14)
  reflected = gradient - 2 * (gradient @ normal) * normal
  for attribute, component in zip(grad.GRADIENT_FIELDS, np.diag(value_map)):
    np.testing.assert_allclose(getattr(cell, attribute), component * reflected, atol=1e-14)


def test_solved_rings_follows_far_closure(monkeypatch):
  monkeypatch.setattr(cc, 'N_MAX', 5)
  monkeypatch.setattr(cc, 'far_riemann', False)
  # 物面环是守恒行；远场环在特征约束下不带时间导数。
  assert list(formmat.solved_rings()) == [1, 2, 3, 4]
  monkeypatch.setattr(cc, 'far_riemann', True)
  assert list(formmat.solved_rings()) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize('far_riemann', [False, True])
def test_build_leaves_only_constraint_rings_without_time_derivative(monkeypatch, far_riemann):
  monkeypatch.setattr(cc, 'N_MAX', 4)
  monkeypatch.setattr(cc, 'S_MAX', 3)
  monkeypatch.setattr(cc, 'far_riemann', far_riemann)
  formmat._rows.clear()
  formmat._cols.clear()
  formmat._vals.clear()
  _, T = formmat.build()
  diagonal = np.asarray(T.diagonal()).reshape(cc.N_MAX, -1)
  expected = np.ones(cc.N_MAX)
  if not far_riemann:
    expected[-1] = 0.0
  np.testing.assert_array_equal(diagonal, np.repeat(expected[:, None], cc.S_MAX * 5, axis=1))
