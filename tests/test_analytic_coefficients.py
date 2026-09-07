"""独立符号求导核对解析系数；不通过扰动物理通量进行数值求导。"""

from functools import cache

import classconfig as cc
import numpy as np
import pytest
import sympy as sp
from linearization import source_coefficients, viscosity, viscous_coefficients, wall_response
from models import FlowModel


@cache
def symbolic_coefficients(kind, sa=True, capped=False):
  rho, u, v, temperature, nu = sp.symbols('rho u v T nu', positive=True)
  q = sp.Matrix([rho, u, v, temperature, nu])
  g = sp.Matrix(5, 2, sp.symbols('g0:10', real=True))
  mu, distance = sp.symbols('mu d', positive=True)
  nx, ny = sp.symbols('nx ny', real=True)
  chi = rho * nu / mu
  fv1 = chi**3 / (chi**3 + cc.Cv1**3)
  mut = rho * nu * fv1 if sa else 0
  if kind == 'viscous':
    velocity_gradient = g[1:3, :]
    strain = (
      velocity_gradient
      + velocity_gradient.T
      - sp.Rational(2, 3) * sp.trace(velocity_gradient) * sp.eye(2)
    )
    traction = (mu + mut) * strain * sp.Matrix([nx, ny])
    heat = cc.cp * (mu / cc.Pr + mut / cc.Prt) * (g[3, 0] * nx + g[3, 1] * ny)
    diffusion = cc.inv_sigma * (mu + rho * nu) * (g[4, 0] * nx + g[4, 1] * ny) if sa else 0
    expression = sp.Matrix(
      [0, traction[0], traction[1], u * traction[0] + v * traction[1] + heat, diffusion]
    )
  else:
    fv2 = 1 - chi / (1 + chi * fv1)
    ft2 = cc.Ct3 * sp.exp(-cc.Ct4 * chi**2)
    modified = sp.Abs(g[1, 1] - g[2, 0]) + fv2 * nu / (cc.kappa * distance) ** 2
    r = cc.rmax if capped else nu / (modified * (cc.kappa * distance) ** 2)
    wall_g = r + cc.Cw2 * (r**6 - r)
    fw = wall_g * ((1 + cc.Cw3**6) / (wall_g**6 + cc.Cw3**6)) ** sp.Rational(1, 6)
    strain = (g[1:3, :] + g[1:3, :].T) / 2
    strain_squared = 2 * sum(x**2 for x in strain)
    expression = sp.Matrix(
      [
        rho * cc.Cb1 * (1 - ft2) * modified * nu
        - rho * (cc.Cw1 * fw - cc.Cb1 * ft2 / cc.kappa**2) * (nu / distance) ** 2
        + cc.Cb2 * cc.inv_sigma * rho * (g[4, 0] ** 2 + g[4, 1] ** 2)
        - cc.C5 * rho * nu**2 * strain_squared / (cc.gamma * cc.R * temperature)
      ]
    )
  # μ 是独立的已知基流参数，故这些符号导数自动对应 μ′=0。
  derivatives = expression.jacobian(list(q) + list(g))
  return sp.lambdify((list(q), list(g), mu, distance, nx, ny), derivatives, 'numpy', cse=True)


@pytest.mark.parametrize('r', [-1e200, -1e8, -12.0, -2.01, -2.0, -0.1, 0.0, 2.01, 9.0, 10.0])
def test_stable_wall_response_against_high_precision_symbolic_derivative(r):
  x = sp.Symbol('r', real=True)
  g = x + sp.Rational(str(cc.Cw2)) * (x**6 - x)
  c6 = sp.Rational(str(cc.Cw3)) ** 6
  z = sp.Symbol('wall_g', real=True)
  scalar = z * ((1 + c6) / (z**6 + c6)) ** sp.Rational(1, 6)
  f = scalar.subs(z, g)
  derivative = (
    sp.factor(sp.diff(scalar, z)).subs(z, g) * sp.diff(g, x) if r < cc.rmax else sp.Integer(0)
  )
  expected = [
    float(z.subs(x, sp.Float(r, 80)).evalf(60)) for z in (f, x * derivative, x**2 * derivative)
  ]
  np.testing.assert_allclose(wall_response(r), expected, rtol=2e-13, atol=1e-300)


@pytest.mark.parametrize('sa', [False, True])
def test_viscous_coefficients_against_symbolic_tensor_derivative(sa, monkeypatch):
  monkeypatch.setattr(cc, 'flow_model', FlowModel.SA if sa else FlowModel.LAMINAR)
  reference = symbolic_coefficients('viscous', sa)
  rng = np.random.default_rng(407)
  for _ in range(20):
    rho = 10 ** rng.uniform(-5, 0.3)
    temperature = rng.uniform(230, 450)
    q = np.array([rho, rng.uniform(-70, 70), rng.uniform(-30, 30), temperature, 0.0])
    mu = viscosity(q)[0]
    q[4] = 10 ** rng.uniform(-1, 2) * mu / rho if sa else 0
    g = rng.normal(size=(5, 2)) * 5
    normal = rng.normal(size=2)
    a, b = viscous_coefficients(q, g, normal)
    expected = np.asarray(reference(q, g.ravel(), mu, 1.0, *normal))
    np.testing.assert_allclose(
      np.column_stack([a, b.reshape(5, 10)]), expected, rtol=3e-12, atol=1e-10
    )


def test_source_coefficients_against_symbolic_paper_equation():
  reference = symbolic_coefficients('source')
  rng = np.random.default_rng(2007)
  tested = 0
  for _ in range(400):
    rho = 10 ** rng.uniform(-4, 0.2)
    q = np.array([rho, 40.0, 3.0, rng.uniform(230, 450), 0.0])
    mu = viscosity(q)[0]
    q[4] = 10 ** rng.uniform(-1, 2) * mu / rho
    g = rng.normal(size=(5, 2)) * 30
    g[4] *= q[4]
    distance = 10 ** rng.uniform(-2, 0)
    chi = rho * q[4] / mu
    fv1 = chi**3 / (chi**3 + cc.Cv1**3)
    fv2 = 1 - chi / (1 + chi * fv1)
    modified = abs(g[1, 1] - g[2, 0]) + fv2 * q[4] / (cc.kappa * distance) ** 2
    r = q[4] / (modified * (cc.kappa * distance) ** 2)
    if not 1e-5 < r < 8:
      continue
    a, b = source_coefficients(q, g, distance)
    expected = np.asarray(reference(q, g.ravel(), mu, distance, 0.0, 0.0)).ravel()
    np.testing.assert_allclose(np.r_[a, b.ravel()], expected, rtol=3e-10, atol=1e-10)
    tested += 1
    if tested == 50:
      break
  assert tested == 50


def test_r_cap_has_zero_wall_function_derivative():
  q = np.array([1.2, 40.0, 3.0, 300.0, 0.0])
  mu = viscosity(q)[0]
  q[4] = 3 * mu / q[0]
  chi = 3.0
  fv1 = chi**3 / (chi**3 + cc.Cv1**3)
  fv2 = 1 - chi / (1 + chi * fv1)
  distance = 0.01
  omega = (1 / 20 - fv2) * q[4] / (cc.kappa * distance) ** 2
  g = np.zeros((5, 2))
  g[1, 1] = omega
  a, b = source_coefficients(q, g, distance)
  expected = symbolic_coefficients('source', capped=True)(q, g.ravel(), mu, distance, 0.0, 0.0)
  np.testing.assert_allclose(np.r_[a, b.ravel()], np.ravel(expected), rtol=1e-11, atol=1e-11)


def test_zero_modified_vorticity_and_adjacent_states_remain_finite():
  q = np.array([1.2, 40.0, 3.0, 300.0, 0.0])
  mu = viscosity(q)[0]
  q[4] = 3 * mu / q[0]
  chi = q[0] * q[4] / mu
  fv1 = chi**3 / (chi**3 + cc.Cv1**3)
  fv2 = 1 - chi / (1 + chi * fv1)
  distance = 0.01
  correction = fv2 * q[4] * (1 / distance**2 / cc.kappa**2)
  g = np.zeros((5, 2))
  for omega in [-correction, np.nextafter(-correction, 0), np.nextafter(-correction, np.inf)]:
    g[1, 1] = omega
    a, b = source_coefficients(q, g, distance)
    expected = symbolic_coefficients('source', capped=True)(q, g.ravel(), mu, distance, 0.0, 0.0)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(np.r_[a, b.ravel()], np.ravel(expected), rtol=1e-11, atol=1e-11)


def test_c5_temperature_and_strain_response(monkeypatch):
  q = np.array([1.2, 40.0, 3.0, 300.0, 0.002])
  g = np.array([[0.4, -0.7], [3.0, 4.0], [-2.0, 5.0], [0.1, 0.2], [0.03, 0.04]])
  a, b = source_coefficients(q, g, 0.1)
  monkeypatch.setattr(cc, 'C5', 0.0)
  plain_a, plain_b = source_coefficients(q, g, 0.1)
  rho, _, _, temperature, nu = q
  strain_squared = 2 * 3**2 + 2 * 5**2 + (4 - 2) ** 2
  factor = 3.5 * rho * nu**2 / (cc.gamma * cc.R * temperature)
  np.testing.assert_allclose(
    a - plain_a,
    factor * strain_squared * np.array([-1 / rho, 0.0, 0.0, 1 / temperature, -2 / nu]),
    rtol=1e-10,
    atol=1e-13,
  )
  expected = np.zeros((5, 2))
  expected[1] = -2 * factor * np.array([6.0, 2.0])
  expected[2] = -2 * factor * np.array([2.0, 10.0])
  np.testing.assert_allclose(b - plain_b, expected, rtol=1e-11, atol=1e-13)


def test_source_does_not_add_density_gradient_terms():
  q = np.array([1.2, 40.0, 3.0, 300.0, 0.002])
  g = np.ones((5, 2))
  g[1, 1] = 20.0
  a, b = source_coefficients(q, g, 0.1)
  g[0] = [300.0, -700.0]
  other_a, other_b = source_coefficients(q, g, 0.1)
  np.testing.assert_array_equal(a, other_a)
  np.testing.assert_array_equal(b, other_b)
  assert not np.any(b[0])


def test_zero_nu_zero_vorticity_uses_continuous_source_limit():
  a, b = source_coefficients(np.array([1.2, 40.0, 3.0, 300.0, 0.0]), np.zeros((5, 2)), 0.1)
  np.testing.assert_array_equal(a, 0.0)
  np.testing.assert_array_equal(b, 0.0)


def test_modified_vorticity_is_not_floored():
  q = np.array([1.2, 40.0, 3.0, 300.0, 0.0])
  mu = viscosity(q)[0]
  q[4] = 3 * mu / q[0]
  g = np.zeros((5, 2))
  g[1, 1] = 0.01
  distance = 0.01
  # χ=3 时 fv2<0，这一状态的 S̃<0。与未设下限的方程符号导数比较。
  a, b = source_coefficients(q, g, distance)
  expected = symbolic_coefficients('source')(q, g.ravel(), mu, distance, 0.0, 0.0)
  np.testing.assert_allclose(np.r_[a, b.ravel()], np.ravel(expected), rtol=3e-11, atol=1e-12)
