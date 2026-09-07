"""论文 (2.2.1)–(2.2.7) 的解析系数；未知量顺序为 ρ、u、v、T、ν̃。"""

import classconfig as cc
import numpy as np


def mass_jacobian(q):
  rho, u, v, temperature, nu = q
  energy = cc.cv * temperature + 0.5 * (u * u + v * v)
  return np.array(
    [
      [1, 0, 0, 0, 0],
      [u, rho, 0, 0, 0],
      [v, 0, rho, 0, 0],
      [energy, rho * u, rho * v, rho * cc.cv, 0],
      [nu, 0, 0, 0, rho],
    ]
  )


def state(cell):
  return np.array([cell.rho, cell.u, cell.v, cell.T, cell.miubl])


def gradients(cell):
  return np.array([cell.rhograd, cell.ugrad, cell.vgrad, cell.Tgrad, cell.miublgrad])


def viscosity(q):
  """基流采用 Sutherland 黏度；下列扰动系数统一取 μ′=0。"""
  rho, _, _, temperature, nu = q
  mu = cc.mu0 * (temperature / cc.T0) ** 1.5 * (cc.T0 + cc.Ts) / (temperature + cc.Ts)
  if cc.active_model().nvar == 4:
    return mu, 0.0
  chi = rho * nu / mu
  fv1 = chi**3 / (chi**3 + cc.Cv1**3)
  return mu, rho * nu * fv1


def viscous_coefficients(q, g, normal):
  """返回黏性面通量的一阶系数 Aq′+B∇q′，法向包含面长。

  动量：τ′n = μeff δD·n + μt′ D·n。
  能量：u′·τn + u·τ′n + k ∂nT′ + k′ ∂nT。
  SA 扩散：(μ+ρν̃) ∂nν̃′/σ + (ν̃ρ′+ρν̃′) ∂nν̃/σ。
  """
  rho, u, v, _, nu = q
  nx, ny = normal
  mu, mut = viscosity(q)
  eff = mu + mut
  a = np.zeros((5, 5))
  b = np.zeros((5, 5, 2))
  dmut = np.zeros(5)
  sa = cc.active_model().nvar == 5
  if sa:
    chi = rho * nu / mu
    fv1 = chi**3 / (chi**3 + cc.Cv1**3)
    # fv1 + χ dfv1/dχ = fv1(4−3fv1)，对应论文式 (2.2.7)。
    response = fv1 * (4 - 3 * fv1)
    dmut[[0, 4]] = response * np.array([nu, rho])
  ux, uy = g[1]
  vx, vy = g[2]
  dxx = (4 * ux - 2 * vy) / 3
  dyy = (4 * vy - 2 * ux) / 3
  dxy = uy + vx
  strain_normal = np.array([dxx * nx + dxy * ny, dxy * nx + dyy * ny])
  a[1] = strain_normal[0] * dmut
  a[2] = strain_normal[1] * dmut
  a[3] = (u * strain_normal[0] + v * strain_normal[1] + cc.cp / cc.Prt * (g[3] @ normal)) * dmut
  a[3, 1:3] += eff * strain_normal
  b[1, 1] = eff * np.array([4 / 3 * nx, ny])
  b[1, 2] = eff * np.array([ny, -2 / 3 * nx])
  b[2, 1] = eff * np.array([-2 / 3 * ny, nx])
  b[2, 2] = eff * np.array([nx, 4 / 3 * ny])
  b[3] = u * b[1] + v * b[2]
  b[3, 3] = cc.cp * (mu / cc.Pr + mut / cc.Prt) * np.asarray(normal)
  if sa:
    a[4, [0, 4]] = cc.inv_sigma * (g[4] @ normal) * np.array([nu, rho])
    b[4, 4] = cc.inv_sigma * (mu + rho * nu) * np.asarray(normal)
  return a, b


def wall_response(raw_r):
  """返回 fw、r dfw/dr、r² dfw/dr，倒数形式避免负大 r 的高次幂溢出。"""
  r = min(raw_r, cc.rmax)
  c6 = cc.Cw3**6
  if abs(r) > 2:
    t = 1 / r
    denominator = cc.Cw2 + (1 - cc.Cw2) * t**5
    inverse_g = t**6 / denominator
    fw = np.copysign(((1 + c6) / (1 + c6 * inverse_g**6)) ** (1 / 6), inverse_g)
    scale = fw * c6 / (1 + c6 * inverse_g**6) / denominator**7
    r_fw = scale * ((1 - cc.Cw2) * t**41 + 6 * cc.Cw2 * t**36)
    r2_fw = scale * ((1 - cc.Cw2) * t**40 + 6 * cc.Cw2 * t**35)
  else:
    wall_g = r + cc.Cw2 * (r**6 - r)
    factor = ((1 + c6) / (wall_g**6 + c6)) ** (1 / 6)
    fw = wall_g * factor
    derivative = factor * c6 / (wall_g**6 + c6) * (1 - cc.Cw2 + 6 * cc.Cw2 * r**5)
    r_fw, r2_fw = r * derivative, r**2 * derivative
  if raw_r >= cc.rmax:
    r_fw = r2_fw = 0.0
  return fw, r_fw, r2_fw


def source_coefficients(q, g, distance):
  """SA 体积源项的一阶行向量 Rq′+G∇q′，装配时只写第五行。

  对论文 (2.1.5) 的产生、破坏、梯度平方及 C5 应变率项逐项求导。
  不对 Ω 或 S̃ 加下限；r 保留原版上限，饱和段取 dr=0。
  """
  rho, _, _, temperature, nu = q
  if not np.isfinite(distance) or distance <= 0:
    raise ValueError('SA source requires a positive wall distance')
  mu, _ = viscosity(q)
  chi = rho * nu / mu
  fv1 = chi**3 / (chi**3 + cc.Cv1**3)
  fv2 = 1 - chi / (1 + chi * fv1)
  fv2_chi = (3 * chi * fv1 * (1 - fv1) - 1) / (1 + chi * fv1) ** 2
  ft2 = cc.Ct3 * np.exp(-cc.Ct4 * chi**2)
  ft2_chi = -2 * cc.Ct4 * chi * ft2
  inv_d2 = 1 / distance**2
  wall_scale = inv_d2 / cc.kappa**2
  ux, uy = g[1]
  vx, vy = g[2]
  curl = uy - vx
  omega = abs(curl)
  modified = omega + fv2 * nu * wall_scale
  # ν̃=0 时破坏项及其一阶导数为零，可直接使用连续极限。
  if nu == 0:
    raw_r = 0.0
  elif modified == 0:
    # 与基流端一致：孤立奇点采用正侧饱和值，不改变产生项中的 S̃。
    raw_r = cc.rmax
  else:
    raw_r = nu * wall_scale / modified
  fw, r_fw, r2_fw = wall_response(raw_r)
  # ρ、ν̃ 两个方向的链式导数；μ 和壁面距离固定。
  chi_q = np.array([nu / mu, rho / mu])
  ft2_q = ft2_chi * chi_q
  modified_q = wall_scale * (nu * fv2_chi * chi_q + np.array([0.0, fv2]))
  if nu == 0:
    fw_q = np.zeros(2)
  else:
    fw_q = np.array([0.0, r_fw / nu]) - r2_fw / (nu * wall_scale) * modified_q
  wall_term = cc.Cw1 * fw - cc.Cb1 / cc.kappa**2 * ft2
  wall_term_q = cc.Cw1 * fw_q - cc.Cb1 / cc.kappa**2 * ft2_q
  production_q = cc.Cb1 * (
    (1 - ft2) * modified * np.array([nu, rho])
    + rho * nu * ((1 - ft2) * modified_q - modified * ft2_q)
  )
  destruction_q = inv_d2 * (wall_term * np.array([nu**2, 2 * rho * nu]) + rho * nu**2 * wall_term_q)
  row = np.zeros(5)
  row[[0, 4]] = production_q - destruction_q
  row[0] += cc.Cb2 * cc.inv_sigma * (g[4] @ g[4])
  gradient = np.zeros((5, 2))
  # d|uy−vx|：在零涡量处采用对称线性化系数 0，不修改 Ω 的值。
  rotation = rho * nu * (cc.Cb1 * (1 - ft2) + cc.Cw1 * cc.kappa**2 * r2_fw) * np.sign(curl)
  gradient[1, 1] = rotation
  gradient[2, 0] = -rotation
  gradient[4] = 2 * cc.Cb2 * cc.inv_sigma * rho * g[4]
  # C5 使用 2SijSij；论文 (2.2.5) 的最后一个 v 梯度应为 vy。
  strain_squared = 2 * ux**2 + 2 * vy**2 + (uy + vx) ** 2
  compressibility = cc.C5 / (cc.gamma * cc.R * temperature)
  row[0] -= compressibility * nu**2 * strain_squared
  row[4] -= 2 * compressibility * rho * nu * strain_squared
  row[3] = compressibility * rho * nu**2 * strain_squared / temperature
  gradient[1] -= 2 * compressibility * rho * nu**2 * np.array([2 * ux, uy + vx])
  gradient[2] -= 2 * compressibility * rho * nu**2 * np.array([uy + vx, 2 * vy])
  return row, gradient
