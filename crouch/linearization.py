"""Local complex-step Jacobians of the dimensional fluxes and standard SA source."""

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


def viscosity(q, molecular_mu=None):
  temperature = q[3]
  mu = molecular_mu
  if mu is None:
    mu = cc.mu0 * (temperature / cc.T0) ** 1.5 * (cc.T0 + cc.Ts) / (temperature + cc.Ts)
  if cc.flow_model.nvar == 4:
    return mu, 0.0
  rho, nu = q[0], q[4]
  chi = rho * nu / mu
  fv1 = chi**3 / (chi**3 + cc.Cv1**3)
  return mu, rho * nu * fv1


def viscous_flux(q, g, normal, molecular_mu=None):
  rho, u, v, _, nu = q
  mu, mut = viscosity(q, molecular_mu)
  eff = mu + mut
  ux, uy = g[1]
  vx, vy = g[2]
  txx = eff * (4 / 3 * ux - 2 / 3 * vy)
  tyy = eff * (4 / 3 * vy - 2 / 3 * ux)
  txy = eff * (uy + vx)
  heat = cc.cp * (mu / cc.Pr + mut / cc.Prt) * g[3]
  diff = cc.inv_sigma * (mu + rho * nu) * g[4] if cc.flow_model.nvar == 5 else np.zeros(2)
  fx = np.array([0, txx, txy, u * txx + v * txy + heat[0], diff[0]])
  fy = np.array([0, txy, tyy, u * txy + v * tyy + heat[1], diff[1]])
  return fx * normal[0] + fy * normal[1]


def source(q, g, distance, molecular_mu=None):
  rho, _, _, _temperature, nu = q
  mu, _ = viscosity(q, molecular_mu)
  chi = rho * nu / mu
  fv1 = chi**3 / (chi**3 + cc.Cv1**3)
  fv2 = 1 - chi / (1 + chi * fv1)
  ft2 = cc.Ct3 * np.exp(-cc.Ct4 * chi**2)
  # Differentiate the active piece of the same guards used for the base flow.
  omega_raw = g[2, 0] - g[1, 1]
  omega = np.sign(omega_raw.real) * omega_raw
  raw_s = omega + fv2 * nu / (cc.kappa * distance) ** 2
  candidates = [raw_s, 0.3 * omega, 1e-20]
  s = candidates[int(np.argmax(np.real(candidates)))]
  raw_r = nu / (s * (cc.kappa * distance) ** 2)
  r = raw_r if raw_r.real < cc.rmax else cc.rmax
  gg = r + cc.Cw2 * (r**6 - r)
  fw = gg * ((1 + cc.Cw3**6) / (gg**6 + cc.Cw3**6)) ** (1 / 6)
  production = cc.Cb1 * (1 - ft2) * s * nu
  destruction = (cc.Cw1 * fw - cc.Cb1 / cc.kappa**2 * ft2) * (nu / distance) ** 2
  cross = cc.Cb2 * cc.inv_sigma * np.dot(g[4], g[4])
  # Conservative rho*nu form of rho*div((mu/rho+nu)*grad(nu)).
  density_correction = cc.inv_sigma * (mu / rho + nu) * np.dot(g[0], g[4])
  return rho * (production - destruction + cross) - density_correction


def jacobians(function, q, g):
  step = 1e-30
  sample = np.atleast_1d(function(q, g))
  jq = np.empty((len(sample), 5))
  jg = np.empty((len(sample), 5, 2))
  for j in range(5):
    perturbed = q.astype(complex)
    perturbed[j] += 1j * step
    jq[:, j] = np.imag(function(perturbed, g)) / step
    for axis in range(2):
      perturbed_g = g.astype(complex)
      perturbed_g[j, axis] += 1j * step
      jg[:, j, axis] = np.imag(function(q, perturbed_g)) / step
  return jq, jg
