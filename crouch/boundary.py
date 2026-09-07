"""Algebraic perturbation conditions at the innermost/outermost cell-center rings."""

import classconfig as cc
import numpy as np


def normal_derivative(points, normal):
  normal = np.asarray(normal) / np.linalg.norm(normal)
  coordinates = (np.asarray(points) - points[0]) @ normal
  scale = np.max(np.abs(coordinates))
  if scale <= 0:
    raise ValueError('Repeated boundary points')
  x = coordinates / scale
  return np.linalg.solve(np.array([np.ones(3), x, x * x]), np.array([0, 1 / scale, 0]))


def _weights(cell, side):
  first = getattr(getattr(cell, side), side)
  second = getattr(getattr(first, side), side)
  face = cell.south if side == 'north' else cell.north
  return normal_derivative(np.array([[c.x, c.y] for c in [cell, first, second]]), face.jacobian[0])


def wing_boundary(cell):
  if cell.index[1] != 1:
    raise ValueError('Not a wall ring cell')
  weights = _weights(cell, 'north')
  for i, name in enumerate(['c', 'n', 'nn']):
    block = np.diag([weights[i], 0, 0, weights[i], 0])
    if i == 0:
      block[1, 1] = block[2, 2] = block[4, 4] = 1
    cell.form_influence(cc.dic[name], block)


def far_boundary(cell):
  if cell.index[1] != cc.N_MAX:
    raise ValueError('Not a farfield ring cell')
  ct = np.sqrt(cc.gamma * cc.R * cell.T) / ((cc.gamma - 1) * cell.T)
  kr = -cc.R * (cc.gamma - 1) * cell.T / cell.rho**cc.gamma
  kt = cc.R / cell.rho ** (cc.gamma - 1)
  nx, ny = cell.north.jacobian[0] / np.linalg.norm(cell.north.jacobian[0])
  plus = np.array([0, nx, ny, ct, 0])
  minus = np.array([0, nx, ny, -ct, 0])
  tangent = np.array([0, -ny, nx, 0, 0])
  entropy = np.array([kr, 0, 0, kt, 0])
  sa = np.array([0, 0, 0, 0, 1])
  inflow = cell.north.vn <= 0
  weights = _weights(cell, 'south')
  for i, name in enumerate(['c', 's', 'ss']):
    block = np.zeros((5, 5))
    block[0] = weights[i] * plus
    if i == 0:
      block[1] = minus
    if inflow:
      if i == 0:
        block[2:] = [tangent, entropy, sa]
    else:
      block[2:] = weights[i] * np.array([tangent, entropy, sa])
    cell.form_influence(cc.dic[name], block)
