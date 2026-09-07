"""Algebraic ring conditions in the original thirteen five-variable blocks."""

import classconfig as cc
import numpy as np


def _unit_normal(normal):
  normal = np.asarray(normal, dtype=float)
  length = np.linalg.norm(normal)
  if normal.shape != (2,) or not np.isfinite(length) or length <= 0:
    raise ValueError('Boundary normal must be finite and nonzero')
  return normal / length


def normal_derivative(points, normal, preferred_stencil=None):
  """Weights reproducing n.grad of every 2D quadratic at points[0].

  SVD whitening removes both stretching and shear before fitting
  [1, z0, z1, z0**2, z0*z1, z1**2]. A preferred three-point radial stencil
  preserves the normal-aligned limit. Its weights are always corrected against
  all 2D moments, so projected distances alone never define a skew derivative.
  No lower-order fallback is permitted.
  """
  points = np.asarray(points, dtype=float)
  normal = _unit_normal(normal)
  if points.ndim != 2 or points.shape[1] != 2 or len(points) < 6:
    raise ValueError('A 2D quadratic boundary stencil needs at least six points')
  if not np.isfinite(points).all():
    raise ValueError('Boundary points must be finite')
  delta = points - points[0]
  _, singular, vt = np.linalg.svd(delta, full_matrices=False)
  if singular[-1] <= np.finfo(float).eps * max(delta.shape) * singular[0]:
    raise ValueError('Rank-deficient boundary geometry')
  transform = vt.T / singular
  x, y = (delta @ transform).T
  design = np.column_stack([np.ones(len(points)), x, y, x * x, x * y, y * y])
  target = np.r_[0, normal @ transform, 0, 0, 0]
  prior = np.zeros(len(points))
  if preferred_stencil is not None:
    slots = np.asarray(preferred_stencil, dtype=int)
    if slots.shape != (3,) or len(set(slots)) != 3 or np.any((slots < 0) | (slots >= len(points))):
      raise ValueError('Preferred boundary stencil must contain three distinct point indices')
    coordinates = (delta @ normal)[slots]
    scale = np.max(abs(coordinates))
    if scale > 0:
      z = coordinates / scale
      radial = np.array([np.ones(3), z, z * z])
      # An ill-conditioned prior is unnecessary: the full 2D constraints still
      # determine a valid stencil, including when radial projections coincide.
      if np.linalg.cond(radial) < 1e6:
        prior[slots] = np.linalg.solve(radial, np.array([0, 1 / scale, 0]))
  correction, _, rank, _ = np.linalg.lstsq(design.T, target - design.T @ prior, rcond=None)
  weights = prior + correction
  if rank != 6:
    raise ValueError('Rank-deficient quadratic boundary stencil')
  if not np.isfinite(weights).all():
    raise ValueError('Nonfinite boundary derivative weights')
  return weights


def _neighbor(cell, side):
  return getattr(getattr(cell, side), side)


def _stencil(cell, side):
  first = _neighbor(cell, side)
  cells = [
    cell,
    _neighbor(cell, 'east'),
    _neighbor(cell, 'west'),
    first,
    _neighbor(first, side),
    _neighbor(first, 'east'),
    _neighbor(first, 'west'),
  ]
  radial = 'n' if side == 'north' else 's'
  names = ['c', 'e', 'w', radial, radial * 2, radial + 'e', radial + 'w']
  face = cell.south if side == 'north' else cell.north
  normal = _unit_normal(face.jacobian[0])
  weights = normal_derivative([[c.x, c.y] for c in cells], normal, preferred_stencil=[0, 3, 4])
  return names, cells, weights, normal


def _add_blocks(cell, names, blocks):
  if not np.isfinite(blocks).all():
    raise ValueError(f'Nonfinite boundary constraints at {cell.index}')
  for name, block in zip(names, blocks):
    cell.form_influence(cc.dic[name], block)


def wing_boundary(cell):
  if cell.index[1] != 1:
    raise ValueError('Not a wall ring cell')
  names, _, weights, _ = _stencil(cell, 'north')
  blocks = np.zeros((len(names), 5, 5))
  blocks[:, 0, 0] = blocks[:, 3, 3] = weights
  blocks[0, 1, 1] = blocks[0, 2, 2] = blocks[0, 4, 4] = 1
  _add_blocks(cell, names, blocks)


def characteristics(cell, normal):
  """Paper invariant derivatives, ordered plus, minus, tangent, entropy, SA.

  I+/- = u_n +/- 2*sqrt(gamma*R*T)/(gamma-1), E = R*T/rho**(gamma-1).
  Their acoustic rows differ from Euler left eigenvectors by
  +/- rho**(gamma-1)/((gamma-1)*a) * dE when entropy is perturbed.
  Speeds assign the paper's incoming/outgoing families under an outward normal;
  these rows are not a decoupled Euler characteristic basis for arbitrary dE.
  """
  nx, ny = _unit_normal(normal)
  if not np.isfinite([cell.rho, cell.T, cell.u, cell.v]).all() or min(cell.rho, cell.T) <= 0:
    raise ValueError('Characteristic base state must be finite with positive rho and T')
  a = np.sqrt(cc.gamma * cc.R * cell.T)
  temperature = np.array(
    [0, 0, 0, np.sqrt(cc.gamma * cc.R) / ((cc.gamma - 1) * np.sqrt(cell.T)), 0]
  )
  velocity = np.array([0, nx, ny, 0, 0])
  entropy = [
    -(cc.gamma - 1) * cc.R * cell.T / cell.rho**cc.gamma,
    0,
    0,
    cc.R / cell.rho ** (cc.gamma - 1),
    0,
  ]
  rows = np.array(
    [velocity + temperature, velocity - temperature, [0, -ny, nx, 0, 0], entropy, [0, 0, 0, 0, 1]]
  )
  un = nx * cell.u + ny * cell.v
  return rows, np.array([un + a, un - a, un, un, un])


def far_boundary(cell):
  if cell.index[1] != cc.N_MAX:
    raise ValueError('Not a farfield ring cell')
  names, cells, weights, normal = _stencil(cell, 'south')
  center, speeds = characteristics(cell, normal)
  # Zero-speed modes are prescribed, including exactly sonic acoustic modes.
  incoming = speeds <= 0
  blocks = np.zeros((len(names), 5, 5))
  for i, (neighbor, weight) in enumerate(zip(cells, weights)):
    coefficients, _ = characteristics(neighbor, normal)
    blocks[i, ~incoming] = weight * coefficients[~incoming]
  blocks[0, incoming] = center[incoming]
  _add_blocks(cell, names, blocks)
