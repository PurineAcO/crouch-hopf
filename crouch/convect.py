"""Crouch et al. (2007), equations 3.1.10--19; first-order SA convection."""

import classconfig as cc
import numpy as np


def _metric(cell, direction):
  # Coordinate increments from opposing face midpoints. Cofactors, not tangents,
  # transform Cartesian fluxes (3.1.11). Orient along increasing logical index.
  ds, dn = np.asarray(cell.jacobian)
  determinant = ds[0] * dn[1] - ds[1] * dn[0]
  if determinant == 0:
    raise ValueError(f'Degenerate coordinate metric at {cell.index}')
  cofactor = np.array([dn[1], -dn[0]]) if direction == 'WE' else np.array([-ds[1], ds[0]])
  return np.sign(determinant) * cofactor


def _flux_jacobian(cell, normal):
  jac = normal[0] * cell.F + normal[1] * cell.G
  if cc.active_model().nvar == 5:
    fx, fy = cell.sa_convect_vec()
    jac[4] = normal[0] * fx + normal[1] * fy
  return jac


def _face_stencil(face):
  if face.direction == 'WE':
    left, right = face.west, face.east
    cells = [left.west.west, left, right, right.east.east]
  else:
    left, right = face.south, face.north
    cells = [left.south.south, left, right, right.north.north]
  normals = [_metric(c, face.direction) for c in cells]
  matrices = np.array([_flux_jacobian(c, m) for c, m in zip(cells, normals)])
  normal = (normals[1] + normals[2]) / 2
  speed = normal @ np.array([(left.u + right.u) / 2, (left.v + right.v) / 2])
  # On symmetry faces the true normal speed is zero. Roundoff must not
  # choose a one-sided acoustic flux and break reflection symmetry.
  speed_scale = np.linalg.norm(normal) * max(np.hypot(c.u, c.v) for c in (left, right))
  sign = 0.0 if abs(speed) <= 64 * np.finfo(float).eps * speed_scale else np.sign(speed)
  minus = np.array([-1 / 6, 5 / 6, 1 / 3, 0])
  plus = np.array([0, 1 / 3, 5 / 6, -1 / 6])
  a_minus = np.einsum('i,ijk->jk', minus, matrices)
  a_plus = np.einsum('i,ijk->jk', plus, matrices)
  # The printed 3.1.14 repeats 1-sign: the left/minus weight must be 1+sign
  # to implement its stated upwind selection and preserve a constant flux.
  w_minus = (1 + cc.alpha_H * sign) / 2
  w_plus = (1 - cc.alpha_H * sign) / 2
  result = []
  for i in range(4):
    block = w_minus * minus[i] * a_minus + w_plus * plus[i] * a_plus
    # Page 930: SA convection is first-order upwind, independent of alpha_H.
    block[4] = 0
    if i == 1:
      block[4] = (1 + sign) / 2 * matrices[1, 4]
    elif i == 2:
      block[4] = (1 - sign) / 2 * matrices[2, 4]
    result.append(block)
  return result


def convect_hybrid(cell):
  for face, sign, names in [
    (cell.east, 1, ['w', 'c', 'e', 'ee']),
    (cell.west, -1, ['ww', 'w', 'c', 'e']),
    (cell.north, 1, ['s', 'c', 'n', 'nn']),
    (cell.south, -1, ['ss', 's', 'c', 'n']),
  ]:
    for name, block in zip(names, _face_stencil(face)):
      cell.form_influence(cc.dic[name], -sign * block / cell.vol)
