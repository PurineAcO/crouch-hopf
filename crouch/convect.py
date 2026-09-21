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
  # On symmetry faces the true normal speed is zero. Use an acoustic scale
  # so absolute base-flow roundoff cannot select one-sided flux near stagnation.
  # At alpha_H=0 the four flow rows are unchanged. The SA row is first-order
  # upwind for every alpha_H, so its zero-speed tie is corrected as well.
  speed_scale = np.linalg.norm(normal) * max(
    max(np.hypot(c.u, c.v), np.sqrt(cc.gamma * cc.R * c.T)) for c in (left, right)
  )
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


def far_face_stencil(face):
  """远场外边界面的单向迎风（Riemann）闭合。

  外部状态的扰动取零，因此只保留出流信息：法向基流速度为正时，面通量由边界单元
  的单侧二阶重构给出，权重 (3/2, -1/2) 之和为 1，故常值扰动不产生伪通量；
  法向速度非正（入流）时扰动通量为零，即入流特征不被激励。SA 对流恒为一阶迎风，
  取边界单元自身值。

  与论文 (2.3.6)/(2.3.7) 的区别：出流特征不再用 ∂n=0 外推（那是部分反射的），
  而是直接以外部的零扰动做迎风选择，因此出流波可以自由离开计算域。
  """
  left, right = face.south, face.north
  cells = [left.south.south, left, right, right.north.north]
  normals = [_metric(c, face.direction) for c in cells]
  matrices = np.array([_flux_jacobian(c, m) for c, m in zip(cells, normals)])
  normal = (normals[1] + normals[2]) / 2
  speed = normal @ np.array([(left.u + right.u) / 2, (left.v + right.v) / 2])
  speed_scale = np.linalg.norm(normal) * max(
    max(np.hypot(c.u, c.v), np.sqrt(cc.gamma * cc.R * c.T)) for c in (left, right)
  )
  sign = 0.0 if abs(speed) <= 64 * np.finfo(float).eps * speed_scale else np.sign(speed)
  blocks = [np.zeros((5, 5)) for _ in range(4)]
  if sign > 0:
    # 用边界单元自身的通量雅可比，作用在单侧重构的扰动上。
    blocks[0] = -0.5 * matrices[1]
    blocks[1] = 1.5 * matrices[1]
    blocks[0][4] = 0.0
    blocks[1][4] = matrices[1][4]
  return blocks


def convect_hybrid(cell):
  for face, sign, names in [
    (cell.east, 1, ['w', 'c', 'e', 'ee']),
    (cell.west, -1, ['ww', 'w', 'c', 'e']),
    (cell.north, 1, ['s', 'c', 'n', 'nn']),
    (cell.south, -1, ['ss', 's', 'c', 'n']),
  ]:
    # 只有远场外边界那一面改用 Riemann 闭合；其余面与论文 3.1.10--19 完全相同。
    if cc.far_riemann and face is cell.north and cell.index[1] == cc.N_MAX:
      blocks = far_face_stencil(face)
    else:
      blocks = _face_stencil(face)
    for name, block in zip(names, blocks):
      cell.form_influence(cc.dic[name], -sign * block / cell.vol)
