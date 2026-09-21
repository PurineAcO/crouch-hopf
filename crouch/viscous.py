"""按原版 13 个局部块装配黏性与 SA 解析系数；S 表示时间右端。"""

import classconfig as cc
import grad
import numpy as np
from linearization import gradients, source_coefficients, state, viscous_coefficients


def prepare_face_diffusion(face):
  # Evaluate the constitutive law at the face state, not by averaging nonlinear derivatives.
  face._q = (state(face.me) + state(face.nei)) / 2
  face._g = (gradients(face.me) + gradients(face.nei)) / 2
  face._diffusion = None


def face_diffusion(face):
  if face._diffusion is not None:
    return face._diffusion
  jq, jg = viscous_coefficients(face._q, face._g, face.jacobian[0])
  operator = grad.green_gauss_face_vari(face)
  names = (
    ['w', 'e', 'nw', 'ne', 'sw', 'se', 'ee', 'ww']
    if face.direction == 'WE'
    else ['s', 'n', 'se', 'sw', 'ss', 'nw', 'nn', 'ne']
  )
  gradient_coefficients = jg
  normal = None
  scale = 0.0
  if cc.viscous_face_central:
    # 面法向导数改用两点中心差分，切向分量仍取两单元平均梯度但去掉其法向部分。
    # Green-Gauss 的面值算术平均会把 2Δ（棋盘格）模态整体消掉，使黏性、热传导与
    # SA 扩散对格点分支零响应；论文 3.1 用的是 second-order central difference。
    area = np.asarray(face.jacobian[0], dtype=float)
    unit = area / np.linalg.norm(area)
    first, second = (face.west, face.east) if face.direction == 'WE' else (face.south, face.north)
    offset = np.array([second.x - first.x, second.y - first.y])
    distance = float(np.linalg.norm(offset))
    if distance <= 0:
      raise ValueError(f'Degenerate face centre distance at {face.direction}')
    direction = offset / distance
    scale = float(direction @ unit) / distance
    normal = np.einsum('ija,a->ij', jg, unit)
    gradient_coefficients = np.einsum('ija,ab->ijb', jg, np.eye(2) - np.outer(unit, unit))
  result = []
  for i, name in enumerate(names):
    block = np.einsum('ija,a->ij', gradient_coefficients, operator[name])
    if i < 2:
      block = block + jq / 2
      if normal is not None:
        # names[0] 是西/南侧单元，names[1] 是东/北侧单元。
        block = block + (scale if i else -scale) * normal
    result.append(block)
  face._diffusion = result
  return result


def cell_diffusion(cell):
  for face, sign, directions in [
    (cell.east, 1, ['c', 'e', 'n', 'ne', 's', 'se', 'ee', 'w']),
    (cell.west, -1, ['w', 'c', 'nw', 'n', 'sw', 's', 'e', 'ww']),
    (cell.south, -1, ['s', 'c', 'se', 'sw', 'ss', 'w', 'n', 'e']),
    (cell.north, 1, ['c', 'n', 'e', 'w', 's', 'nw', 'nn', 'ne']),
  ]:
    for name, block in zip(directions, face_diffusion(face)):
      cell.form_influence(cc.dic[name], sign * block / cell.vol)


def cell_source(cell):
  q0 = state(cell)
  jq, jg = source_coefficients(q0, gradients(cell), cell.sad)
  for name, weights in grad.green_gauss_cell_vari(cell).items():
    block = np.zeros((5, 5))
    block[4] = jg @ weights
    if name == 'c':
      block[4] += jq
    cell.form_influence(cc.dic[name], block)
