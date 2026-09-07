"""Viscous and SA Jacobian assembly. Matrix S represents the temporal RHS."""

import classconfig as cc
import grad
import numpy as np
from linearization import gradients, jacobians, source, state, viscosity, viscous_flux


def prepare_face_diffusion(face):
  # Evaluate the constitutive law at the face state, not by averaging nonlinear derivatives.
  face._q = (state(face.me) + state(face.nei)) / 2
  face._g = (gradients(face.me) + gradients(face.nei)) / 2
  face._diffusion = None


def face_diffusion(face):
  if face._diffusion is not None:
    return face._diffusion
  mu = viscosity(face._q)[0]
  jq, jg = jacobians(
    lambda q, g: viscous_flux(q, g, face.jacobian[0], molecular_mu=mu), face._q, face._g
  )
  operator = grad.green_gauss_face_vari(face)
  names = (
    ['w', 'e', 'nw', 'ne', 'sw', 'se', 'ee', 'ww']
    if face.direction == 'WE'
    else ['s', 'n', 'se', 'sw', 'ss', 'nw', 'nn', 'ne']
  )
  result = []
  for i, name in enumerate(names):
    block = np.einsum('ija,a->ij', jg, operator[name])
    if i < 2:
      block = block + jq / 2
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
  mu = viscosity(q0)[0]
  jq, jg = jacobians(lambda q, g: source(q, g, cell.sad, molecular_mu=mu), q0, gradients(cell))
  for name, weights in grad.green_gauss_cell_vari(cell).items():
    block = np.zeros((5, 5))
    block[4] = np.einsum('ja,a->j', jg[0], weights)
    if name == 'c':
      block[4] += jq[0]
    cell.form_influence(cc.dic[name], block)
