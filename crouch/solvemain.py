"""Assemble Crouch-Py for the supplied base flow; retain physical units in S."""

import argparse
import json
import time
from pathlib import Path

import boundary
import classconfig as cc
import convect
import formmat
import grad
import numpy as np
import readrans
import scipy.sparse as sp
import viscous
from models import SA_FORMULATION, FlowModel, validate_base_model


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('case', type=Path)
  parser.add_argument('--alpha', type=float, choices=[0.0, 0.2, 1.0], default=cc.alpha_H)
  parser.add_argument('--model', choices=[m.value for m in FlowModel], required=True)
  args = parser.parse_args()
  cc.alpha_H = args.alpha
  cc.flow_model = FlowModel(args.model)
  root = args.case.resolve()
  start = time.perf_counter()
  readrans.read_rans(str(root / 'ransdata.txt'), str(root / 'edge.txt'))
  parameters_path = root / 'input/parameters.json'
  parameters = json.loads(parameters_path.read_text())
  nu_tilde = np.array(
    [cc.goto_HALOcell((s, n)).miubl for n in range(1, cc.N_MAX + 1) for s in range(1, cc.S_MAX + 1)]
  )
  validate_base_model(cc.flow_model, parameters, nu_tilde)
  print(f'Loaded {cc.S_MAX} x {cc.N_MAX}; model={cc.flow_model.value}', flush=True)
  formmat._rows.clear()
  formmat._cols.clear()
  formmat._vals.clear()
  for s in range(1, cc.S_MAX + 1):
    boundary.wing_boundary(cc.goto_HALOcell((s, 1)))
    boundary.far_boundary(cc.goto_HALOcell((s, cc.N_MAX)))
  for n in range(1, cc.N_MAX + 1):
    for s in range(1, cc.S_MAX + 1):
      c = cc.goto_HALOcell((s, n))
      grad.green_gauss_from_JST(c, c.north, c.south, c.east, c.west)
  for face in cc.FaceList_NS + cc.FaceList_WE:
    viscous.prepare_face_diffusion(face)
  print('Boundary and gradients ready', flush=True)
  for n in range(2, cc.N_MAX):
    for s in range(1, cc.S_MAX + 1):
      c = cc.goto_HALOcell((s, n))
      convect.convect_hybrid(c)
      viscous.cell_diffusion(c)
      if cc.flow_model is FlowModel.SA:
        viscous.cell_source(c)
      if not np.isfinite(c.influence).all():
        raise ValueError(f'Nonfinite block at {(s, n)}')
    if n % 8 == 0:
      print(f'Linearized ring {n}/{cc.N_MAX}; {time.perf_counter() - start:.1f}s', flush=True)
  for n in range(1, cc.N_MAX + 1):
    for s in range(1, cc.S_MAX + 1):
      formmat.formmat(cc.goto_HALOcell((s, n)))
  S, T = formmat.build()
  # Local blocks share five entries; laminar matrices expose exactly four.
  # Check decoupling before discarding the inactive SA row/column.
  if cc.flow_model is FlowModel.LAMINAR:
    keep = np.arange(S.shape[0]).reshape(-1, 5)[:, :4].ravel()
    extra = np.arange(S.shape[0]).reshape(-1, 5)[:, 4]
    coupling = S[keep][:, extra]
    if coupling.nnz and np.any(coupling.data != 0):
      raise ValueError('Unexpected coupling to SA in laminar operator')
    S = S[keep][:, keep].tocsc()
    T = T.tocsr()[keep][:, keep].tocsc()
  assert np.isfinite(S.data).all()
  sp.save_npz(root / 'S.npz', S)
  sp.save_npz(root / 'T.npz', T)
  info = {
    'thermodynamics': parameters['thermodynamics'],
    'shape': S.shape,
    'nnz': S.nnz,
    'assembly_seconds': time.perf_counter() - start,
    'grid': [cc.S_MAX, cc.N_MAX],
    'model': cc.flow_model.value,
    'nvar': cc.flow_model.nvar,
    'molecular_viscosity_linearization': 'frozen',
    'coefficient_method': 'analytic',
    'sa_formulation': SA_FORMULATION if cc.flow_model is FlowModel.SA else None,
    'boundary_derivative': 'quadratic-2d',
    'time_convention': 'exp(lambda*t)',
    'alpha_H': cc.alpha_H,
  }
  (root / 'assembly.json').write_text(json.dumps(info, indent=2) + '\n')
  print(info, flush=True)


if __name__ == '__main__':
  main()
