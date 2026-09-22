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
  parser.add_argument(
    '--alpha',
    type=float,
    default=cc.alpha_H,
    help='混合格式系数 alpha_H；论文 (3.1.19) 允许 0 <= alpha_H <= 1',
  )
  parser.add_argument(
    '--far-riemann',
    action='store_true',
    default=cc.far_riemann,
    help='远场外边界改用单向迎风(Riemann)闭合，取代论文 (2.3.6)/(2.3.7) 的特征约束',
  )
  parser.add_argument(
    '--viscous-face-central',
    action='store_true',
    default=cc.viscous_face_central,
    help='黏性/热传导/SA 扩散的面法向导数改用两点中心差分（论文 3.1 的写法）',
  )
  parser.add_argument('--model', choices=[m.value for m in FlowModel], required=True)
  args = parser.parse_args()
  if not 0.0 <= args.alpha <= 1.0:
    raise ValueError('Mixed-scheme weight must satisfy 0 <= alpha_H <= 1')
  cc.alpha_H = args.alpha
  cc.far_riemann = args.far_riemann
  cc.viscous_face_central = args.viscous_face_central
  cc.flow_model = FlowModel(args.model)
  root = args.case.resolve()
  start = time.perf_counter()
  readrans.read_rans(str(root / 'ransdata.txt'), str(root / 'edge.txt'))
  parameters_path = root / 'input/parameters.json'
  parameters = json.loads(parameters_path.read_text(encoding='utf-8'))
  nu_tilde = np.array(
    [cc.goto_HALOcell((s, n)).miubl for n in range(1, cc.N_MAX + 1) for s in range(1, cc.S_MAX + 1)]
  )
  validate_base_model(cc.flow_model, parameters, nu_tilde)
  print(f'Loaded {cc.S_MAX} x {cc.N_MAX}; model={cc.flow_model.value}', flush=True)
  formmat._rows.clear()
  formmat._cols.clear()
  formmat._vals.clear()
  # 物面条件只通过虚单元镜像在**面**上施加（_WALL_MAP：u,v,ν̃ 取负 ⇒ 面平均值为 0，
  # ρ,T 同值 ⇒ 两点面法向导数为 0），物面环因此是普通守恒行。
  # 远场环在特征约束下仍是代数行；Riemann 闭合时它与内部环一样按面装配。
  if not cc.far_riemann:
    for s in range(1, cc.S_MAX + 1):
      boundary.far_boundary(cc.goto_HALOcell((s, cc.N_MAX)))
  for n in range(1, cc.N_MAX + 1):
    for s in range(1, cc.S_MAX + 1):
      c = cc.goto_HALOcell((s, n))
      grad.green_gauss_from_JST(c, c.north, c.south, c.east, c.west)
  # 虚单元的几何与基流梯度：与 formmat 的虚单元值映射一致；边界面黏性系数要用它。
  for s in range(1, cc.S_MAX + 1):
    wall = cc.goto_HALOcell((s, 1)).south
    outer = cc.goto_HALOcell((s, cc.N_MAX)).north
    for k in range(1, cc.HALO + 1):
      grad.mirror_ghost(
        cc.goto_HALOcell((s, 1 - k)), cc.goto_HALOcell((s, k)), wall, formmat._WALL_MAP
      )
      grad.mirror_ghost(
        cc.goto_HALOcell((s, cc.N_MAX + k)),
        cc.goto_HALOcell((s, cc.N_MAX + 1 - k)),
        outer,
        formmat._FAR_MAP,
      )
  for face in cc.FaceList_NS + cc.FaceList_WE:
    viscous.prepare_face_diffusion(face)
  print('Boundary and gradients ready', flush=True)
  for n in formmat.solved_rings():
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
    # 边界条件在边界面中点上取值（面值/面法向导数算子），不是首末单元中心。
    'boundary_derivative': 'quadratic-2d-face',
    'time_convention': 'exp(lambda*t)',
    'alpha_H': cc.alpha_H,
    'far_riemann': cc.far_riemann,
    'viscous_face_central': cc.viscous_face_central,
  }
  (root / 'assembly.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
  print(info, flush=True)


if __name__ == '__main__':
  main()
