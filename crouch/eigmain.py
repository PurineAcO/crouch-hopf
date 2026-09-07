"""Scaled, boundary-eliminated shift-invert stability analysis."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from models import FlowModel


def reduce_boundary(S, T, ns, nn, scales, time_scale):
  nvar = len(scales)
  fullscale = np.tile(scales, ns * nn)
  scaled = (sp.diags(1 / fullscale) @ S @ sp.diags(fullscale)).tocsr()
  interior = np.flatnonzero(T.diagonal())
  rows = interior.tolist()
  cols = np.arange(len(interior)).tolist()
  data = np.ones(len(interior)).tolist()
  for ring in [0, nn - 1]:
    for s in range(ns):
      indices = np.arange((ring * ns + s) * nvar, (ring * ns + s + 1) * nvar)
      block = scaled[indices][:, indices].toarray()
      off = scaled[indices][:, interior]
      rownorm = np.maximum(np.max(np.abs(block), axis=1), 1e-300)
      weights = -np.linalg.solve(block / rownorm[:, None], np.eye(nvar) / rownorm[:, None])
      mapped = (sp.csr_matrix(weights) @ off).tocoo()
      rows.extend(indices[mapped.row].tolist())
      cols.extend(mapped.col.tolist())
      data.extend(mapped.data.tolist())
  prolongation = sp.csr_matrix((data, (rows, cols)), shape=(S.shape[0], len(interior)))
  boundary = np.flatnonzero(T.diagonal() == 0)
  constraint = scaled[boundary] @ prolongation
  constraint_error = float(np.max(np.abs(constraint.data))) if constraint.nnz else 0.0
  if constraint_error > 1e-7:
    raise ValueError(f'Boundary elimination error: {constraint_error}')
  reduced = (scaled[interior] @ prolongation * time_scale).tocsc()
  return reduced, prolongation, scaled, fullscale, constraint_error


def wake_sector(ns, interior_rings, nvar=4):
  """Reflection-antisymmetric basis; nu-tilde has the same parity as rho."""
  if ns % 2:
    raise ValueError('Reflection reduction requires an even azimuthal count')
  half = ns // 2
  full = np.arange(interior_rings * ns * nvar).reshape(interior_rings, ns, nvar)
  left = full[:, :half].ravel()
  right = full[:, ::-1][:, :half].ravel()
  columns = np.arange(len(left))
  parity = np.tile([-1, -1, 1, -1, -1][:nvar], interior_rings * half)
  return sp.csc_matrix(
    (
      np.concatenate([np.ones(len(left)), parity]) / np.sqrt(2),
      (np.concatenate([left, right]), np.concatenate([columns, columns])),
    ),
    shape=(len(left) * 2, len(left)),
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('case', type=Path)
  parser.add_argument('--sigma-real', type=float, default=0)
  parser.add_argument('--sigma-imag', type=float, default=0.75)
  parser.add_argument('--k', type=int, default=12)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--label', default='wake')
  parser.add_argument('--wake-symmetry', action='store_true')
  args = parser.parse_args()
  root = args.case.resolve()
  parameters = json.loads((root / 'input/parameters.json').read_text())
  assembly = json.loads((root / 'assembly.json').read_text())
  model = FlowModel(assembly['model'])
  nvar = model.nvar
  if parameters['model'] != model.value:
    raise ValueError('Base-flow and assembled operator models disagree')
  ns, nn = parameters['nt'], parameters['nr']
  unit_time = parameters['D_m'] / parameters['U_m_s']
  scales = np.array(
    [
      parameters['rho_kg_m3'],
      parameters['U_m_s'],
      parameters['U_m_s'],
      parameters['T_K'],
    ]
  )
  if model is FlowModel.SA:
    scales = np.append(scales, parameters['mu_Pa_s'] / parameters['rho_kg_m3'])
  if not np.isfinite(scales).all() or np.any(scales <= 0) or unit_time <= 0:
    raise ValueError('Reference scales must be finite and positive')
  S = sp.load_npz(root / 'S.npz')
  T = sp.load_npz(root / 'T.npz')
  expected = (ns * nn * nvar, ns * nn * nvar)
  if S.shape != expected or T.shape != expected:
    raise ValueError(f'{model.value} matrices must have shape {expected}')
  A, P, scaled, _fullscale, bc_error = reduce_boundary(S, T, ns, nn, scales, unit_time)
  symmetry_error = None
  if args.wake_symmetry:
    perm = np.arange(A.shape[0]).reshape(nn - 2, ns, nvar)[:, ::-1].ravel()
    signs = sp.diags(np.tile([1, 1, -1, 1, 1][:nvar], (nn - 2) * ns))
    mirrored = signs @ A.tocsr()[perm][:, perm] @ signs
    symmetry_error = float(spla.norm(A - mirrored) / spla.norm(A))
    if symmetry_error > 1e-10:
      raise ValueError(f'Operator is not reflection invariant: {symmetry_error}')
    sector = wake_sector(ns, nn - 2, nvar)
    A = (sector.T @ A @ sector).tocsc()
    P = P @ sector
  print(f'Reduced to {A.shape}, nnz={A.nnz}; boundary error={bc_error:.3e}', flush=True)
  sigma = complex(args.sigma_real, args.sigma_imag)
  start = time.perf_counter()
  factor = spla.splu(A.astype(complex) - sigma * sp.eye(A.shape[0], format='csc'))
  factor_seconds = time.perf_counter() - start
  print(f'LU: {factor_seconds:.2f}s; factor nnz={factor.L.nnz + factor.U.nnz}', flush=True)
  inverse = spla.LinearOperator(A.shape, matvec=factor.solve, dtype=np.complex128)
  rng = np.random.default_rng(args.seed)
  eigenvalues, eigenvectors = spla.eigs(
    A.astype(complex),
    k=args.k,
    sigma=sigma,
    which='LM',
    OPinv=inverse,
    ncv=max(3 * args.k, 40),
    v0=rng.normal(size=A.shape[0]),
    tol=1e-9,
    maxiter=3000,
  )
  full = P @ eigenvectors
  rows = []
  for i, value in enumerate(eigenvalues):
    rhs = value * (T @ full[:, i])
    lhs = unit_time * (scaled @ full[:, i])
    residual = np.linalg.norm(lhs - rhs) / (np.linalg.norm(lhs) + np.linalg.norm(rhs) + 1e-300)
    if residual > 1e-6:
      raise ValueError(f'Eigenpair {i} has residual {residual}')
    rows.append(
      [
        i,
        value.real,
        value.imag,
        value.imag / (2 * np.pi),
        value.real / unit_time,
        value.imag / (2 * np.pi * unit_time),
        residual,
      ]
    )
  rows = np.asarray(rows)
  np.savetxt(
    root / f'{args.label}_eigenvalues.csv',
    rows,
    delimiter=',',
    header='mode,growth_D_U,omega_D_U,St,growth_per_s,frequency_Hz,residual',
    comments='',
  )
  np.savez_compressed(
    root / f'{args.label}_modes.npz', eigenvalues=eigenvalues, modes=full, scales=scales
  )
  info = {
    'model': model.value,
    'nvar': nvar,
    'sigma': [sigma.real, sigma.imag],
    'wake_symmetry': args.wake_symmetry,
    'reflection_error': symmetry_error,
    'k': args.k,
    'seed': args.seed,
    'factor_seconds': factor_seconds,
    'total_seconds': time.perf_counter() - start,
    'factor_nnz': factor.L.nnz + factor.U.nnz,
    'max_residual': float(rows[:, -1].max()),
    'boundary_error': bc_error,
  }
  (root / f'{args.label}_solve.json').write_text(json.dumps(info, indent=2) + '\n')
  print(rows, flush=True)
  print(info, flush=True)


if __name__ == '__main__':
  main()
