"""Scaled full sparse generalized stability eigenproblem."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from models import FlowModel

try:
  import resource
except ImportError:  # Native Windows: RSS metadata is optional, solving is not.
  resource = None


def peak_rss_bytes():
  """Process peak RSS where supported; no dependency is required on Windows."""
  if resource is None:
    return None
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
    1 if sys.platform == 'darwin' else 1024
  )


def scale_pencil(S, T, scales, time_scale):
  """q = D*x and t = time_scale*tau; S is already the temporal RHS."""
  scales = np.asarray(scales, dtype=float)
  if (
    scales.ndim != 1
    or not scales.size
    or not np.isfinite(scales).all()
    or np.any(scales <= 0)
    or not np.isfinite(time_scale)
    or time_scale <= 0
  ):
    raise ValueError('Reference scales must be finite and positive')
  if S.shape != T.shape or S.shape[0] != S.shape[1] or S.shape[0] % len(scales):
    raise ValueError('Incompatible pencil shape and variable scales')
  if not np.isfinite(S.data).all() or not np.isfinite(T.data).all():
    raise ValueError('Pencil coefficients must be finite, including algebraic constraints')
  fullscale = np.tile(scales, S.shape[0] // len(scales))
  D, invD = sp.diags(fullscale), sp.diags(1 / fullscale)
  A = (time_scale * (invD @ S @ D)).tocsc()
  B = (invD @ T @ D).tocsc()
  if not np.isfinite(A.data).all() or not np.isfinite(B.data).all():
    raise ValueError('Scaled pencil coefficients must be finite')
  return A, B


def reflection_basis(ns, nn, nvar):
  """Orthonormal antisymmetric basis on ALL rings, including algebraic boundaries.

  Reflection is (s,n)->(ns-1-s,n) about y=0. For the wake sector,
  rho,u,T,nu change sign and v does not. There are no fixed cells for even ns.
  """
  if ns < 2 or ns % 2 or nn < 1 or nvar not in (4, 5):
    raise ValueError('Reflection requires even nt >= 2, nr >= 1, and four or five variables')
  indices = np.arange(ns * nn * nvar).reshape(nn, ns, nvar)
  left = indices[:, : ns // 2].ravel()
  right = indices[:, ::-1][:, : ns // 2].ravel()
  parity = np.tile(np.array([-1, -1, 1, -1, -1])[:nvar], nn * ns // 2)
  rows = np.column_stack([left, right]).ravel()
  data = np.column_stack([np.ones(len(left)), parity]).ravel() / np.sqrt(2)
  return sp.csc_matrix(
    (data, rows, np.arange(0, 2 * len(left) + 1, 2)), shape=(ns * nn * nvar, len(left))
  )


def project_reflection(A, B, ns, nn, nvar):
  """Require both full matrices to commute with reflection before projection.

  The maximum relative L1 defect of any equation is used, independently for A
  and B. This prevents large conservation rows hiding asymmetric boundary rows
  or an asymmetric mass matrix. Neither matrix is symmetrized or eliminated.
  """
  P = reflection_basis(ns, nn, nvar)
  if A.shape != (P.shape[0], P.shape[0]) or B.shape != A.shape:
    raise ValueError('Reflection grid and full pencil dimensions disagree')
  perm = np.arange(P.shape[0]).reshape(nn, ns, nvar)[:, ::-1].ravel()
  signs = np.tile(np.array([1, 1, -1, 1, 1])[:nvar], nn * ns)
  errors = {}
  for name, matrix in [('A', A), ('B', B)]:
    matrix = sp.csr_matrix(matrix)
    if not np.isfinite(matrix.data).all():
      raise ValueError(f'Reflection requires finite {name}')
    mirrored = matrix[perm][:, perm].multiply(signs[:, None]).multiply(signs)
    defect = np.asarray(abs(matrix - mirrored).sum(axis=1)).ravel()
    norm = np.asarray(abs(matrix).sum(axis=1)).ravel()
    denominator = np.maximum(np.maximum(norm, norm[perm]), np.finfo(float).tiny)
    errors[name] = float(np.max(defect / denominator))
    if errors[name] > 1e-10:
      raise ValueError(
        f'{name} does not commute with reflection: row-relative defect {errors[name]}'
      )
  return (P.T @ A @ P).tocsc(), (P.T @ B @ P).tocsc(), P, errors


def validate_reflection_input(root, ns, nn, scales, diameter):
  """Verify the supported O-grid's y=0 reflection without modifying input data.

  Check cell coordinates, volumes, wall distances, base fields, and face
  midpoints, normals and connectivity. The CLI refuses missing geometry, too;
  a symmetric matrix alone does not establish symmetry of a stale input file.
  """
  if ns < 2 or ns % 2 or nn < 1:
    raise ValueError('Reflection requires even nt >= 2 and nr >= 1')
  root = Path(root)
  if not (root / 'ransdata.txt').is_file() or not (root / 'edge.txt').is_file():
    raise ValueError('Reflection requires ransdata.txt and edge.txt geometry and base-flow inputs')
  rows = np.loadtxt(root / 'ransdata.txt', skiprows=1, ndmin=2)
  if rows.shape != (ns * nn, 11) or not np.isfinite(rows).all():
    raise ValueError('Invalid reflection base-flow table')
  rows = rows[np.lexsort((rows[:, 0], rows[:, 1]))]
  ids = np.column_stack([np.tile(np.arange(1, ns + 1), nn), np.repeat(np.arange(1, nn + 1), ns)])
  if not np.array_equal(rows[:, :2], ids):
    raise ValueError('Reflection base-flow indices do not cover the full grid')
  cells = rows[:, 2:].reshape(nn, ns, 9)
  if np.any(cells[:, :, [2, 3, 4, 7]] <= 0) or np.any(cells[:, :, 8] < 0):
    raise ValueError('Invalid reflection base-flow geometry or thermodynamic state')
  errors = {}

  def check(name, actual, reflected, floor=0.0):
    denominator = np.maximum(np.maximum(abs(actual), abs(reflected)), floor)
    error = float(np.max(abs(actual - reflected) / np.maximum(denominator, np.finfo(float).tiny)))
    errors[name] = error
    if not np.isfinite(error) or error > 1e-10:
      raise ValueError(f'Reflection input is not symmetric: {name}, relative defect {error}')

  floors = [
    diameter,
    diameter,
    0,
    0,
    scales[0],
    scales[1],
    scales[2],
    scales[3],
    scales[4] if len(scales) == 5 else 0,
  ]
  names = ['x', 'y', 'wall_distance', 'volume', 'rho', 'u', 'v', 'T', 'nu']
  for column, (name, floor) in enumerate(zip(names, floors)):
    parity = -1 if name in ('y', 'v') else 1
    check(name, cells[:, :, column], parity * cells[:, ::-1, column], floor)

  dtype = [('kind', 'U2')] + [
    (name, 'f8')
    for name in ['s', 'n', 'id', 's1', 'n1', 'id1', 's2', 'n2', 'id2', 'nx', 'ny', 'mx', 'my']
  ]
  edges = np.loadtxt(root / 'edge.txt', skiprows=1, dtype=dtype, ndmin=1)
  if len(edges) != ns * (2 * nn + 1) or np.any(~np.isin(edges['kind'], ['NS', 'WE'])):
    raise ValueError('Reflection requires complete O-grid face connectivity')
  for name in edges.dtype.names[1:]:
    if not np.isfinite(edges[name]).all():
      raise ValueError('Reflection requires finite face geometry')
  for kind, rings in [('NS', nn + 1), ('WE', nn)]:
    faces = edges[edges['kind'] == kind]
    faces = faces[np.lexsort((faces['s'], faces['n']))]
    s = np.tile(np.arange(1, ns + 1), rings)
    n = np.repeat(np.arange(1, rings + 1), ns)
    if (
      len(faces) != len(s) or not np.array_equal(faces['s'], s) or not np.array_equal(faces['n'], n)
    ):
      raise ValueError('Reflection face indices do not cover the full grid')
    expected = (
      [s, n - 1, s, np.where(n <= nn, n, 0)] if kind == 'NS' else [(s - 2) % ns + 1, n, s, n]
    )
    for field, values in zip(['s1', 'n1', 's2', 'n2'], expected):
      if not np.array_equal(faces[field], values):
        raise ValueError('Reflection requires the supported O-grid face connectivity')
    faces = faces.reshape(rings, ns)
    permutation = np.arange(ns)[::-1] if kind == 'NS' else (-np.arange(ns)) % ns
    length = np.hypot(faces['nx'], faces['ny'])
    if np.any(length <= 0):
      raise ValueError('Reflection face normals must be nonzero')
    for field in ['mx', 'my', 'nx', 'ny']:
      parity = (
        -1
        if field == 'my' or (field == 'ny' and kind == 'NS') or (field == 'nx' and kind == 'WE')
        else 1
      )
      floor = diameter if field in ('mx', 'my') else np.maximum(length, length[:, permutation])
      check(f'{kind}_{field}', faces[field], parity * faces[field][:, permutation], floor)
  return errors


def eigenpair_residuals(A, B, values, vectors):
  """Full pencil residuals, including a separate rowwise algebraic backward error."""
  if not np.isfinite(values).all() or not np.isfinite(vectors).all():
    raise ValueError('Nonfinite eigenpairs')
  lhs, mass = A @ vectors, B @ vectors
  rhs = mass * values
  residuals = np.linalg.norm(lhs - rhs, axis=0) / np.maximum(
    np.linalg.norm(lhs, axis=0) + np.linalg.norm(rhs, axis=0), np.finfo(float).tiny
  )
  mass_rows = B.tocsr(copy=True)
  mass_rows.eliminate_zeros()
  algebraic = np.diff(mass_rows.indptr) == 0
  constraint = np.asarray(abs(A[algebraic]).sum(axis=1)) * np.linalg.norm(vectors, axis=0)
  errors = abs(lhs[algebraic]) / np.maximum(constraint, np.finfo(float).tiny)
  bc_error = float(errors.max()) if errors.size else 0.0
  if not np.isfinite(residuals).all() or np.max(residuals) > 1e-6:
    raise ValueError(f'Full eigenpair residual exceeds tolerance: {residuals}')
  if not np.isfinite(bc_error) or bc_error > 1e-6:
    raise ValueError(f'Algebraic constraint residual exceeds tolerance: {bc_error}')
  return residuals, bc_error


def solve_reflection(A, B, ns, nn, nvar, sigma, k, seed=42, ordering='COLAMD'):
  reduced_A, reduced_B, P, errors = project_reflection(A, B, ns, nn, nvar)
  values, vectors, _, info = solve_generalized(reduced_A, reduced_B, sigma, k, seed, ordering)
  full = P @ vectors
  residuals, bc_error = eigenpair_residuals(A, B, values, full)
  info.update(boundary_error=bc_error, reflection_errors=errors, solve_dimension=P.shape[1])
  return values, full, residuals, info


def solve_generalized(A, B, sigma, k, seed=42, ordering='COLAMD'):
  """Use K=(A-sigma*B)^-1 B, preserving all coupled algebraic rows.

  Zero transformed eigenvalues represent infinite eigenvalues. Sparse LU avoids
  per-cell elimination and dense Schur complements. Check the unbalanced pencil.
  """
  A, B = sp.csc_matrix(A, dtype=complex), sp.csc_matrix(B, dtype=complex)
  A.eliminate_zeros()
  B.eliminate_zeros()
  n = A.shape[0]
  if A.shape != B.shape or A.shape != (n, n):
    raise ValueError('Pencil matrices must be square and have the same shape')
  if not np.isfinite(A.data).all() or not np.isfinite(B.data).all():
    raise ValueError('Pencil coefficients must be finite, including algebraic constraints')
  if not np.isfinite(sigma):
    raise ValueError('Shift must be finite')
  if not 1 <= k < n - 1:
    raise ValueError('ARPACK requires 1 <= k < matrix size - 1')
  rownorm = np.maximum(abs(A).max(axis=1).toarray().ravel(), abs(B).max(axis=1).toarray().ravel())
  if np.any(rownorm == 0):
    raise ValueError('Pencil contains an empty equation')
  balance = sp.diags(1 / rownorm)
  Ab, Bb = (balance @ A).tocsc(), (balance @ B).tocsc()
  start = time.perf_counter()
  if ordering not in ('COLAMD', 'MMD_AT_PLUS_A'):
    raise ValueError('Ordering must be COLAMD or MMD_AT_PLUS_A')
  factor = spla.splu(Ab - sigma * Bb, permc_spec=ordering)
  factor_seconds = time.perf_counter() - start
  inverse = spla.LinearOperator(
    A.shape, matvec=lambda x, lu=factor, mass=Bb: lu.solve(mass @ x), dtype=np.complex128
  )
  mu, vectors = spla.eigs(
    inverse,
    k=k,
    which='LM',
    ncv=min(n, max(3 * k, 40)),
    v0=np.random.default_rng(seed).normal(size=n),
    tol=1e-9,
    maxiter=3000,
  )
  if not np.isfinite(mu).all() or np.any(np.abs(mu) <= np.finfo(float).tiny):
    raise ValueError('Shift-invert returned an infinite or invalid eigenvalue')
  values = sigma + 1 / mu
  # SuperLU's storage count avoids materializing potentially multi-GiB L/U copies.
  info = {
    'factor_seconds': factor_seconds,
    'factor_nnz': factor.nnz,
    'factor_nnz_definition': 'SuperLU allocated factor storage; includes reserved slots',
    'ordering': ordering,
    'solve_dimension': n,
  }
  del inverse, factor, Ab, Bb
  residuals, bc_error = eigenpair_residuals(A, B, values, vectors)
  info['boundary_error'] = bc_error
  return values, vectors, residuals, info


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('case', type=Path)
  parser.add_argument('--sigma-real', type=float, default=0)
  parser.add_argument('--sigma-imag', type=float, default=0.75)
  parser.add_argument('--k', type=int, default=12)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--label', default='wake')
  parser.add_argument(
    '--wake-symmetry',
    action='store_true',
    help='Exact antisymmetric projection on all rings; requires symmetric input',
  )
  parser.add_argument('--ordering', choices=['COLAMD', 'MMD_AT_PLUS_A'], default='COLAMD')
  args = parser.parse_args()
  root = args.case.resolve()
  parameters = json.loads((root / 'input/parameters.json').read_text())
  assembly = json.loads((root / 'assembly.json').read_text())
  model = FlowModel(assembly['model'])
  nvar = model.nvar
  if parameters['model'] != model.value:
    raise ValueError('Base-flow and assembled operator models disagree')
  ns, nn = parameters['nt'], parameters['nr']
  diameter, speed = parameters['D_m'], parameters['U_m_s']
  if not np.isfinite([diameter, speed]).all() or min(diameter, speed) <= 0:
    raise ValueError('Reference scales must be finite and positive')
  unit_time = diameter / speed
  scales = np.array(
    [
      parameters['rho_kg_m3'],
      parameters['U_m_s'],
      parameters['U_m_s'],
      parameters['T_K'],
    ]
  )
  if not np.isfinite(scales).all() or np.any(scales <= 0):
    raise ValueError('Reference scales must be finite and positive')
  if model is FlowModel.SA:
    scales = np.append(scales, parameters['mu_Pa_s'] / parameters['rho_kg_m3'])
  if (
    not np.isfinite(scales).all()
    or np.any(scales <= 0)
    or not np.isfinite(unit_time)
    or unit_time <= 0
  ):
    raise ValueError('Reference scales must be finite and positive')
  input_errors = (
    validate_reflection_input(root, ns, nn, scales, diameter) if args.wake_symmetry else None
  )
  S = sp.load_npz(root / 'S.npz')
  T = sp.load_npz(root / 'T.npz')
  expected = (ns * nn * nvar, ns * nn * nvar)
  if S.shape != expected or T.shape != expected:
    raise ValueError(f'{model.value} matrices must have shape {expected}')
  A, B = scale_pencil(S, T, scales, unit_time)
  del S, T
  print(f'Full pencil {A.shape}, nnz={A.nnz}', flush=True)
  sigma = complex(args.sigma_real, args.sigma_imag)
  start = time.perf_counter()
  if args.wake_symmetry:
    eigenvalues, full, residuals, solve_info = solve_reflection(
      A, B, ns, nn, nvar, sigma, args.k, args.seed, args.ordering
    )
  else:
    eigenvalues, full, residuals, solve_info = solve_generalized(
      A, B, sigma, args.k, args.seed, args.ordering
    )
  rows = []
  for i, value in enumerate(eigenvalues):
    residual = residuals[i]
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
    'reflection_error': max(solve_info['reflection_errors'].values())
    if args.wake_symmetry
    else None,
    'reflection_errors': solve_info.get('reflection_errors'),
    'reflection_input_errors': input_errors,
    'ordering': args.ordering,
    'full_dimension': A.shape[0],
    'solve_dimension': solve_info['solve_dimension'],
    'factor_nnz_definition': solve_info['factor_nnz_definition'],
    'peak_rss_bytes': peak_rss_bytes(),
    'k': args.k,
    'seed': args.seed,
    'factor_seconds': solve_info['factor_seconds'],
    'total_seconds': time.perf_counter() - start,
    'factor_nnz': solve_info['factor_nnz'],
    'max_residual': float(rows[:, -1].max()),
    'boundary_error': solve_info['boundary_error'],
  }
  (root / f'{args.label}_solve.json').write_text(json.dumps(info, indent=2) + '\n')
  print(rows, flush=True)
  print(info, flush=True)


if __name__ == '__main__':
  main()
