"""Full descriptor reflection tests; references use independent dense algebra."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
from eigmain import (
  eigenpair_residuals,
  peak_rss_bytes,
  project_reflection,
  reflection_basis,
  scale_pencil,
  solve_generalized,
  solve_reflection,
  validate_reflection_input,
)


def _pencil(nvar=4, ns=6):
  # A reflection-invariant cycle, with physical u/v coupling and nonuniform
  # symmetric coefficients. Both algebraic rings couple azimuthally and to
  # each other. Their exact solution is wall=interior=far, so H is the finite
  # spectrum independently of the boundary matrix C.
  shift = np.roll(np.eye(ns), 1, axis=1)
  laplace = shift + shift.T - 2 * np.eye(ns)
  circulation = shift - shift.T
  cross = np.zeros((nvar, nvar))
  cross[1, 2] = cross[2, 1] = 1
  theta = 2 * np.pi * (np.arange(ns) + 0.5) / ns
  H = (
    np.kron(np.eye(ns), np.diag(-0.3 * np.arange(1, nvar + 1)))
    + 0.1 * np.kron(laplace, np.eye(nvar))
    + 0.08 * np.kron(circulation, cross)
    + 0.07 * np.kron(np.diag(np.cos(theta)), np.diag(np.arange(1, nvar + 1)))
  )
  eye = np.eye(ns * nvar)
  C = 2 * eye + 0.15 * np.kron(shift + shift.T, np.eye(nvar))
  A = sp.csc_matrix(
    np.block(
      [
        [C, -C - 0.1 * eye, 0.1 * eye],
        [0.2 * eye, H - 0.5 * eye, 0.3 * eye],
        [0.1 * eye, -C - 0.1 * eye, C],
      ]
    )
  )
  B = sp.diags(np.r_[np.zeros(ns * nvar), np.ones(ns * nvar), np.zeros(ns * nvar)], format='csc')
  return A, B, H


def _wake_reflect(vectors, ns, nn, nvar):
  shaped = vectors.reshape(nn, ns, nvar, -1)
  parity = np.array([-1, -1, 1, -1, -1])[:nvar]
  return (shaped[:, ::-1] * parity[None, None, :, None]).reshape(vectors.shape)


@pytest.mark.parametrize('nvar', [4, 5])
def test_full_basis_is_orthonormal_and_includes_both_boundaries(nvar):
  P = reflection_basis(6, 3, nvar)
  assert P.shape == (18 * nvar, 9 * nvar)
  np.testing.assert_allclose((P.T @ P).toarray(), np.eye(9 * nvar), atol=3e-16)
  values = P @ np.random.default_rng(7).normal(size=(9 * nvar, 2))
  np.testing.assert_array_equal(values, _wake_reflect(values, 6, 3, nvar))
  assert np.linalg.norm(values[: 6 * nvar]) > 0
  assert np.linalg.norm(values[-6 * nvar :]) > 0
  np.testing.assert_array_equal(np.diff(P.tocsr().indptr), 1)


@pytest.mark.parametrize('nvar', [4, 5])
@pytest.mark.parametrize('ordering', ['COLAMD', 'MMD_AT_PLUS_A'])
def test_full_coupled_projection_matches_independent_finite_spectrum(nvar, ordering):
  S, T, H = _pencil(nvar)
  scales = np.array([1e-5, 70, 70, 300, 1e-3])[:nvar]
  A, B = scale_pencil(S, T, scales, 0.2)
  sigma = -0.04 + 0.02j
  values, full, residuals, info = solve_reflection(A, B, 6, 3, nvar, sigma, 2, ordering=ordering)
  # The exact constraint solution gives eigenvalues .2*eig(H). Classify the
  # independent H eigenvectors by physical parity, without using P or P.T H P.
  known, vectors = la.eig(H)
  parity_error = np.linalg.norm(vectors - _wake_reflect(vectors, 6, 1, nvar), axis=0)
  candidates = 0.2 * known[parity_error < 1e-9]
  assert len(candidates) == 3 * nvar
  expected = candidates[np.argsort(abs(candidates - sigma))[:2]]
  np.testing.assert_allclose(np.sort_complex(values), np.sort_complex(expected), atol=2e-11)
  np.testing.assert_allclose(full[: 6 * nvar], full[6 * nvar : 12 * nvar], atol=2e-11)
  np.testing.assert_allclose(full[-6 * nvar :], full[6 * nvar : 12 * nvar], atol=2e-11)
  np.testing.assert_array_equal(full, _wake_reflect(full, 6, 3, nvar))
  assert residuals.max() < 1e-9 and info['boundary_error'] < 1e-10
  assert max(info['reflection_errors'].values()) < 1e-14
  assert info['ordering'] == ordering
  assert info['solve_dimension'] == 9 * nvar
  # Compare with the full descriptor solve too; it includes both parity sectors.
  all_values, _, _, _ = solve_generalized(A, B, sigma, 6, ordering=ordering)
  assert max(min(abs(all_values - value)) for value in values) < 2e-11


@pytest.mark.parametrize('matrix_name', ['A', 'B'])
def test_asymmetric_small_boundary_row_cannot_hide_behind_interior_scale(matrix_name):
  A, B, _ = _pencil()
  scales = np.r_[np.full(24, 1e-12), np.full(24, 1e12), np.full(24, 1e-12)]
  A, B = (sp.diags(scales) @ A).tolil(), (sp.diags(scales) @ B).tolil()
  matrix = A if matrix_name == 'A' else B
  matrix[0, 0] += 1e-13
  with pytest.raises(ValueError, match=f'{matrix_name} does not commute'):
    project_reflection(A.tocsc(), B.tocsc(), 6, 3, 4)


def test_asymmetric_mass_coupling_rejected():
  A, B, _ = _pencil()
  B = B.tolil()
  B[24, 25] = 0.2
  with pytest.raises(ValueError, match='B does not commute'):
    project_reflection(A, B.tocsc(), 6, 3, 4)


def test_lifted_constraints_are_checked_in_full_equations():
  A, B, _ = _pencil()
  values, full, _, _ = solve_reflection(A, B, 6, 3, 4, -0.04 + 0.02j, 1)
  # Poison just one retained boundary degree of freedom after lifting.
  full[0, 0] += 1
  with pytest.raises(ValueError, match='residual exceeds tolerance'):
    eigenpair_residuals(A, B, values, full)


@pytest.mark.parametrize('ns,nn,nvar', [(5, 3, 4), (0, 3, 4), (6, 0, 4), (6, 3, 3)])
def test_unsupported_reflection_layout_rejected(ns, nn, nvar):
  with pytest.raises(ValueError, match='Reflection requires'):
    reflection_basis(ns, nn, nvar)


def _input(root, nvar=4):
  ns, nn = 6, 3
  radii = np.linspace(0.5, 2, nn + 1)
  theta = np.arange(ns) * 2 * np.pi / ns
  xy = radii[:, None, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1)[None]
  cells = []
  for n in range(nn):
    for s in range(ns):
      x, y = np.mean([xy[n, s], xy[n, (s + 1) % ns], xy[n + 1, s], xy[n + 1, (s + 1) % ns]], axis=0)
      volume = 0.5 * (radii[n + 1] ** 2 - radii[n] ** 2) * np.sin(2 * np.pi / ns)
      cells.append(
        [
          s + 1,
          n + 1,
          x,
          y,
          np.hypot(x, y) - 0.4,
          volume,
          1 + 0.01 * x,
          1.0,
          0.1 * y,
          300 + 0.01 * x,
          1e-4 if nvar == 5 else 0.0,
        ]
      )
  np.savetxt(root / 'ransdata.txt', cells, header='s n x y sad vol rho u v T miubl', comments='')
  with (root / 'edge.txt').open('w') as stream:
    stream.write('type s n id c1_s c1_n c1_id c2_s c2_n c2_id nx ny mx my\n')
    for n in range(nn + 1):
      for s in range(ns):
        a, b = xy[n, s], xy[n, (s + 1) % ns]
        d = b - a
        data = [
          s + 1,
          n + 1,
          0,
          s + 1,
          n,
          0,
          s + 1,
          n + 1 if n < nn else 0,
          0,
          d[1],
          -d[0],
          *((a + b) / 2),
        ]
        stream.write('NS ' + ' '.join(map(str, data)) + '\n')
    for n in range(nn):
      for s in range(ns):
        a, b = xy[n, s], xy[n + 1, s]
        d = b - a
        data = [
          s + 1,
          n + 1,
          0,
          (s - 1) % ns + 1,
          n + 1,
          0,
          s + 1,
          n + 1,
          0,
          -d[1],
          d[0],
          *((a + b) / 2),
        ]
        stream.write('WE ' + ' '.join(map(str, data)) + '\n')
  return np.array([1.0, 1.0, 1.0, 300.0, 1e-5])[:nvar]


def test_symmetric_input_including_normals_and_connectivity(tmp_path):
  scales = _input(tmp_path)
  errors = validate_reflection_input(tmp_path, 6, 3, scales, 1.0)
  assert max(errors.values()) < 1e-13


@pytest.mark.parametrize('column', [2, 3, 4, 5, 6, 7, 8, 9])
def test_asymmetric_cell_geometry_or_base_state_rejected(tmp_path, column):
  scales = _input(tmp_path)
  rows = np.loadtxt(tmp_path / 'ransdata.txt', skiprows=1)
  rows[0, column] += 0.01
  np.savetxt(tmp_path / 'ransdata.txt', rows, header='s n x y sad vol rho u v T miubl', comments='')
  with pytest.raises(ValueError, match='not symmetric'):
    validate_reflection_input(tmp_path, 6, 3, scales, 1.0)


@pytest.mark.parametrize('column', [4, 10, 11, 12, 13])
def test_asymmetric_face_or_connectivity_rejected(tmp_path, column):
  scales = _input(tmp_path)
  lines = (tmp_path / 'edge.txt').read_text().splitlines()
  first = lines[1].split()
  first[column] = str(float(first[column]) + 0.01)
  lines[1] = ' '.join(first)
  (tmp_path / 'edge.txt').write_text('\n'.join(lines) + '\n')
  with pytest.raises(ValueError, match='not symmetric|connectivity'):
    validate_reflection_input(tmp_path, 6, 3, scales, 1.0)


def test_missing_input_cannot_enable_projection(tmp_path):
  with pytest.raises(ValueError, match='requires ransdata'):
    validate_reflection_input(tmp_path, 6, 3, np.ones(4), 1.0)


@pytest.mark.parametrize('nvar', [4, 5])
def test_reflection_cli_lifts_and_preserves_outputs(tmp_path, nvar):
  _input(tmp_path, nvar)
  A, B, _ = _pencil(nvar)
  sp.save_npz(tmp_path / 'S.npz', A)
  sp.save_npz(tmp_path / 'T.npz', B)
  model = 'laminar' if nvar == 4 else 'sa'
  (tmp_path / 'input').mkdir()
  (tmp_path / 'input/parameters.json').write_text(
    json.dumps(
      {
        'nt': 6,
        'nr': 3,
        'D_m': 1,
        'U_m_s': 1,
        'rho_kg_m3': 1,
        'T_K': 300,
        'mu_Pa_s': 1e-5,
        'model': model,
      }
    )
  )
  (tmp_path / 'assembly.json').write_text(json.dumps({'model': model}))
  script = Path(__file__).resolve().parents[1] / 'crouch/eigmain.py'
  result = subprocess.run(
    [
      sys.executable,
      str(script),
      str(tmp_path),
      '--wake-symmetry',
      '--ordering',
      'MMD_AT_PLUS_A',
      '--sigma-real',
      '-.2',
      '--sigma-imag',
      '.1',
      '--k',
      '2',
      '--label',
      'paritycheck',
    ],
    check=False,
    capture_output=True,
    text=True,
    timeout=30,
    env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'},
  )
  assert result.returncode == 0, result.stdout + result.stderr
  data = np.load(tmp_path / 'paritycheck_modes.npz')
  report = json.loads((tmp_path / 'paritycheck_solve.json').read_text())
  assert data['modes'].shape == (18 * nvar, 2)
  assert report['wake_symmetry'] and report['solve_dimension'] == 9 * nvar
  assert report['full_dimension'] == 18 * nvar
  if peak_rss_bytes() is None:
    assert report['peak_rss_bytes'] is None
  else:
    assert report['peak_rss_bytes'] > 0
  assert report['max_residual'] < 1e-6 and report['boundary_error'] < 1e-6
  assert report['ordering'] == 'MMD_AT_PLUS_A'
  np.testing.assert_array_equal(data['modes'], _wake_reflect(data['modes'], 6, 3, nvar))
  physical = np.tile(data['scales'], 18)[:, None] * data['modes']
  np.testing.assert_allclose(A @ physical, (B @ physical) * data['eigenvalues'], atol=1e-9)
  assert (tmp_path / 'paritycheck_eigenvalues.csv').is_file()


def test_import_and_rss_without_resource_module(monkeypatch):
  import builtins
  import importlib.util

  import eigmain

  original_import = builtins.__import__

  def without_resource(name, *args, **kwargs):
    if name == 'resource':
      raise ImportError('resource is unavailable on native Windows')
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, '__import__', without_resource)
  spec = importlib.util.spec_from_file_location('eigmain_without_resource', eigmain.__file__)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  assert module.peak_rss_bytes() is None
  assert json.dumps({'peak_rss_bytes': module.peak_rss_bytes()}) == '{"peak_rss_bytes": null}'


@pytest.mark.parametrize('platform,multiplier', [('linux', 1024), ('darwin', 1)])
def test_rss_metadata_keeps_platform_units(monkeypatch, platform, multiplier):
  from types import SimpleNamespace

  import eigmain

  monkeypatch.setattr(
    eigmain,
    'resource',
    SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=123456)),
  )
  monkeypatch.setattr(eigmain.sys, 'platform', platform)
  assert eigmain.peak_rss_bytes() == 123456 * multiplier
