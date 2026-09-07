import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
from eigmain import scale_pencil, solve_generalized


def coupled_pencil(size):
  """Both boundary rings couple around the circumference and to each other.

  The constraints imply wall=interior=far. Interior dynamics then have rates.
  Neither boundary cell can be eliminated independently.
  """
  eye = sp.eye(size, format='csr')
  shift = sp.csr_matrix(
    (np.ones(size), (np.arange(size), (np.arange(size) + 1) % size)), shape=(size, size)
  )
  C = 2 * eye + 0.25 * shift
  rates = np.linspace(-3, -1, size)
  S = sp.bmat(
    [
      [C, -C - 0.1 * eye, 0.1 * eye],
      [0.2 * eye, sp.diags(rates - 0.5), 0.3 * eye],
      [0.1 * eye, -C - 0.1 * eye, C],
    ],
    format='csr',
  )
  T = sp.diags(np.r_[np.zeros(size), np.ones(size), np.zeros(size)], format='csr')
  return S, T, rates


@pytest.mark.parametrize('scales', [[1], [1e-5, 70, 70, 300], [1e-5, 70, 70, 300, 1e-3]])
def test_full_coupled_pencil_known_rates(scales):
  size = 4 * len(scales)
  S, T, rates = coupled_pencil(size)
  A, B = scale_pencil(S, T, scales, 0.2)
  values, vectors, residuals, info = solve_generalized(A, B, -0.18 + 0.03j, 2)
  np.testing.assert_allclose(np.sort(values.real), 0.2 * rates[-2:], atol=2e-10)
  np.testing.assert_allclose(values.imag, 0, atol=2e-10)
  np.testing.assert_allclose(vectors[:size], vectors[size : 2 * size], atol=2e-10)
  np.testing.assert_allclose(vectors[2 * size :], vectors[size : 2 * size], atol=2e-10)
  assert residuals.max() < 1e-8
  assert info['boundary_error'] < 1e-10
  # Independent full dense generalized solve is a small-test oracle only.
  reference = la.eigvals(S.toarray(), T.toarray())
  reference = reference[np.isfinite(reference)]
  np.testing.assert_allclose(np.sort(reference.real), rates, atol=1e-12)


def test_complex_finite_eigenvalues_and_nonidentity_mass():
  S, T, _ = coupled_pencil(8)
  S = S.tolil()
  # Reduced first two interior variables form an oscillator with growth -0.2.
  S[8:10, 8:10] = [[-0.7, -2], [2, -0.7]]
  multiplier = sp.diags(np.linspace(0.2, 2, 24))
  S, T = multiplier @ S.tocsr(), multiplier @ T
  values, vectors, residuals, _ = solve_generalized(S, T, -0.1 + 1.9j, 1)
  np.testing.assert_allclose(values, [-0.2 + 2j], atol=1e-11)
  assert residuals.max() < 1e-11
  np.testing.assert_allclose(S @ vectors, (T @ vectors) * values, atol=1e-11)


@pytest.mark.parametrize('time_scale', [0, -1, np.inf, np.nan])
def test_invalid_time_scale(time_scale):
  S, T, _ = coupled_pencil(4)
  with pytest.raises(ValueError, match='finite and positive'):
    scale_pencil(S, T, [1], time_scale)


@pytest.mark.parametrize('which', ['S', 'T'])
def test_nonfinite_constraint_rejected(which):
  S, T, _ = coupled_pencil(4)
  if which == 'S':
    S[0, 0] = np.nan
  else:
    T = T.tolil()
    T[0, 0] = np.inf
    T = T.tocsr()
  with pytest.raises(ValueError, match='finite'):
    solve_generalized(S, T, 0.1, 2)
  with pytest.raises(ValueError, match='finite'):
    scale_pencil(S, T, [1], 1)


@pytest.mark.parametrize('k', [0, 11, 12])
def test_invalid_arpack_size(k):
  S, T, _ = coupled_pencil(4)
  with pytest.raises(ValueError, match='ARPACK'):
    solve_generalized(S, T, 0.1, k)


@pytest.mark.parametrize('model,nvar', [('laminar', 4), ('sa', 5)])
def test_cli_full_modes_and_units(tmp_path, model, nvar):
  ns = 4
  size = ns * nvar
  S, T, rates = coupled_pencil(size)
  sp.save_npz(tmp_path / 'S.npz', S)
  sp.save_npz(tmp_path / 'T.npz', T)
  (tmp_path / 'input').mkdir()
  parameters = {
    'model': model,
    'nt': ns,
    'nr': 3,
    'rho_kg_m3': 1e-5,
    'U_m_s': 20,
    'D_m': 10,
    'T_K': 300,
    'mu_Pa_s': 1e-5,
  }
  (tmp_path / 'input/parameters.json').write_text(json.dumps(parameters))
  (tmp_path / 'assembly.json').write_text(json.dumps({'model': model}))
  script = Path(__file__).resolve().parents[1] / 'crouch/eigmain.py'
  result = subprocess.run(
    [
      sys.executable,
      str(script),
      str(tmp_path),
      '--k',
      '2',
      '--sigma-real',
      '-0.45',
      '--sigma-imag',
      '0.01',
      '--label',
      'full',
    ],
    check=False,
    capture_output=True,
    text=True,
    timeout=30,
    env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'},
  )
  assert result.returncode == 0, result.stdout + result.stderr
  data = np.load(tmp_path / 'full_modes.npz')
  np.testing.assert_allclose(np.sort(data['eigenvalues'].real), rates[-2:] * 0.5, atol=1e-10)
  assert data['modes'].shape == (3 * size, 2)
  physical = np.tile(data['scales'], ns * 3)[:, None] * data['modes']
  np.testing.assert_allclose(S @ physical, (T @ physical) * data['eigenvalues'] / 0.5, atol=1e-9)
  table = np.loadtxt(tmp_path / 'full_eigenvalues.csv', delimiter=',', skiprows=1)
  np.testing.assert_allclose(table[:, 4], table[:, 1] / 0.5)
  np.testing.assert_allclose(table[:, 5], table[:, 2] / np.pi)
  report = json.loads((tmp_path / 'full_solve.json').read_text())
  assert report['model'] == model and report['nvar'] == nvar
  assert report['boundary_error'] < 1e-10 and report['max_residual'] < 1e-8


def test_bad_algebraic_mode_cannot_hide_behind_large_interior_rows(monkeypatch):
  import eigmain

  A = sp.diags([1, 1e12, 2e12, 3e12], format='csc')
  B = sp.diags([0, 1e12, 1e12, 1e12], format='csc')
  # Differential row is exact for lambda=1; algebraic x0=0 is violated.
  vector = np.array([[1], [1], [0], [0]], dtype=complex)
  monkeypatch.setattr(eigmain.spla, 'eigs', lambda *a, **kw: (np.array([1.0]), vector))
  with pytest.raises(ValueError, match='Algebraic constraint residual'):
    solve_generalized(A, B, 0, 1)
