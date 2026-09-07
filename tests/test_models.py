import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crouch'))
import classconfig as cc
import linearization as lin
from eigmain import wake_sector
from models import FlowModel, validate_base_model


def test_sa_viscosity_response_survives_model_switch(monkeypatch):
  q = np.array([1.2, 40.0, 3.0, 300.0, 2e-4])
  g = np.ones((5, 2))
  monkeypatch.setattr(cc, 'flow_model', FlowModel.SA)
  mu, mut = lin.viscosity(q)
  assert mut > 0
  sa_flux = lin.viscous_flux(q, g, [1, 0], molecular_mu=mu)
  jq, _ = lin.jacobians(lambda q, g: lin.viscous_flux(q, g, [1, 0], molecular_mu=mu), q, g)
  assert np.linalg.norm(jq[:4, 4]) > 0
  monkeypatch.setattr(cc, 'flow_model', FlowModel.LAMINAR)
  assert lin.viscosity(q)[1] == 0
  assert not np.allclose(lin.viscous_flux(q, g, [1, 0], molecular_mu=mu), sa_flux)
  monkeypatch.setattr(cc, 'flow_model', FlowModel.SA)
  np.testing.assert_array_equal(lin.viscous_flux(q, g, [1, 0], molecular_mu=mu), sa_flux)


@pytest.mark.parametrize('model', list(FlowModel))
def test_recorded_base_model_must_match(model):
  other = 'sa' if model is FlowModel.LAMINAR else 'laminar'
  with pytest.raises(ValueError, match='does not match'):
    validate_base_model(model, {'model': other}, np.zeros(3))
  validate_base_model(model, {'model': model.value}, np.zeros(3))


@pytest.mark.parametrize('values', [[1e-20], [np.nan], [-1]])
def test_laminar_rejects_nonzero_or_invalid_sa_input(values):
  with pytest.raises(ValueError):
    validate_base_model(FlowModel.LAMINAR, {'model': 'laminar'}, np.array(values))


def test_sa_reflection_sector():
  basis = wake_sector(8, 3, 5)
  np.testing.assert_allclose((basis.T @ basis).toarray(), np.eye(60), atol=1e-15)
  q = (basis @ np.random.default_rng(21).normal(size=60)).reshape(3, 8, 5)
  np.testing.assert_allclose(q[:, ::-1], q * [-1, -1, 1, -1, -1], atol=1e-15)


@pytest.mark.parametrize('model', list(FlowModel))
def test_eigen_cli_uses_assembled_model(tmp_path, model):
  import json
  import os
  import subprocess

  import scipy.sparse as sp

  ns, nn, nvar = 8, 3, model.nvar
  ring = ns * nvar
  # Independent scalar decay rates, with boundary values tied to the interior.
  # This exercises the actual CLI, dimensional scaling, and boundary restoration.
  rates = np.linspace(-2, -1, ring)
  s = sp.bmat(
    [
      [sp.eye(ring), -sp.eye(ring), None],
      [None, sp.diags(rates), None],
      [None, -sp.eye(ring), sp.eye(ring)],
    ],
    format='csr',
  )
  t = sp.diags([0] * ring + [1] * ring + [0] * ring)
  sp.save_npz(tmp_path / 'S.npz', s)
  sp.save_npz(tmp_path / 'T.npz', t)
  (tmp_path / 'assembly.json').write_text(json.dumps({'model': model.value}))
  (tmp_path / 'input').mkdir()
  parameters = {
    'nt': ns,
    'nr': nn,
    'D_m': 1,
    'U_m_s': 1,
    'rho_kg_m3': 1,
    'T_K': 300,
    'mu_Pa_s': 1e-5,
    'model': model.value,
  }
  (tmp_path / 'input/parameters.json').write_text(json.dumps(parameters))
  script = Path(__file__).resolve().parents[1] / 'crouch/eigmain.py'
  result = subprocess.run(
    [
      sys.executable,
      str(script),
      str(tmp_path),
      '--k',
      '2',
      '--sigma-real',
      '-0.9',
      '--sigma-imag',
      '0',
    ],
    check=False,
    capture_output=True,
    text=True,
    timeout=30,
    env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'},
  )
  assert result.returncode == 0, result.stderr
  data = np.load(tmp_path / 'wake_modes.npz')
  np.testing.assert_allclose(np.sort(data['eigenvalues'].real), rates[-2:], atol=1e-12)
  assert data['modes'].shape == (ns * nn * nvar, 2)
  report = json.loads((tmp_path / 'wake_solve.json').read_text())
  assert report['nvar'] == nvar and report['model'] == model.value
  assert report['max_residual'] < 1e-12


def test_missing_model_metadata_is_rejected():
  with pytest.raises(ValueError, match='must record model'):
    validate_base_model(FlowModel.LAMINAR, {}, np.zeros(3))


def test_operator_requires_explicit_model(monkeypatch):
  monkeypatch.setattr(cc, 'flow_model', None)
  with pytest.raises(ValueError, match='Select a flow model'):
    lin.viscosity(np.array([1, 40, 0, 300, 0]))
