import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crouch'))
from eigmain import reduce_boundary
from linearization import jacobians, viscosity, viscous_flux


def test_four_variable_boundary_elimination():
  s = np.eye(12)
  s[:4, 4:8] = -np.eye(4)
  s[8:, 4:8] = -np.eye(4)
  s[4:8, 4:8] = -2 * np.eye(4)
  t = sp.diags([0] * 4 + [1] * 4 + [0] * 4)
  reduced, prolongation, _, _, error = reduce_boundary(
    sp.csr_matrix(s), t, 1, 3, np.array([1e-5, 70, 70, 300]), 1.0
  )
  assert reduced.shape == (4, 4)
  np.testing.assert_allclose(reduced.toarray(), -2 * np.eye(4))
  np.testing.assert_allclose(prolongation.toarray(), np.vstack([np.eye(4)] * 3))
  assert error == 0


def test_laminar_viscous_flux_has_no_sa_coupling(monkeypatch):
  import classconfig as cc
  from models import FlowModel

  monkeypatch.setattr(cc, 'flow_model', FlowModel.LAMINAR)
  q = np.array([1.6e-5, 40, 3, 300, 0])
  g = np.zeros((5, 2))
  g[1:4] = [[2, 3], [4, 5], [6, 7]]
  mu, mut = viscosity(q)
  assert mut == 0
  jq, jg = jacobians(lambda q, g: viscous_flux(q, g, [1.0, 0.0], molecular_mu=mu), q, g)
  np.testing.assert_allclose(jq[:4, 4], 0, atol=1e-50)
  np.testing.assert_array_equal(jg[:4, 4], 0)
  np.testing.assert_array_equal(jq[:4, 3], 0)


def test_wake_sector_is_orthonormal_and_has_expected_parity():
  from eigmain import wake_sector

  basis = wake_sector(8, 3)
  np.testing.assert_allclose((basis.T @ basis).toarray(), np.eye(48), atol=1e-15)
  q = (basis @ np.random.default_rng(21).normal(size=48)).reshape(3, 8, 4)
  np.testing.assert_allclose(q[:, ::-1], q * [-1, -1, 1, -1], atol=1e-15)
