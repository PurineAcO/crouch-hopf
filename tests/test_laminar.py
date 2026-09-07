import numpy as np
from linearization import viscosity, viscous_coefficients


def test_laminar_viscous_flux_has_no_sa_coupling(monkeypatch):
  import classconfig as cc
  from models import FlowModel

  monkeypatch.setattr(cc, 'flow_model', FlowModel.LAMINAR)
  q = np.array([1.6e-5, 40, 3, 300, 0])
  g = np.zeros((5, 2))
  g[1:4] = [[2, 3], [4, 5], [6, 7]]
  _mu, mut = viscosity(q)
  assert mut == 0
  jq, jg = viscous_coefficients(q, g, [1.0, 0.0])
  np.testing.assert_allclose(jq[:4, 4], 0, atol=1e-50)
  np.testing.assert_array_equal(jg[:4, 4], 0)
  np.testing.assert_array_equal(jq[:4, 3], 0)
