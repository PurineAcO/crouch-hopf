import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from import_cylinder import convert


@pytest.mark.parametrize('failure', ['unconverged', 'model', 'nu_tilde'])
def test_import_rejects_unusable_baseflow_before_writing(tmp_path, failure):
  source = tmp_path / 'source'
  source.mkdir()
  parameters = {'model': 'laminar', 'converged': failure != 'unconverged', 'nt': 1, 'nr': 1}
  (source / 'parameters.json').write_text(json.dumps(parameters))
  model = 'sa' if failure == 'model' else 'laminar'
  nu = 1 if failure == 'nu_tilde' else 0
  np.savetxt(
    source / 'baseflow.dat',
    [[1, 0, 1, 1, 0, 300, 100, 0.2, nu]],
    header=f'TITLE="step 200 model={model}"\nVARIABLES',
    comments='',
  )
  target = tmp_path / 'out'
  with pytest.raises(ValueError):
    convert(source, target)
  assert not target.exists()


def test_import_rejects_unidentified_sa_equation(tmp_path):
  source = tmp_path / 'source'
  source.mkdir()
  (source / 'parameters.json').write_text(json.dumps({'model': 'sa', 'converged': True}))
  with pytest.raises(ValueError, match='sa_formulation=crouch-2007'):
    convert(source, tmp_path / 'out')
  assert not (tmp_path / 'out').exists()
