"""Model selection shared by assembly, constitutive laws and eigenvalue scaling."""

from enum import Enum

SA_FORMULATION = 'crouch-2007'


class FlowModel(str, Enum):
  LAMINAR = 'laminar'
  SA = 'sa'

  @property
  def nvar(self):
    return 4 if self is FlowModel.LAMINAR else 5


def validate_thermodynamics(parameters):
  import classconfig as cc
  import numpy as np

  recorded = parameters.get('thermodynamics', {})
  expected = {'R': cc.R, 'Cp': cc.cp, 'Cv': cc.cv, 'gamma': cc.gamma}
  for name, value in expected.items():
    supplied = recorded.get(name)
    if not isinstance(supplied, (int, float)) or not np.isclose(
      supplied, value, rtol=1e-12, atol=0
    ):
      raise ValueError(f'Base-flow thermodynamics must record matching {name}={value}')


def validate_base_model(model, parameters, nu_tilde):
  import numpy as np

  if 'model' not in parameters:
    raise ValueError('Base-flow parameters must record model')
  recorded = parameters['model']
  if FlowModel(recorded) is not model:
    raise ValueError(f'Base-flow model {recorded} does not match requested {model.value}')
  if model is FlowModel.SA and parameters.get('sa_formulation') != SA_FORMULATION:
    raise ValueError('SA base flow must record sa_formulation=crouch-2007')
  if not np.isfinite(nu_tilde).all() or np.any(nu_tilde < 0):
    raise ValueError('Base-flow nu-tilde must be finite and nonnegative')
  if model is FlowModel.LAMINAR and np.any(nu_tilde != 0):
    raise ValueError('Laminar input must have zero nu-tilde')
  validate_thermodynamics(parameters)
