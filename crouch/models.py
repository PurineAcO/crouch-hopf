"""Model selection shared by assembly, constitutive laws and eigenvalue scaling."""

from enum import Enum


class FlowModel(str, Enum):
  LAMINAR = 'laminar'
  SA = 'sa'

  @property
  def nvar(self):
    return 4 if self is FlowModel.LAMINAR else 5


def validate_base_model(model, parameters, nu_tilde):
  import numpy as np

  recorded = parameters.get('model')
  if recorded is not None and FlowModel(recorded) is not model:
    raise ValueError(f'Base-flow model {recorded} does not match requested {model.value}')
  if not np.isfinite(nu_tilde).all() or np.any(nu_tilde < 0):
    raise ValueError('Base-flow nu-tilde must be finite and nonnegative')
  if model is FlowModel.LAMINAR and np.any(nu_tilde != 0):
    raise ValueError('Laminar input must have zero nu-tilde')
