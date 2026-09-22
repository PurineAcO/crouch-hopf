"""Boundary conditions on the boundary faces, in the original thirteen five-variable blocks.

条件都作用在**边界面**上，而不是首/末层单元中心：

* 物面（``ring.south``）：û=v̂=ν̃̂=0、∂ρ̂/∂n=∂T̂/∂n=0，对应论文 (2.3.4)。它不用代数行，
  而是由虚单元镜像 ``formmat._WALL_MAP`` 直接写成**面值**关系（面平均值为零、法向差分为零），
  所以物面环与其他环一样由守恒律控制；
* 远场（``ring.north``）：论文 (2.3.5) 的线性化不变量，入流特征取零面值、出流特征取
  零面法向导数，对应论文 (2.3.6)/(2.3.7)。它无法写成单胞虚单元映射，因此仍是代数约束行。

远场环的代数行不带时间导数；物面条件只通过虚单元值/梯度进入相邻单元的平衡。
"""

import classconfig as cc
import numpy as np


def _unit_normal(normal):
  normal = np.asarray(normal, dtype=float)
  length = np.linalg.norm(normal)
  return normal / length


def _quadratic_weights(points, origin, normal, derivative, preferred_stencil=None):
  """在 origin（边界面上一点）处取值或取法向导数的单侧二次权重。

  SVD whitening removes both stretching and shear before fitting
  [1, z0, z1, z0**2, z0*z1, z1**2]. A preferred three-point radial stencil
  preserves the normal-aligned limit. Its weights are always corrected against
  all 2D moments, so projected distances alone never define a skew derivative.
  No lower-order fallback is permitted.

  ``points`` 全是参考单元中心；``origin`` 是条件作用点，它本身不是未知量，
  因此返回的权重只落在 ``points`` 上。``derivative=True`` 给出 n·grad，
  ``derivative=False`` 给出面值。
  """
  points = np.asarray(points, dtype=float)
  origin = np.asarray(origin, dtype=float)
  normal = _unit_normal(normal)
  if points.ndim != 2 or points.shape[1] != 2 or len(points) < 6:
    raise ValueError('A 2D quadratic boundary stencil needs at least six points')
  if origin.shape != (2,) or not np.isfinite(origin).all():
    raise ValueError('Boundary condition point must be a finite 2D point')
  if not np.isfinite(points).all():
    raise ValueError('Boundary points must be finite')
  delta = points - origin
  _, singular, vt = np.linalg.svd(delta, full_matrices=False)
  if singular[-1] <= np.finfo(float).eps * max(delta.shape) * singular[0]:
    raise ValueError('Rank-deficient boundary geometry')
  transform = vt.T / singular
  x, y = (delta @ transform).T
  design = np.column_stack([np.ones(len(points)), x, y, x * x, x * y, y * y])
  # 目标泛函在 origin 处取值：面值算子取 1，法向导数算子取 n·grad。
  target = np.r_[0, normal @ transform, 0, 0, 0] if derivative else np.r_[1, 0, 0, 0, 0, 0]
  prior = np.zeros(len(points))
  if preferred_stencil is not None:
    slots = np.asarray(preferred_stencil, dtype=int)
    if slots.shape != (3,) or len(set(slots)) != 3 or np.any((slots < 0) | (slots >= len(points))):
      raise ValueError('Preferred boundary stencil must contain three distinct point indices')
    coordinates = (delta @ normal)[slots]
    scale = np.max(abs(coordinates))
    if scale > 0:
      z = coordinates / scale
      radial = np.array([np.ones(3), z, z * z])
      # An ill-conditioned prior is unnecessary: the full 2D constraints still
      # determine a valid stencil, including when radial projections coincide.
      if np.linalg.cond(radial) < 1e6:
        rhs = np.array([0, 1 / scale, 0]) if derivative else np.array([1.0, 0.0, 0.0])
        prior[slots] = np.linalg.solve(radial, rhs)
  correction, _, rank, _ = np.linalg.lstsq(design.T, target - design.T @ prior, rcond=None)
  weights = prior + correction
  if rank != 6:
    raise ValueError('Rank-deficient quadratic boundary stencil')
  if not np.isfinite(weights).all():
    raise ValueError('Nonfinite boundary weights')
  return weights


def normal_derivative(points, normal, preferred_stencil=None, origin=None):
  """法向 n·grad 的二次精确权重；``origin=None`` 时作用在 ``points[0]`` 上。"""
  points = np.asarray(points, dtype=float)
  if origin is None:
    origin = points[0]
  return _quadratic_weights(points, origin, normal, True, preferred_stencil)


def face_value(points, origin, normal, preferred_stencil=None):
  """在 ``origin``（边界面中点）处重建面值的二次精确权重。"""
  return _quadratic_weights(points, origin, normal, False, preferred_stencil)


def _neighbor(cell, side):
  return getattr(getattr(cell, side), side)


def _stencil(cell, side):
  """边界条件的单侧单元模板：``(names, cells, face)``。

  ``side`` 指向域内（物面用 ``north``、远场用 ``south``），模板全部落在内侧；
  ``face`` 是条件作用的那个边界面。
  """
  first = _neighbor(cell, side)
  cells = [
    cell,
    _neighbor(cell, 'east'),
    _neighbor(cell, 'west'),
    first,
    _neighbor(first, side),
    _neighbor(first, 'east'),
    _neighbor(first, 'west'),
  ]
  radial = 'n' if side == 'north' else 's'
  names = ['c', 'e', 'w', radial, radial * 2, radial + 'e', radial + 'w']
  face = cell.south if side == 'north' else cell.north
  return names, cells, face


def _face_operators(cells, face):
  """面值算子、面法向导数算子与面法向；两者的作用点都是边界面中点。"""
  points = [[c.x, c.y] for c in cells]
  normal = _unit_normal(face.jacobian[0])
  value = face_value(points, face.mid, normal, preferred_stencil=[0, 3, 4])
  gradient = normal_derivative(points, normal, preferred_stencil=[0, 3, 4], origin=face.mid)
  return value, gradient, normal


def _add_blocks(cell, names, blocks):
  if not np.isfinite(blocks).all():
    raise ValueError(f'Nonfinite boundary constraints at {cell.index}')
  for name, block in zip(names, blocks):
    cell.form_influence(cc.dic[name], block)


def characteristics(state, normal):
  """Paper invariant derivatives, ordered plus, minus, tangent, entropy, SA.

  I+/- = u_n +/- 2*sqrt(gamma*R*T)/(gamma-1), E = R*T/rho**(gamma-1).
  Their acoustic rows differ from Euler left eigenvectors by
  +/- rho**(gamma-1)/((gamma-1)*a) * dE when entropy is perturbed.
  Speeds assign the paper's incoming/outgoing families under an outward normal;
  these rows are not a decoupled Euler characteristic basis for arbitrary dE.

  ``state`` 可以是单元，也可以是面（只读取 rho/T/u/v），因此远场条件可以用
  面自己的基流状态而不是环单元的状态来定系数与入/出流归属。
  """
  nx, ny = _unit_normal(normal)
  if not np.isfinite([state.rho, state.T, state.u, state.v]).all() or min(state.rho, state.T) <= 0:
    raise ValueError('Characteristic base state must be finite with positive rho and T')
  a = np.sqrt(cc.gamma * cc.R * state.T)
  temperature = np.array(
    [0, 0, 0, np.sqrt(cc.gamma * cc.R) / ((cc.gamma - 1) * np.sqrt(state.T)), 0]
  )
  velocity = np.array([0, nx, ny, 0, 0])
  entropy = [
    -(cc.gamma - 1) * cc.R * state.T / state.rho**cc.gamma,
    0,
    0,
    cc.R / state.rho ** (cc.gamma - 1),
    0,
  ]
  rows = np.array(
    [velocity + temperature, velocity - temperature, [0, -ny, nx, 0, 0], entropy, [0, 0, 0, 0, 1]]
  )
  un = nx * state.u + ny * state.v
  return rows, np.array([un + a, un - a, un, un, un])


def far_boundary(cell):
  """远场条件作用在远场外边界面上：入流特征取零面值，出流特征取零面法向导数。

  面法向由 ``edge.txt`` 给出并指向域外（``(c-mid)·n_hat < 0``），因此
  ``speeds <= 0`` 恰好是“携带外部信息进入计算域”的那几族。
  """
  if cell.index[1] != cc.N_MAX:
    raise ValueError('Not a farfield ring cell')
  names, cells, face = _stencil(cell, 'south')
  value, gradient, normal = _face_operators(cells, face)
  boundary, speeds = characteristics(face, normal)
  # Zero-speed modes are prescribed, including exactly sonic acoustic modes.
  incoming = speeds <= 0
  blocks = np.zeros((len(names), 5, 5))
  for i, neighbor in enumerate(cells):
    coefficients, _ = characteristics(neighbor, normal)
    # 出流族：∂Î/∂n = 0，用面法向导数算子作用在各单元自身的系数上。
    blocks[i, ~incoming] = gradient[i] * coefficients[~incoming]
    # 入流族：Î = 0，用面值算子把面处的不变量表达成内侧单元的组合。
    blocks[i, incoming] = value[i] * boundary[incoming]
  _add_blocks(cell, names, blocks)
