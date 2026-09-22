import classconfig as cc
import numpy as np


def green_gauss_face_vari(face: cc.face_class):
  """计算某个`face`的梯度影响矩阵,根据green-gauss方法,会和8个网格挂钩\n
  为了看起来舒服,会返回一个字典."""

  if face.direction == 'WE':
    grad_dic = {}
    east = face.east
    west = face.west
    grad_dic['w'] = (
      west.north.jacobian[0]
      - west.south.jacobian[0]
      + west.east.jacobian[0]
      - west.west.jacobian[0]
    ) / west.vol / 4 - east.west.jacobian[0] / east.vol / 4
    grad_dic['e'] = (
      east.north.jacobian[0]
      - east.south.jacobian[0]
      + east.east.jacobian[0]
      - east.west.jacobian[0]
    ) / east.vol / 4 + west.east.jacobian[0] / west.vol / 4
    grad_dic['nw'] = (west.north.jacobian[0]) / west.vol / 4
    grad_dic['ne'] = (east.north.jacobian[0]) / east.vol / 4
    grad_dic['sw'] = -(west.south.jacobian[0]) / west.vol / 4
    grad_dic['se'] = -(east.south.jacobian[0]) / east.vol / 4
    grad_dic['ee'] = (east.east.jacobian[0]) / east.vol / 4
    grad_dic['ww'] = -(west.west.jacobian[0]) / west.vol / 4
    return grad_dic

  elif face.direction == 'NS':
    grad_dic = {}
    north = face.north
    south = face.south
    grad_dic['n'] = (
      north.north.jacobian[0]
      - north.south.jacobian[0]
      + north.east.jacobian[0]
      - north.west.jacobian[0]
    ) / north.vol / 4 + south.north.jacobian[0] / south.vol / 4
    grad_dic['s'] = (
      south.north.jacobian[0]
      - south.south.jacobian[0]
      + south.east.jacobian[0]
      - south.west.jacobian[0]
    ) / south.vol / 4 - north.south.jacobian[0] / north.vol / 4
    grad_dic['ne'] = (north.east.jacobian[0]) / north.vol / 4
    grad_dic['se'] = (south.east.jacobian[0]) / south.vol / 4
    grad_dic['nw'] = -(north.west.jacobian[0]) / north.vol / 4
    grad_dic['sw'] = -(south.west.jacobian[0]) / south.vol / 4
    grad_dic['nn'] = (north.north.jacobian[0]) / north.vol / 4
    grad_dic['ss'] = -(south.south.jacobian[0]) / south.vol / 4
    return grad_dic

  else:
    raise ValueError("face.direction must be 'WE' or 'NS'")


def green_gauss_cell_vari(cell: cc.cell_class):
  """计算某个`cell`的梯度影响矩阵,根据green-gauss方法,会和5个网格挂钩\n
  为了看起来舒服,会返回一个字典."""
  grad_dic = {}
  grad_dic['c'] = (
    (
      cell.north.jacobian[0]
      - cell.south.jacobian[0]
      + cell.east.jacobian[0]
      - cell.west.jacobian[0]
    )
    / cell.vol
    / 2
  )
  grad_dic['n'] = (cell.north.jacobian[0]) / cell.vol / 2
  grad_dic['s'] = -(cell.south.jacobian[0]) / cell.vol / 2
  grad_dic['e'] = (cell.east.jacobian[0]) / cell.vol / 2
  grad_dic['w'] = -(cell.west.jacobian[0]) / cell.vol / 2
  return grad_dic


def green_gauss_from_JST(
  cell: cc.cell_class,
  facenorth: cc.face_class,
  facesouth: cc.face_class,
  faceeast: cc.face_class,
  facewest: cc.face_class,
):
  """基于Green-Guass的梯度构建,一般按照北、南、东、西来写"""
  u_vec = np.array([facenorth.u, facesouth.u, faceeast.u, facewest.u])
  v_vec = np.array([facenorth.v, facesouth.v, faceeast.v, facewest.v])
  miubl_vec = np.array([facenorth.miubl, facesouth.miubl, faceeast.miubl, facewest.miubl])
  T_vec = np.array([facenorth.T, facesouth.T, faceeast.T, facewest.T])
  nx_vec = np.array([facenorth.nx, -facesouth.nx, faceeast.nx, -facewest.nx])
  ny_vec = np.array([facenorth.ny, -facesouth.ny, faceeast.ny, -facewest.ny])
  rho_vec = np.array([facenorth.rho, facesouth.rho, faceeast.rho, facewest.rho])
  cell.rhograd = np.array([np.dot(rho_vec, nx_vec), np.dot(rho_vec, ny_vec)]) / cell.vol
  cell.ugrad = np.array([np.dot(u_vec, nx_vec), np.dot(u_vec, ny_vec)]) / cell.vol
  cell.vgrad = np.array([np.dot(v_vec, nx_vec), np.dot(v_vec, ny_vec)]) / cell.vol
  cell.miublgrad = np.array([np.dot(miubl_vec, nx_vec), np.dot(miubl_vec, ny_vec)]) / cell.vol
  cell.Tgrad = np.array([np.dot(T_vec, nx_vec), np.dot(T_vec, ny_vec)]) / cell.vol


GRADIENT_FIELDS = ('rhograd', 'ugrad', 'vgrad', 'Tgrad', 'miublgrad')


def mirror_ghost(cell: cc.cell_class, source: cc.cell_class, face: cc.face_class, value_map):
  """补齐虚单元的几何与基流梯度，使其与 ``formmat`` 的虚单元值映射一致。

  ``source`` 是该虚单元在 ``formmat._ghost_target`` 里折叠到的实单元：虚单元中心是
  ``source`` 中心关于边界面 ``face`` 中点的镜像，取值关系是 ``c(-n) = s_c c(n)``，
  ``s_c`` 取 ``value_map`` 的对角。对镜像场求导得到

    切向分量乘 ``s_c``，法向分量乘 ``-s_c``

  即 ``grad c(-n) = s_c * (切向分量 - 法向分量)``。物面用 ``_WALL_MAP``（u,v,ν̃ 取负）时，
  虚单元与实单元的面平均值给出 û=v̂=ν̃̂=0、ρ̂/T̂ 的两点面法向导数为零，而 u/v/ν̃ 的
  两点面法向导数保留——壁面摩擦与壁面热流就是靠它进入首层守恒律的。
  ``fill_ghost`` 只填了物理量、坐标留成 (0,0)，而两点面法向导数与边界面黏性系数
  需要这里的坐标。
  """
  normal = np.asarray(face.jacobian[0], dtype=float)
  normal = normal / np.linalg.norm(normal)
  tangent = np.array([-normal[1], normal[0]])
  midpoint = np.asarray(face.mid, dtype=float)
  cell.x, cell.y = midpoint - (np.array([source.x, source.y]) - midpoint)
  for attribute, sign in zip(GRADIENT_FIELDS, np.diag(value_map)):
    gradient = np.asarray(getattr(source, attribute), dtype=float)
    normal_part = float(gradient @ normal)
    tangential_part = float(gradient @ tangent)
    setattr(cell, attribute, sign * (tangential_part * tangent - normal_part * normal))
  return cell
