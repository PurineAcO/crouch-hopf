"""核对物面/远场面的几何朝向与面状态，供面处边界条件使用。

输出每一项的含义：

* ``n_hat``：``face.jacobian[0]`` 的单位化向量，即面法向；
* ``(c-mid).n_hat``：环心相对面中点的法向偏移。物面应当为**正**（法向指进流体），
  远场面应当为**负**（法向指向域外），否则特征入/出流的判据会整体反号；
* 面状态由 ``face_class.form_physics`` 给出（两侧算术平均）。

用法::

  uv run python tools/check_boundary_faces.py runs/<case>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'crouch'))

import classconfig as cc
import readrans


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('case', type=Path)
  args = parser.parse_args()
  root = args.case.resolve()
  readrans.read_rans(str(root / 'ransdata.txt'), str(root / 'edge.txt'))
  print(f'{cc.S_MAX} x {cc.N_MAX}, HALO={cc.HALO}')
  for label, index in (('物面环 n=1', 1), ('远场环 n=N_MAX', cc.N_MAX)):
    print(f'--- {label} ---')
    offsets, heights = [], []
    for s in (1, 40, 120, 200, 280):
      cell = cc.goto_HALOcell((s, index))
      face = cell.south if index == 1 else cell.north
      normal = np.asarray(face.jacobian[0], dtype=float)
      unit = normal / np.linalg.norm(normal)
      offset = np.array([cell.x - face.mid[0], cell.y - face.mid[1]])
      projection = float(offset @ unit)
      offsets.append(projection)
      heights.append(float(np.linalg.norm(offset)))
      print(
        f'  s={s:>3} mid=({face.mid[0]:+.5f}, {face.mid[1]:+.5f}) n_hat=({unit[0]:+.4f}, {unit[1]:+.4f}) '
        f'(c-mid).n_hat={projection:+.3e} |c-mid|={np.linalg.norm(offset):.3e} '
        f'face rho={face.rho:.5f} u={face.u:+.5f} v={face.v:+.5f} T={face.T:.3f} nu={face.miubl:.3e}'
      )
    sign = 1 if index == 1 else -1
    consistent = all(value * sign > 0 for value in offsets)
    print(f'  朝向一致: {consistent}；法向偏移 min={min(offsets):+.3e} max={max(offsets):+.3e}')
  print('--- 远场入射/出流分布（按面状态计算 un = u_n）---')
  counts = {'in': 0, 'out': 0}
  for s in range(1, cc.S_MAX + 1):
    cell = cc.goto_HALOcell((s, cc.N_MAX))
    face = cell.north
    unit = np.asarray(face.jacobian[0], dtype=float)
    unit = unit / np.linalg.norm(unit)
    un = unit @ np.array([face.u, face.v])
    counts['in' if un < 0 else 'out'] += 1
  print(f'  入射面 {counts["in"]} 个，出流面 {counts["out"]} 个（共 {cc.S_MAX}）')


if __name__ == '__main__':
  main()
