"""在真实装配矩阵上核对边界结构：物面守恒、远场代数、改动范围。

四项检查：

1. ``T`` 的零行必须**恰好**是远场环（未用 ``--far-riemann`` 时）；物面环与其他环一样带时间
   导数。这是"物面条件通过虚单元镜像作用在面上、物面环仍是守恒律"的直接证据。
2. 远场入流行是面值算子（权重之和为 1）乘面处不变量行，整行和应等于面自身不变量行的分量和。
3. 与参考装配对照：除物面环外所有行必须逐位一致，说明本步改动只落在物面环。
4. 报告物面环/内部环/远场环的行范数区间，供记录。

用法::

  uv run python tools/verify_boundary_rows.py <case> [--reference <另一装配目录>]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'crouch'))

import boundary
import classconfig as cc
import readrans


def flat_row(s, n, component):
  return ((n - 1) * cc.S_MAX + (s - 1)) * 5 + component


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('case', type=Path)
  parser.add_argument('--reference', type=Path, default=None)
  args = parser.parse_args()
  root = args.case.resolve()
  readrans.read_rans(str(root / 'ransdata.txt'), str(root / 'edge.txt'))
  S = sp.csr_matrix(sp.load_npz(root / 'S.npz'))
  T = sp.csr_matrix(sp.load_npz(root / 'T.npz'))
  if S.shape[0] != cc.S_MAX * cc.N_MAX * 5:
    raise SystemExit(f'{S.shape} 与网格 {cc.S_MAX}x{cc.N_MAX} 不一致')
  print(f'{cc.S_MAX} x {cc.N_MAX}, nnz={S.nnz}')

  # ——— 1. T 的零行 ———
  zero_rows = np.flatnonzero(np.asarray(abs(T).sum(axis=1)).ravel() == 0)
  expected = np.array([flat_row(s, cc.N_MAX, k) for s in range(1, cc.S_MAX + 1) for k in range(5)])
  print(f'--- T 零行 {len(zero_rows)} 个（期望 {len(expected)}，即远场环）---')
  if not np.array_equal(np.sort(zero_rows), expected):
    raise SystemExit('T 零行不是恰好远场环')

  # ——— 2. 远场入流行的整行和 ———
  worst_far, counted = 0.0, 0
  for s in range(1, cc.S_MAX + 1):
    face = cc.goto_HALOcell((s, cc.N_MAX)).north
    rows, speeds = boundary.characteristics(face, np.asarray(face.jacobian[0]))
    incoming = speeds <= 0
    if not incoming.any():
      continue
    counted += 1
    for family in np.flatnonzero(incoming):
      total = S[flat_row(s, cc.N_MAX, int(family))].sum()
      worst_far = max(worst_far, abs(total - rows[family].sum()))
  print(
    f'--- 远场入流行整行和 vs 面处不变量行分量和：{counted} 个环有入流，'
    f'max|差| = {worst_far:.3e} ---'
  )
  if worst_far > 1e-9:
    raise SystemExit('远场入流族检查失败')

  # ——— 3. 与参考装配对照，改动只应落在物面环 ———
  if args.reference is not None:
    reference = sp.csr_matrix(sp.load_npz(args.reference.resolve() / 'S.npz'))
    if reference.shape != S.shape:
      raise SystemExit(f'参考装配形状 {reference.shape} 不一致')
    changed = np.flatnonzero(np.asarray(abs(S - reference).sum(axis=1)).ravel() != 0)
    wall = np.array([flat_row(s, 1, k) for s in range(1, cc.S_MAX + 1) for k in range(5)])
    print(f'--- 与参考装配的差异行：{len(changed)} 个（期望 {len(wall)}，即物面环）---')
    if not np.array_equal(np.sort(changed), wall):
      raise SystemExit('差异不止物面环')

  # ——— 4. 行范数记录 ———
  magnitude = np.asarray(abs(S).sum(axis=1)).ravel()
  blocks = (
    ('物面环', magnitude[0 : cc.S_MAX * 5]),
    ('内部环', magnitude[cc.S_MAX * 5 : (cc.N_MAX - 1) * cc.S_MAX * 5]),
    ('远场环', magnitude[(cc.N_MAX - 1) * cc.S_MAX * 5 :]),
  )
  for label, block in blocks:
    print(
      f'--- {label} 行绝对和：min={block.min():.3e} 中位={np.median(block):.3e} '
      f'max={block.max():.3e} ---'
    )
  print('全部检查通过')


if __name__ == '__main__':
  main()
