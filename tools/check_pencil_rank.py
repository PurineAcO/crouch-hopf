"""检查 (S, T) 铅笔是否存在"整行为零"的结构性奇异。

如果某一行在 S 和 T 中同时为零，则对任意 σ，该行的方程为 0 = 0，
铅笔就对所有 σ 都奇异——移位反演会返回一整簇位置由舍入误差决定的伪特征值，
而且这些特征矢量会挤在同一个结构上。这正好是我们观察到的现象。
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

VARIABLES = ('rho', 'u', 'v', 'T', 'nu~')


def main():
  case = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('runs/naca0012-ma076-re1e7-c5-a32-nsc160')
  norms, maximum = {}, 0.0
  for name in ('S', 'T'):
    matrix = sp.csr_matrix(sp.load_npz(case / f'{name}.npz'))
    absolute = abs(matrix)
    norms[name] = np.asarray(absolute.sum(axis=1)).ravel()
    maximum = max(maximum, norms[name].max(), np.abs(matrix.data).max())
    print(
      f'{name}: shape={matrix.shape} nnz={matrix.nnz} |data|max={np.abs(matrix.data).max():.3e} '
      f'行绝对和 min={norms[name].min():.3e} 中位={np.median(norms[name]):.3e} '
      f'max={norms[name].max():.3e}'
    )
  dimension = len(norms['S'])
  tolerance = 1e-12 * maximum
  zero_s = norms['S'] < tolerance
  zero_t = norms['T'] < tolerance
  both = zero_s & zero_t
  print(f'\n阈值 = {tolerance:.3e}')
  print(
    f'  S 整行为零: {int(zero_s.sum())} 行；T 整行为零: {int(zero_t.sum())} 行；'
    f'两者同时为零: {int(both.sum())} 行 / {dimension}'
  )
  for label, mask in (('S 零行', zero_s), ('T 零行', zero_t), ('S 与 T 都零行', both)):
    index = np.flatnonzero(mask)
    if not len(index):
      continue
    print(
      f'  {label}: 变量分布 {dict(Counter(VARIABLES[i % 5] for i in index))}，'
      f'涉及单元 {len({i // 5 for i in index})} 个，'
      f'行号前 5 {index[:5].tolist()}，是否连续 {bool(np.all(np.diff(index) == 1))}'
    )


if __name__ == '__main__':
  main()
