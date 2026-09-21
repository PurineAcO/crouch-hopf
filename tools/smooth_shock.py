"""论文 §3.2 的激波平滑：把定常场沿网格线平滑后，在激波附近与原场混合。

用法（本机 PowerShell）：

```powershell
uv run python tools/smooth_shock.py runs/<case> runs/<case>-nsc160 --cycles 160
```

论文 (3.2.2)：沿主流动方向的网格线做 N_SC 个循环的平滑，

    q(i,j) <- q(i,j) + 0.5*c_i*[q(i+1,j) - 2q(i,j) + q(i-1,j)]

全文取 c_i = 0.1（每循环等效扩散 D = 0.5*c_i = 0.05）。论文 (3.2.1) 再用只在激波附近
非零的混合函数 S 混合：q_final = (1-S)*q_original + S*q_smooth。

论文没有给出 S 的解析形式，本脚本的取法（都可调，运行时会打印诊断量）：

1. 在**原场**上算环向压力梯度 |dp/ds|，用弧长归一；
2. 只在 ``--window`` 给出的弦向窗口内、并按上下表面分别取该径向线上的最大值作为激波位置；
   前缘吸力峰的 |dp/ds| 往往比激波还大，不加窗口会把激波误判到前缘；
3. 该最大值低于 ``--threshold`` 倍（窄带内）全场最大值时，认为这条线上该表面没有激波，
   S = 0。默认 0.05：边界层把激波抹平后，近壁行的压力梯度只有超声速区激波的约 12%，
   阈值取 0.2 会把激波脚整段排除掉；
4. S(n,s) = exp(-ds²/(2σ_s²))，ds 为该行内沿环向的弧长距离（按周长取短边），
   σ_s = ``--sigma-cells`` 个激波处的当地步长；多条激波取各高斯的最大值。

只读原算例目录：输出写到新目录（复制 mesh.txt 与 parameters.json，重写 baseflow.dat），
原目录与基流求解器都不动。平滑量与论文一致，为 ρ、u、v、T、ν̃；p 与 Ma 由 ρRT 与
sqrt(gamma*R*T) 重算以保证自洽。更新是凸组合，ρ、T、p 的正性与 ν̃ 的非负性保持。
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

GAMMA, R = 1.4, 287.05
SMOOTHED = [2, 3, 4, 5, 8]  # rho u v T nu_tilde


def arc_lengths(xy):
  """相邻单元的环向步长 step[n,s] = |X[n,s+1]-X[n,s]|、累计弧长与周长。"""
  step = np.linalg.norm(np.roll(xy, -1, axis=1) - xy, axis=2)
  cumulative = np.cumsum(step, axis=1) - step[:, :1]
  return step, cumulative, cumulative[:, -1] + step[:, -1]


def ring_gap(cumulative, circumference, centre):
  """同一行内各点与 centre 之间的环向弧长，取短边。"""
  direct = np.abs(cumulative - cumulative[centre])
  return np.minimum(direct, circumference - direct)


def smoothed_field(q, cycles, coefficient, periodic):
  """(3.2.2)：沿 s 方向做 cycles 个循环的显式扩散（同时更新，非 Gauss-Seidel）。"""
  field = q.copy()
  weight = 0.5 * coefficient
  values = field[:, :, SMOOTHED]
  for _ in range(cycles):
    if periodic:
      neighbours = np.roll(values, 1, axis=1) + np.roll(values, -1, axis=1)
    else:  # 未闭合的环：端点用单侧邻元复制
      left = np.concatenate([values[:, :1], values[:, :-1]], axis=1)
      right = np.concatenate([values[:, 1:], values[:, -1:]], axis=1)
      neighbours = left + right
    values = values + weight * (neighbours - 2 * values)
  field[:, :, SMOOTHED] = values
  return field


def shock_weights(q, chord_x0, chord, args):
  """(3.2.1) 的混合函数 S，以及每条径向线/每个表面上找到的激波位置。

  两遍检测：先用贴壁行在 ``--window`` 内定出激波弦向位置，再在它 ±``--band`` 的窄带内
  逐行找 |dp/ds| 峰。窗口内最强的梯度往往在前缘吸力爬升段，直接按全局阈值会把近壁的
  激波脚筛掉，所以阈值只与该窄带内的最大值比较。
  """
  xy = q[:, :, :2]
  step, cumulative, circumference = arc_lengths(xy)
  xc = (xy[:, :, 0] - chord_x0) / chord
  gradient = np.abs(np.roll(q[:, :, 6], -1, axis=1) - np.roll(q[:, :, 6], 1, axis=1))
  gradient /= step + np.roll(step, 1, axis=1)
  inside = (xc >= args.window[0]) & (xc <= args.window[1])
  if not inside.any():
    raise SystemExit(f'弦向窗口 {args.window} 内没有单元；检查 --window')

  centres = {}
  for upper in (True, False):
    mask = inside[args.wall_row] & (
      (xy[args.wall_row, :, 1] >= 0) if upper else (xy[args.wall_row, :, 1] < 0)
    )
    if mask.any():
      centres[upper] = float(
        xc[args.wall_row, int(np.where(mask, gradient[args.wall_row], 0).argmax())]
      )
  if not centres:
    raise SystemExit(f'贴壁行 {args.wall_row} 在窗口内没有单元；检查 --window / --wall-row')

  band = np.zeros_like(gradient, dtype=bool)
  for centre in centres.values():
    band |= np.abs(xc - centre) <= args.band
  reference = float(gradient[band].max())
  weights = np.zeros(xy.shape[:2])
  found = []
  for row in range(xy.shape[0]):
    for upper, centre in centres.items():
      mask = inside[row] & band[row] & ((xy[row, :, 1] >= 0) if upper else (xy[row, :, 1] < 0))
      if not mask.any():
        continue
      masked = np.where(mask, gradient[row], 0.0)
      column = int(masked.argmax())
      if masked[column] <= args.threshold * reference:
        continue
      sigma = max(args.sigma_cells * step[row, (column - 1) % xy.shape[1]], 1e-12)
      gap = ring_gap(cumulative[row], circumference[row], column)
      weights[row] = np.maximum(weights[row], np.exp(-0.5 * (gap / sigma) ** 2))
      found.append((row, column, float(xc[row, column]), upper))
  return weights, found, reference, centres


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('case', type=Path)
  parser.add_argument('out', type=Path)
  parser.add_argument('--cycles', type=int, default=160, help='论文的 N_SC')
  parser.add_argument('--coefficient', type=float, default=0.1, help='论文的 c_i')
  parser.add_argument('--sigma-cells', type=float, default=8.0, help='混合带半宽（当地步长数）')
  parser.add_argument(
    '--threshold', type=float, default=0.05, help='激波梯度阈值（相对窄带内最大值）'
  )
  parser.add_argument(
    '--band', type=float, default=0.08, help='以贴壁行激波位置为中心的搜索半宽 x/c'
  )
  parser.add_argument(
    '--window', type=float, nargs=2, default=[0.25, 0.80], help='激波搜索的弦向窗口 x/c'
  )
  parser.add_argument('--wall-row', type=int, default=0, help='贴壁的径向行号（交付场为 0）')
  args = parser.parse_args()

  case, out = args.case.resolve(), args.out.resolve()
  if out.exists():
    raise SystemExit(f'输出目录已存在，换一个名字：{out}')
  parameters = json.loads((case / 'parameters.json').read_text(encoding='utf-8'))
  ns, nr = parameters['nt'], parameters['nr']
  lines = (case / 'baseflow.dat').read_text(encoding='utf-8').splitlines()
  q = np.array([[float(value) for value in line.split()] for line in lines[2:]])
  if q.shape != (nr * ns, 9):
    raise SystemExit(f'场文件形状 {q.shape} 与 parameters.json 的 {nr}x{ns} 不符')
  q = q.reshape(nr, ns, 9)

  wall_x = q[args.wall_row, :, 0]
  chord_x0, chord = float(wall_x.min()), float(wall_x.max() - wall_x.min())
  step = np.linalg.norm(np.roll(q[:, :, :2], -1, axis=1) - q[:, :, :2], axis=2)
  # 环向是否闭合：首末单元的间距应与该处当地步长同量级（远场步长远大于近壁步长）
  wrap = np.linalg.norm(q[:, 0, :2] - q[:, -1, :2], axis=1)
  local = 0.5 * (step[:, 0] + step[:, -1])
  periodic = bool(np.all(wrap < 3 * local))

  weights, found, reference, centres = shock_weights(q, chord_x0, chord, args)
  smooth = smoothed_field(q, args.cycles, args.coefficient, periodic)
  blended = q.copy()
  blended[:, :, SMOOTHED] = q[:, :, SMOOTHED] + weights[:, :, None] * (
    smooth[:, :, SMOOTHED] - q[:, :, SMOOTHED]
  )
  blended[:, :, 6] = blended[:, :, 2] * R * blended[:, :, 5]
  blended[:, :, 7] = np.linalg.norm(blended[:, :, 3:5], axis=2) / np.sqrt(
    GAMMA * R * blended[:, :, 5]
  )

  def thickness(field):
    """论文 p.15 的激波厚度 δP = max(ΔP)/max|dP/ds|，按弦长无量纲。"""
    gradient = np.abs(np.roll(field[:, :, 6], -1, axis=1) - field[:, :, 6]) / step
    return float(np.ptp(field[:, :, 6]) / gradient.max() / chord)

  title = lines[0].rstrip().rstrip('"')
  title += f' shock_smoothing=nsc{args.cycles}-ci{args.coefficient:g}'
  title += f'-sigma{args.sigma_cells:g}-v1"'
  out.mkdir(parents=True)
  shutil.copyfile(case / 'mesh.txt', out / 'mesh.txt')
  parameters = dict(parameters)
  parameters['provenance'] = {
    **parameters.get('provenance', {}),
    'shock_smoothing': {
      'paper': 'Crouch et al. (2007) 3.2, eqs. (3.2.1)-(3.2.2)',
      'cycles': args.cycles,
      'coefficient': args.coefficient,
      'sigma_cells': args.sigma_cells,
      'gradient_threshold': args.threshold,
      'window_x_over_c': list(args.window),
      'direction': 'circumferential, ' + ('periodic' if periodic else 'one-sided'),
      'blend': 'q_final = (1-S)*q_original + S*q_smooth, S = max exp(-ds^2/(2*sigma_s^2))',
      'variables': 'rho u v T nu_tilde; p and Ma recomputed from rho*R*T',
      'source_case': case.name,
      'source_baseflow_sha256': hashlib.sha256((case / 'baseflow.dat').read_bytes()).hexdigest(),
    },
  }
  (out / 'parameters.json').write_text(
    json.dumps(parameters, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
  )
  body = '\n'.join(' '.join(f'{value:.8e}' for value in row) for row in blended.reshape(-1, 9))
  (out / 'baseflow.dat').write_text(f'{title}\n{lines[1]}\n{body}\n', encoding='utf-8')

  print(
    f'{case.name} -> {out.name}: N_SC={args.cycles}, c_i={args.coefficient:g}, '
    f'S 半宽 {args.sigma_cells:g} 格, 阈值 {args.threshold:g}, 窗口 {args.window}'
  )
  print(f'  环向闭合 {periodic}；弦长 {chord:.6f}；窄带内最大 |dp/ds| = {reference:.3e} Pa/m')
  names = {True: '上表面', False: '下表面'}
  print(
    '  贴壁行定出的激波位置：'
    + '，'.join(f'{names[up]} x/c={centre:.4f}' for up, centre in centres.items())
  )
  upper = [item[2] for item in found if item[3]]
  lower = [item[2] for item in found if not item[3]]
  print(f'  检出激波 {len(found)} 处：上表面 {len(upper)} 条线、下表面 {len(lower)} 条线')
  if upper:
    print(f'    上表面 x/c {min(upper):.4f} ~ {max(upper):.4f}')
  if lower:
    print(f'    下表面 x/c {min(lower):.4f} ~ {max(lower):.4f}')
  print(
    f'  混合带：S>0.5 的单元 {int((weights > 0.5).sum())}，S>0.01 的单元 '
    f'{int((weights > 0.01).sum())}，S 最大 {weights.max():.3f}'
  )
  print(
    f'  激波厚度 δP/c：原场 {thickness(q):.3e} -> 平滑 {thickness(blended):.3e}'
    f'（论文：2 格 -> 约 10 格）'
  )
  for name, column in zip(('rho', 'u', 'v', 'T', 'nu~'), SMOOTHED):
    change = np.abs(blended[:, :, column] - q[:, :, column])
    percent = 100 * change.max() / np.abs(q[:, :, column]).max()
    print(f'    max|Δ{name}| = {change.max():.3e}（{percent:.2f}% of max|{name}|）')
  print(
    f'  正性：rho>0 {bool((blended[:, :, 2] > 0).all())}, T>0 '
    f'{bool((blended[:, :, 5] > 0).all())}, nu~>=0 {bool((blended[:, :, 8] >= 0).all())}'
  )


if __name__ == '__main__':
  main()
