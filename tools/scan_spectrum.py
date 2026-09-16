"""Scan the complex plane with a series of shifts and merge the results.

移位反演（``eigmain.py --sigma-imag``）只会返回**离 σ 最近的 k 个**特征值，所以单个移位必然
给出一个小团。要得到成片的谱，只能要么把 k 开大，要么沿一条线扫多个移位再合并去重。本脚本做
后一件事：给定频率区间与步长，逐个移位调用 ``crouch/eigmain.py``（已存在的标签会跳过），
最后把所有 ``*_eigenvalues.csv`` 合并、去重，并报告最大增长率的特征值。

用法::

  uv run python tools/scan_spectrum.py runs/<case> --omega 0.05 0.55 0.05 --k 12

每个移位只在 ω_r 轴上取值、σ_real = 0，因此扫描范围与步长是唯一需要选的东西；脚本会把
每个移位的结果分别保存（``*_eigenvalues.csv`` / ``*_modes.npz`` / ``*_solve.json``），
合并结果写入 ``<case>/scan_merged.csv``。
"""

import argparse
import ast
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = 'window,sigma_imag,mode,St,omega_D_U,growth_D_U,growth_per_s,frequency_Hz,residual'


def label_for(omega):
  return f'scan{round(omega * 1000):04d}'


def collect(case, labels):
  """Return (numeric rows, window tags); rows hold sigma_imag, mode, St, omega_r, omega_i."""
  values, tags = [], []
  for omega, label in labels:
    data = np.genfromtxt(case / f'{label}_eigenvalues.csv', delimiter=',', names=True)
    for index in range(len(data)):
      values.append(
        [
          omega,
          index,
          data['St'][index],
          data['omega_D_U'][index],
          data['growth_D_U'][index],
          data['growth_per_s'][index],
          data['frequency_Hz'][index],
          data['residual'][index],
        ]
      )
      tags.append(label)
  return np.asarray(values, dtype=float), tags


def deduplicate(rows):
  """Drop eigenvalues repeated by neighbouring shifts (same omega_r and growth)."""
  order = np.lexsort((rows[:, 4], rows[:, 3]))
  ordered = rows[order]
  keep = np.ones(len(ordered), dtype=bool)
  keep[1:] = (np.abs(np.diff(ordered[:, 3])) > 1e-6) | (np.abs(np.diff(ordered[:, 4])) > 1e-6)
  return order[keep]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('case', type=Path)
  parser.add_argument(
    '--omega', type=float, nargs=3, metavar=('START', 'STOP', 'STEP'), required=True
  )
  parser.add_argument('--k', type=int, default=12)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--ordering', default='COLAMD')
  parser.add_argument('--skip-existing', action='store_true', default=True)
  parser.add_argument(
    '--merge-all',
    action='store_true',
    help='merge every scan*_eigenvalues.csv already in the case directory (do not solve)',
  )
  args = parser.parse_args()
  case = args.case.resolve()
  if args.merge_all:
    labels = []
    for path in sorted(case.glob('scan*_eigenvalues.csv')):
      digits = path.name[4:-16]
      if not digits.isdigit():
        raise SystemExit(f'Unexpected scan file name: {path.name}')
      labels.append((int(digits) / 1000.0, path.name[:-16]))
    if not labels:
      raise SystemExit(f'No scan*_eigenvalues.csv found under {case}')
  else:
    start, stop, step = args.omega
    if step <= 0 or stop < start:
      raise SystemExit('--omega needs START < STOP and STEP > 0')
    shifts = np.arange(start, stop + 0.5 * step, step)
    labels = [(float(omega), label_for(float(omega))) for omega in shifts]
  for omega, label in labels:
    if (case / f'{label}_eigenvalues.csv').is_file():
      print(f'{label}: cached (sigma_imag={omega:g})', flush=True)
      continue
    print(f'{label}: solving k={args.k} at sigma_imag={omega:g}', flush=True)
    command = [
      sys.executable,
      str(ROOT / 'crouch/eigmain.py'),
      str(case),
      '--sigma-real',
      '0',
      '--sigma-imag',
      str(omega),
      '--k',
      str(args.k),
      '--seed',
      str(args.seed),
      '--ordering',
      args.ordering,
      '--label',
      label,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
      raise SystemExit(f'{label} failed:\n{result.stdout}\n{result.stderr}')
    tail = [line for line in result.stdout.strip().splitlines() if line.startswith('{')][-1]
    info = ast.literal_eval(tail)
    print(
      f'  {info["factor_seconds"]:.1f}s factor, {info["total_seconds"]:.1f}s total, '
      f'max residual {info["max_residual"]:.1e}',
      flush=True,
    )
  rows, tags = collect(case, labels)
  keep = deduplicate(rows)
  unique = rows[keep]
  target = case / 'scan_merged.csv'
  lines = [COLUMNS]
  for index in keep:
    row = rows[index]
    lines.append(
      f'{tags[index]},{row[0]:.4f},{int(row[1])},{row[2]:.8e},{row[3]:.8e},{row[4]:.8e},'
      f'{row[5]:.8e},{row[6]:.8e},{row[7]:.2e}'
    )
  target.write_text('\n'.join(lines) + '\n')
  print(f'\nmerged {len(rows)} -> {len(unique)} unique eigenvalues; wrote {target}')
  print(f'omega_r coverage {unique[:, 3].min():.3f} .. {unique[:, 3].max():.3f}')
  print(f'omega_i coverage {unique[:, 4].min():+.4f} .. {unique[:, 4].max():+.4f}')
  best = int(np.argmax(unique[:, 4]))
  print(
    f'largest growth: omega_i={unique[best, 4]:+.5f} at omega_r={unique[best, 3]:.4f} '
    f'(St={unique[best, 2]:.4f})'
  )


if __name__ == '__main__':
  main()
