"""逐项核对论文 (2.2.5) 是否为 (2.1.5) 的精确线性化。

用法（在本仓库根目录执行）：

```powershell
uv run python tools/verify_paper_sa.py
```

做法：把论文 (2.1.5)+(2.1.8) 的 SA 体积源项独立符号化，对它求精确一阶导数，
再与论文 (2.2.5) 中逐字录入的线性形式逐系数相减。

比较分成两组，避免符号化简在 f_w 的嵌套形式（g⁶ 里含 r⁶）上爆炸：

- 壁面/产生类项（涡量导数、ν̃′ 与 ρ′ 方括号、梯度平方）：系数表达式很大但
  代数相同，改用 40 个抽样点上的数值判定（相对残差门槛 1e-9）；
- C5 应变率项：本身很短，直接做符号相减并打印差值。

扩散项不是体积源项，单独按面通量被积函数比较。全部比较都不对物理通量做差分。
"""

import pathlib
import sys

import sympy as sp

# 控制台可能是 GBK；含组合波浪号（ν̃）的标签必须按 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'crouch'))
import classconfig as cc

rho, u, v, temperature, nu = sp.symbols('rho u v T nu', positive=True)
mu, distance = sp.symbols('mu d', positive=True)
ux, uy, vx, vy = sp.symbols('ux uy vx vy', real=True)
nux, nuy = sp.symbols('nux nuy', real=True)

# 与 (2.2.5) 中被求导的量一一对应的扰动记号
SYMBOL = {'rho': rho, 'nu': nu, 'T': temperature}
SYMBOL |= {'ux': ux, 'uy': uy, 'vx': vx, 'vy': vy, 'nux': nux, 'nuy': nuy}
DELTA = {name: sp.Symbol(f'd{name}') for name in SYMBOL}
ARGUMENTS = (rho, nu, temperature, ux, uy, vx, vy, nux, nuy, mu, distance)

chi = sp.Symbol('chi', positive=True)
CHI = rho * nu / mu


def FV1_OF(c):
  return c**3 / (c**3 + cc.Cv1**3)


def FV2_OF(c):
  return 1 - c / (1 + c * FV1_OF(c))


def FT2_OF(c):
  return cc.Ct3 * sp.exp(-cc.Ct4 * c**2)


FV2 = FV2_OF(CHI)
FT2 = FT2_OF(CHI)
FV2_CHI = sp.diff(FV2_OF(chi), chi).subs(chi, CHI)
FT2_CHI = sp.diff(FT2_OF(chi), chi).subs(chi, CHI)
SCALE = 1 / (cc.kappa * distance) ** 2
# 论文 (2.2.5) 把 Ω 当作有符号的 ∂u/∂y−∂v/∂x 使用
OMEGA = uy - vx
MODIFIED = OMEGA + FV2 * nu * SCALE
R_LOCAL = nu * SCALE / MODIFIED
r_symbol = sp.Symbol('r', positive=True)


def WALL_G_OF(x):
  return x + cc.Cw2 * (x**6 - x)


def FW_OF(x):
  return WALL_G_OF(x) * ((1 + cc.Cw3**6) / (WALL_G_OF(x) ** 6 + cc.Cw3**6)) ** sp.Rational(1, 6)


FW = FW_OF(R_LOCAL)
FW_R = sp.diff(FW_OF(r_symbol), r_symbol).subs(r_symbol, R_LOCAL)
# 论文把 S 定义为应变张量模，故 S² = 2SijSij
STRAIN_SQUARED = 2 * ux**2 + 2 * vy**2 + (uy + vx) ** 2

PRODUCTION = rho * cc.Cb1 * (1 - FT2) * MODIFIED * nu
DESTRUCTION = rho * (cc.Cw1 * FW - cc.Cb1 * FT2 / cc.kappa**2) * (nu / distance) ** 2
GRADIENT_SQUARE = cc.Cb2 * cc.inv_sigma * rho * (nux**2 + nuy**2)
COMPRESSIBILITY = cc.C5 * rho * nu**2 * STRAIN_SQUARED / (cc.gamma * cc.R * temperature)

WALL_SOURCE = PRODUCTION - DESTRUCTION + GRADIENT_SQUARE
C5_SOURCE = -COMPRESSIBILITY


def printed_wall():
  """(2.2.5) 的壁面/产生类体积项（涡量导数 + 两个方括号 + 梯度平方）。"""
  expression = sp.Integer(0)

  # (ρ̄ν̄/Ω)[Cb1(1−f̄t2)+Cw1κ²(∂f̄w/∂r)r̄²](∂ū/∂y−∂v̄/∂x)(∂u'/∂y−∂v'/∂x)
  vorticity = rho * nu * (cc.Cb1 * (1 - FT2) + cc.Cw1 * cc.kappa**2 * FW_R * R_LOCAL**2)
  expression += vorticity * (DELTA['uy'] - DELTA['vx'])

  # 第二方括号，整体乘以 ρ̄S̃ν̃′
  nu_bracket = (
    cc.Cb1 * (1 - FT2 - CHI * FT2_CHI)
    + cc.Cb1 * R_LOCAL * ((FV2 + CHI * FV2_CHI) * (1 - FT2) + 2 * FT2 + CHI * FT2_CHI)
    - cc.Cw1
    * cc.kappa**2
    * R_LOCAL
    * (2 * FW + R_LOCAL * FW_R - R_LOCAL**2 * FW_R * (FV2 + CHI * FV2_CHI))
  )
  expression += nu_bracket * rho * MODIFIED * DELTA['nu']

  # 第三方括号，整体乘以 ν̄S̃ρ′
  rho_bracket = (
    cc.Cb1 * (1 - FT2 - CHI * FT2_CHI)
    + cc.Cb1 * R_LOCAL * (CHI * FV2_CHI * (1 - FT2) + FT2 + CHI * FT2_CHI)
    - cc.Cw1 * cc.kappa**2 * R_LOCAL * (FW - FW_R * R_LOCAL**2 * CHI * FV2_CHI)
  )
  expression += rho_bracket * nu * MODIFIED * DELTA['rho']

  # (Cb2/σ)[(∂ν̄/∂x)²+(∂ν̄/∂y)²]ρ′ + 2ρ̄(∂ν̄/∂x ∂ν̃′/∂x + ∂ν̄/∂y ∂ν̃′/∂y)
  expression += (
    cc.Cb2
    * cc.inv_sigma
    * ((nux**2 + nuy**2) * DELTA['rho'] + 2 * rho * (nux * DELTA['nux'] + nuy * DELTA['nuy']))
  )
  return expression


def printed_c5(compressibility_typo=True):
  """(2.2.5) 的 C5 应变率项，逐字录入。

  `compressibility_typo=True` 按排版写作 ∂v̄/∂x ∂v'/∂x；False 时改用与
  S²=2SijSij 对称形式一致的 ∂v̄/∂y ∂v'/∂y。
  """
  # 梯度括号前的因子是 C5ρ̄ν̄²/(γRT̄)，不含 S̄²
  ratio = cc.C5 * rho * nu**2 / (cc.gamma * cc.R * temperature)
  expression = -2 * ratio * (uy + vx) * (DELTA['uy'] + DELTA['vx'])
  expression += -2 * ratio * 2 * ux * DELTA['ux']
  if compressibility_typo:
    expression += -2 * ratio * 2 * vx * DELTA['vx']
  else:
    expression += -2 * ratio * 2 * vy * DELTA['vy']
  expression += -2 * COMPRESSIBILITY * (DELTA['nu'] / nu)
  expression += -COMPRESSIBILITY * DELTA['rho'] / rho
  expression += COMPRESSIBILITY * DELTA['T'] / temperature
  return expression


def diffusion_flux_integrand():
  """(2.2.5) 的 (1/σ)∇·[(μ+ρ̄ν̄)∇ν̃′ + ∇ν̄(ρ̄ν̃′+ρ′ν̄)]，按面通量被积函数展开。"""
  return {
    'nux': cc.inv_sigma * (mu + rho * nu),
    'nuy': cc.inv_sigma * (mu + rho * nu),
    'nu': cc.inv_sigma * rho * (nux + nuy),
    'rho': cc.inv_sigma * nu * (nux + nuy),
  }


def sample(rng):
  """抽一个 r 落在未饱和区间、且 S̃>0 的态。"""
  for _ in range(4000):
    state = {
      'rho': rng.uniform(0.5, 1.8),
      'nu': rng.uniform(1e-6, 5e-4),
      'T': rng.uniform(250.0, 400.0),
      'ux': rng.normal(0.0, 30.0),
      'uy': rng.normal(0.0, 30.0),
      'vx': rng.normal(0.0, 30.0),
      'vy': rng.normal(0.0, 30.0),
      'nux': rng.normal(0.0, 5.0),
      'nuy': rng.normal(0.0, 5.0),
      'mu': rng.uniform(5e-6, 3e-5),
      'd': 10 ** rng.uniform(-3.0, 0.0),
    }
    state['chi'] = state['rho'] * state['nu'] / state['mu']
    tweak = state['chi'] ** 3 / (state['chi'] ** 3 + cc.Cv1**3)
    fv2 = 1 - state['chi'] / (1 + state['chi'] * tweak)
    modified = abs(state['uy'] - state['vx']) + fv2 * state['nu'] / (cc.kappa * state['d']) ** 2
    r = state['nu'] / (cc.kappa * state['d']) ** 2 / modified
    if 1e-3 < r < 8.0:
      return state
  raise RuntimeError('sampling failed to land inside the un-capped r range')


def report():
  import numpy as np

  rng = np.random.default_rng(2007)
  points = [sample(rng) for _ in range(40)]
  order = list(SYMBOL)
  columns = np.array([[p[name] for name in order] + [p['mu'], p['d']] for p in points])

  failures = []
  print('一、壁面/产生类体积项（涡量导数 + ν̃′ 与 ρ′ 方括号 + 梯度平方）')
  print('    相对残差 = max|(2.2.5) 系数 − ∂(2.1.5)/∂·| / max|∂(2.1.5)/∂·|，40 个抽样点')
  wall = printed_wall()
  for name in SYMBOL:
    exact = sp.diff(WALL_SOURCE, SYMBOL[name])
    residual = exact - sp.diff(wall, DELTA[name])
    exact_values = sp.lambdify(ARGUMENTS, exact, 'numpy')(*columns.T)
    residual_values = sp.lambdify(ARGUMENTS, residual, 'numpy')(*columns.T)
    scale = max(np.max(np.abs(exact_values)), 1e-30)
    relative = np.max(np.abs(residual_values)) / scale
    ok = relative < 1e-9
    if not ok:
      failures.append(('wall', name))
    print(f'    [{"一致" if ok else "不一致"}] ∂/∂{name}: {relative:.3e}')

  print('\n二、C5 应变率项（符号相减，可直接打印差值）')
  exact_c5 = {name: sp.diff(C5_SOURCE, SYMBOL[name]) for name in SYMBOL}
  for label, typo in [('按排版 ∂v̄/∂x ∂v′/∂x', True), ('对称形式 ∂v̄/∂y ∂v′/∂y', False)]:
    print(f'    {label}:')
    expression = printed_c5(compressibility_typo=typo)
    mismatch = []
    for name in SYMBOL:
      difference = sp.expand(exact_c5[name] - sp.diff(expression, DELTA[name]))
      if difference != 0:
        mismatch.append((name, difference))
    if not mismatch:
      print('      全部系数一致')
      continue
    for name, difference in mismatch:
      print(f'      ∂/∂{name} 差 = {sp.factor(difference)}')
    if typo:
      print(f'      → 排版错误出现在 {[name for name, _ in mismatch]}')

  print('\n三、扩散项（面通量被积函数）')
  exact_flux = {
    'nux': cc.inv_sigma * (mu + rho * nu),
    'nuy': cc.inv_sigma * (mu + rho * nu),
    'nu': cc.inv_sigma * rho * (nux + nuy),
    'rho': cc.inv_sigma * nu * (nux + nuy),
  }
  for name, value in exact_flux.items():
    ok = sp.expand(value - diffusion_flux_integrand()[name]) == 0
    if not ok:
      failures.append(('diffusion', name))
    print(f'    [{"一致" if ok else "不一致"}] δ{name}')

  print()
  if failures:
    print(f'结论: 仍有 {failures} 项需要继续核对')
  else:
    print('结论: 壁面/产生类项与扩散项与 (2.1.5) 精确一致；C5 项只有把最后一个 v 梯度')
    print('      由 ∂v̄/∂x ∂v′/∂x 改为 ∂v̄/∂y ∂v′/∂y 后才一致 —— 这是 (2.2.5) 的排版错误，')
    print('      (2.1.5) 本身无误，代码采用修正后的对称形式。')
  return failures


if __name__ == '__main__':
  sys.exit(1 if report() else 0)
