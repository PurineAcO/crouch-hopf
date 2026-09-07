# Crouch-Py

二维可压缩定常基流的全局线性稳定性分析，支持层流与 Spalart–Allmaras（SA）模型。输入为 O 型网格及基流，输出为广义特征矩阵和扰动模态。

## 模型选择

装配时必须显式指定模型：

```sh
uv sync --frozen
uv run python crouch/solvemain.py /path/to/case --model laminar --alpha 0
uv run python crouch/eigmain.py /path/to/case --sigma-imag 0.73 --k 12
```

SA 基流使用 `--model sa`，例如 NACA 0012 的 SA-RANS 稳定性分析。两端必须使用同一物理模型；该接口尚不构成对 NACA 0012 算例的验证。`--alpha` 可取 0、0.2、1，分别为论文的中心、混合与迎风格式。圆柱临界点对照论文图 3 时使用 0。

| 设置 | 层流 `laminar` | 湍流 `sa` |
| --- | --- | --- |
| 特征变量 | ρ、u、v、T | ρ、u、v、T、ν̃ |
| 湍流黏度 | 严格为零 | μt=ρν̃fv1 |
| SA 输运、源项 | 不装配到最终算子 | 保留 |
| 基流 ν̃ | 必须全部为零 | 必须有限、非负 |

局部雅可比共用五分量顺序；层流在验证 SA 列与流动方程无耦合后，提取四变量矩阵。保留第五分量的局部结构用于两种模型共用重构与边界代码。两种模型均冻结基流分子黏度，暂不包含 Sutherland 温度导数。

## 输入与输出

每个算例目录包含：

- `ransdata.txt`：首行为列名，至少含 `s n x y sad vol rho u v T miubl`；`miubl` 是 SA 工作变量 ν̃，单位 m²/s。周向和径向索引从 1 开始。
- `edge.txt`：首行为 `type s n id c1_s c1_n c1_id c2_s c2_n c2_id nx ny mx my`，类型为 `NS` 或 `WE`，法向含面长。
- `input/parameters.json`：`nt`、`nr`、`D_m`、`U_m_s`、`rho_kg_m3`、`T_K`、`mu_Pa_s`；建议同时记录 `model`，装配时会核对其与 `--model` 一致。旧数据未记录模型时，由调用者显式声明，程序不能从零湍黏度唯一推断模型。

物性和 SA 常数在根目录 `config.json`。其中 `solver.model` 给出模块级默认值，命令行装配始终要求 `--model`。输出 `S.npz`、`T.npz` 和 `assembly.json`；特征值求解读取装配记录的模型并校验矩阵维数。重复装配同一目录会覆盖矩阵，应为不同工况使用独立目录。

`eigmain.py` 使用移位反演与边界消元，采用 `exp(λt)` 约定。输出增长率 `Re(λD/U)`、角频率 `Im(λD/U)`、`St`、完整方程相对残差，以及各模态和量纲尺度。`--wake-symmetry` 仅适用于反射对称基流和匹配的周向编号；启用前检查算子反射对称性，不能用于一般攻角翼型基流。

## 方法与代码

- `convect.py`：Crouch 等（2007）式 3.1.10–19，分别重构系数矩阵与扰动，再求通量；SA 对流采用一阶迎风。式 3.1.14 印刷式重复使用 `1-sign`，实现按迎风方向将左权重写为 `1+sign`。
- `linearization.py`、`viscous.py`：分子/湍流黏度、复步长雅可比、中心黏性项与 SA 源项。
- `boundary.py`、`formmat.py`：边界约束、周期及虚单元映射、稀疏矩阵装配。
- `models.py`、`solvemain.py`、`eigmain.py`：模型校验、装配入口、特征求解。

扰动边界施加在首末单元中心环。基流离散与稳定性离散不同，特征残差小不代表空间误差小。既有圆柱层流扫描给出临界 Re 约 47，尚未验证网格无关性。

```sh
uv run ruff check crouch tests
uv run ruff format --check crouch tests
OPENBLAS_NUM_THREADS=2 uv run pytest -q
```

文献：J. D. Crouch, A. Garbaruk, D. Magidov, *Predicting the onset of flow unsteadiness based on global instability*, J. Comput. Phys. 224 (2007), 924–940. [DOI](https://doi.org/10.1016/j.jcp.2006.10.035)。
