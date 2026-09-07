# 安装并运行稳定性分析

这个程序读取已经收敛的定常流场，计算小扰动的增长率和形状。它不能代替基流求解器。先完成 [PurineCFD-R2 入门教程](https://github.com/PurineAcO/PurineCFD-R2/blob/main/docs/quickstart.md)，得到完整的圆柱结果文件夹。

## 1. 安装 Python 工具

以下命令在 **Ubuntu 终端**执行。Windows 用户也在 WSL 的 Ubuntu 中操作，不用 PowerShell。先执行：

```sh
sudo apt update
sudo apt install -y git curl python3
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv --version
```

`uv` 帮你下载程序需要的 Python 库并放在独立目录，不需要手工配置 Python 环境。安装方法来自 [uv 官方文档](https://docs.astral.sh/uv/getting-started/installation/)。

```sh
mkdir -p ~/cfd
cd ~/cfd
git clone https://github.com/PurineAcO/crouch-hopf.git
cd ~/cfd/crouch-hopf
uv sync --frozen
```

首次安装需要联网。看到安装完成且没有错误即可。以后每次使用先执行 `cd ~/cfd/crouch-hopf`，不必再次 clone。尚未合并的 draft PR 应使用 PR 描述中的下载命令；这里下载的是 main。

## 2. 检查安装

```sh
cd ~/cfd/crouch-hopf
OPENBLAS_NUM_THREADS=2 uv run pytest -q
```

看到 `passed` 且没有 `failed` 表示自检通过。后续命令前的 `OPENBLAS_NUM_THREADS=2` 限制数学库使用两个线程，避免大量线程争抢 CPU。

## 3. 导入收敛的圆柱结果

假定基流结果位于上一份教程的默认目录：

```sh
cd ~/cfd/crouch-hopf
uv run python tools/import_cylinder.py ~/cfd/PurineCFD-R2/run/cylinder-laminar-re47 runs/cylinder-laminar-re47 --symmetrize
```

该命令自动生成 `ransdata.txt`、`edge.txt` 和 `input/parameters.json`。它会检查收敛标记、模型、正密度/温度，以及圆柱网格几何；不同工况要用不同输出目录，不能覆盖。

`--symmetrize` 只消除对称圆柱基流的浮点舍入差异，允许的归一化改变量不超过 1e-10。一般翼型、非零攻角或非对称基流不能使用它。本导入器只适用于附带的圆柱 O 型网格，不接受任意 Fluent 或 NACA 0012 文件；其他网格需要按[输入约定](numerics.md)另行导出和检查。

## 4. 装配并求解

```sh
OPENBLAS_NUM_THREADS=2 uv run python crouch/solvemain.py runs/cylinder-laminar-re47 --model laminar --alpha 0
OPENBLAS_NUM_THREADS=2 uv run python crouch/eigmain.py runs/cylinder-laminar-re47 --sigma-imag 0.73 --k 12
```

第一行生成矩阵，第二行计算移位附近的 12 个特征模态。`--alpha 0` 与论文圆柱中性点曲线采用的中心格式一致。求解保留所有边界自由度，使用完整稀疏广义特征方程。

若只研究镜面对称圆柱的反对称分支，可复用已经装配的矩阵：

```sh
OPENBLAS_NUM_THREADS=2 uv run python crouch/eigmain.py runs/cylinder-laminar-re47 --sigma-imag 0.73 --k 12 --wake-symmetry --ordering COLAMD --label wake_antisymmetric
```

`--wake-symmetry` 要求偶数周向单元数，以及关于 y=0 的对称网格、基流和完整矩阵。它保留所有径向环，求解后将模态恢复到完整网格，并重新检查完整方程和边界约束。几何或矩阵不满足对称性时会报错，不会自动修改输入。它只返回反对称分支，不包含另一对称分支。

`--label wake_antisymmetric` 将结果保存为 `wake_antisymmetric_eigenvalues.csv`、`wake_antisymmetric_modes.npz` 和 `wake_antisymmetric_solve.json`，便于与已有完整求解比较。同一标签会覆盖该标签的旧结果；省略标签时仍为 `wake`。`--k 6` 表示六个模态，选项和值之间需要空格。

`--ordering` 接受 `COLAMD`（默认）或 `MMD_AT_PLUS_A`，仅控制稀疏 LU 排序。在已有 Re47 算例中，COLAMD 反对称求解的峰值内存约为 0.55 GiB；MMD 试算超过 3.36 GiB 后停止，因此本次研究采用 COLAMD。每个新网格仍需检查求解报告中的 `peak_rss_bytes` 和 `factor_nnz`，不能据此保证更细网格满足 15 GiB 限制。

装配会显示 `Linearized ring`，求解先显示完整矩阵大小，结束时输出特征值和包含 LU 用时的报告。稀疏矩阵分解可能一段时间不输出文字，这不一定表示卡住。内存需求取决于网格和模型，远大于输入文件大小；遇到 `Killed` 应先检查可用内存，不要反复启动多个求解进程。

SA 算例应先导入 SA 基流，再在装配时使用 `--model sa`。程序要求输入元数据明确记录模型，不支持缺少模型标签的旧输入。特征值求解自动读取装配所用的模型，不需要再次选择。SA 输入还须具有 `sa_formulation=crouch-2007` 标记；旧版 SA 基流需要重新计算，不能通过修改标签继续使用。

## 5. 查看数值与云图

```sh
cat runs/cylinder-laminar-re47/wake_eigenvalues.csv
```

表头 `mode` 是模态编号；`growth_D_U` 是无量纲增长率，正值表示增长，负值表示衰减；`omega_D_U` 是角频率，`St` 是 Strouhal 数。`residual` 检查特征方程求解精度，不能证明网格误差小。

**编号 0 不保证是物理尾迹模态。**下面先画编号 0 来检查画图功能；根据表格、尾迹结构及邻近工况的连续性选择实际研究模态后，替换 `--mode` 后面的编号。

```sh
uv run python tools/plot_mode.py runs/cylinder-laminar-re47 --mode 0
```

图文件为 `runs/cylinder-laminar-re47/wake_mode_0.png` 和同名 PDF。图中的扰动幅值经过归一化，不是实际振荡幅值。Windows 用户可输入：

```sh
explorer.exe runs/cylinder-laminar-re47
```

这会打开结果文件夹。后续研究应保存命令、输入文件、`assembly.json`、`wake_solve.json` 和全部结果，不能只保存云图。

## 常见问题

| 提示 | 下一步 |
| --- | --- |
| `uv: command not found` | 重新执行 `source ~/.local/bin/env`，确认第 1 步安装成功 |
| `Base flow has not converged` | 回到基流程序完成收敛计算，不能使用 200 步安装检查结果 |
| `model` 缺失或不一致 | 核对基流来源，使用新版运行入口重新导出；不要猜模型 |
| `Output exists` | 换一个新目录名 |
| `Killed` | 检查系统内存和同时运行的进程；网格调整会影响精度，需要重新验证 |
| 特征值或残差报错 | 保存完整错误、输入参数和所用命令，交给负责数值方法的同门检查 |

基流参数须记录 `thermodynamics` 中的 R、Cp、Cv、gamma，并与当前配置一致；Purine 场文件须带 `thermo=ideal-air-cv717625-v1` 标记。旧版 Cv=717.645 的层流及 SA 基流均需重新计算，不能仅修改标签。
