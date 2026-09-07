# Crouch-Py

基于定常基流的二维全局线性稳定性分析，支持层流四变量与 SA-RANS 五变量，采用 Crouch 等（2007）的对流离散和稀疏移位反演。

**第一次使用请阅读[安装、运行和画图教程](docs/quickstart.md)。**教程从 Ubuntu/WSL 环境开始，并说明如何接收 PurineCFD-R2 的结果。

- [数值方法、输入格式与验证约定](docs/numerics.md)
- [基流求解器 PurineCFD-R2](https://github.com/PurineAcO/PurineCFD-R2)

已安装 uv 且已有收敛圆柱结果时，在仓库目录执行：

```sh
uv sync --frozen
uv run python tools/import_cylinder.py ../PurineCFD-R2/run/cylinder-laminar-re47 runs/cylinder-laminar-re47 --symmetrize
OPENBLAS_NUM_THREADS=2 uv run python crouch/solvemain.py runs/cylinder-laminar-re47 --model laminar --alpha 0
OPENBLAS_NUM_THREADS=2 uv run python crouch/eigmain.py runs/cylinder-laminar-re47 --sigma-imag 0.73 --k 12
```

对满足镜面对称检查的圆柱算例，可在特征值命令中增加 `--wake-symmetry --ordering COLAMD`，只求反射反对称分支并减小分解内存。投影覆盖全部径向环及耦合边界约束，保存的模态仍在完整网格上。该选项不改变装配矩阵或边界权重；非对称几何、基流或矩阵会被拒绝。排序默认采用 COLAMD；MMD 并不保证更省内存。详见[精确反射投影与排序](docs/numerics.md#精确反射投影与排序)。

SA 基流选择 `--model sa`，并须由包含 C5 修正的求解器重新计算，元数据及场文件须标明 `sa_formulation=crouch-2007`。输入必须明确记录模型，层流输入的 ν̃ 必须为零。分子黏度在线性化时冻结，暂不包含 Sutherland 温度导数。模型选择不需要修改源码。

当前维护入口是 Python 的 `crouch/` 和 `tools/`。旧 C++ 移植、旧网格转换/MATLAB 脚本及未使用的控制台兼容模块已移除；历史版本仍可在 Git 历史中查阅。附带导入器只接受圆柱 O 型网格；其他几何须按输入约定提供经过验证的数据，不能套用圆柱转换命令。

基流参数须记录 `thermodynamics` 中的 R、Cp、Cv、gamma，并与当前配置一致；Purine 场文件须带 `thermo=ideal-air-cv717625-v1` 标记。旧版 Cv=717.645 的层流及 SA 基流均需重新计算，不能仅修改标签。
