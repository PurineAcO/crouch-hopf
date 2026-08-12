import time

import classconfig as cc
import readrans
import boundary
import grad
import convect
import turbulence
import formmat
import eigmain
import console
from tqdm import tqdm

_BAR = dict(colour="green", ncols=92, ascii=False, leave=False,
            bar_format="{l_bar}{bar:38}{r_bar}")

_t_all = time.time()

# ════════════════════ 1/6 读取 RANS 数据 ════════════════════
console.section("1/6  Read RANS data")
_t = time.time()
readrans.read_rans("ransdata.txt", "edge.txt")
console.ok(f"Read done: S_MAX={cc.S_MAX}, N_MAX={cc.N_MAX}, time {time.time()-_t:.2f}s")

# ════════════════════ 2/6 计算边界条件 ════════════════════
console.section("2/6  Boundary conditions")
_t = time.time()
for s in range(1, cc.S_MAX + 1):
    boundary.far_boundary(cc.goto_HALOcell((s, cc.N_MAX)))
    boundary.wing_boundary(cc.goto_HALOcell((s, 1)))
console.ok(f"Wall+far-field BC on {2 * cc.S_MAX} cells, time {time.time()-_t:.2f}s")

# ════════════════════ 3/6 计算梯度 ════════════════════
console.section("3/6  Gradients (Green-Gauss)")
_t = time.time()
for n in tqdm(range(1, cc.N_MAX + 1), desc="Gradient", **_BAR):
    for s in range(1, cc.S_MAX + 1):
        cell = cc.goto_HALOcell((s, n))
        grad.green_gauss_from_JST(cell, cell.north, cell.south, cell.east, cell.west)
for face in tqdm(cc.FaceList_NS, desc="Face grad NS", **_BAR):
    face.grad_2nd_mid()
for face in tqdm(cc.FaceList_WE, desc="Face grad WE", **_BAR):
    face.grad_2nd_mid()
console.ok(f"Gradients done, time {time.time()-_t:.2f}s")

# ════════════════════ 4/6 构建对流项 ════════════════════
console.section("4/6  Convective fluxes (skip wall/far-field)")
_t = time.time()
for n in tqdm(range(2, cc.N_MAX), desc="Convect", **_BAR):
    for s in range(1, cc.S_MAX + 1):
        convect.convect_hybrid(cc.goto_HALOcell((s, n)))
console.ok(f"Convection done, time {time.time()-_t:.2f}s")

# ════════════════════ 5/6 构建扩散项与源项 ════════════════════
console.section("5/6  Diffusion & source (S-A)")
_t = time.time()
for n in tqdm(range(1, cc.N_MAX + 1), desc="SA constants", **_BAR):
    for s in range(1, cc.S_MAX + 1):
        turbulence.SA_calc_constants(cc.goto_HALOcell((s, n)))
for s in tqdm(range(1, cc.S_MAX + 1), desc="SA ghost    ", **_BAR):
    for n in (0, -1):
        turbulence.SA_calc_constants(cc.goto_HALOcell((s, n)))
    for n in (cc.N_MAX + 1, cc.N_MAX + 2):
        turbulence.SA_calc_constants(cc.goto_HALOcell((s, n)))
for face in tqdm(cc.FaceList_NS, desc="Face diff NS", **_BAR):
    turbulence.diffusion_2nd_mid_SA(face)
for face in tqdm(cc.FaceList_WE, desc="Face diff WE", **_BAR):
    turbulence.diffusion_2nd_mid_SA(face)
for n in tqdm(range(2, cc.N_MAX), desc="Diff+Source ", **_BAR):
    for s in range(1, cc.S_MAX + 1):
        cell = cc.goto_HALOcell((s, n))
        turbulence.cell_diffusion(cell)
        turbulence.cell_source(cell)
console.ok(f"Diffusion+source done, time {time.time()-_t:.2f}s")

# ════════════════════ 6/6 组装矩阵与特征值求解 ════════════════════
console.section("6/6  Assemble S/T + eigenvalue analysis")
_t = time.time()
for n in tqdm(range(1, cc.N_MAX + 1), desc="Assemble    ", **_BAR):
    for s in range(1, cc.S_MAX + 1):
        formmat.formmat(cc.goto_HALOcell((s, n)))
S, T = formmat.build()
console.info(f"S: {S.shape}  nnz={S.nnz}   |   T: {T.shape}  nnz={T.nnz}  (interior cells {int(T.diagonal().sum())})")

eigmain.solve_eig(S, T)

console.ok(f"All done, total time {time.time()-_t_all:.1f}s")