import classconfig as cc
import readrans
import boundary
import grad
import convect
import turbulence
import formmat
import eigmain
import time
from tqdm import tqdm

# —————————— 阅读RANS数据 ——————————————
# 这一步之后,应该在cc建立起了cc.NMAX 和cc.SMAX,对于每个cell都应该建立了无粘通量F和G.
# cell 是按照二维数组结构化存储,但是face是摊开存储的,每个cell同别的face和cell链式连接.
# face应该建立起了面上物理量和链式连接关系.
# TODO: 日后可能提供C-Block网格的接口,因此以下函数内部均使用链式访问进行操作.

starttime = time.time()
readrans.read_rans("ransdata.txt","edge.txt")
print("RANS data read done.S_MAX=",cc.S_MAX,"N_MAX=",cc.N_MAX,"time:",time.time()-starttime)

# —————————— 计算边界条件 ——————————————

for s in range(1,cc.S_MAX+1):
    cell_far = cc.goto_HALOcell((s,cc.N_MAX))
    cell_wing = cc.goto_HALOcell((s,1))
    boundary.far_boundary(cell_far)
    boundary.wing_boundary(cell_wing)

# —————————— 计算梯度 ——————————————

for n in tqdm(range(1,cc.N_MAX+1), desc="Gradient", ascii=True):
    for s in range(1,cc.S_MAX+1):
        cell = cc.goto_HALOcell((s,n))
        grad.green_gauss_from_JST(cell,cell.north,cell.south,cell.east,cell.west)

for face in tqdm(cc.FaceList_NS, desc="Face grad (NS)", ascii=True):
    face.grad_2nd_mid()

for face in tqdm(cc.FaceList_WE, desc="Face grad (WE)", ascii=True):
    face.grad_2nd_mid()

# —————————— 构建对流项 ——————————————
# 构建时跳过物面和远场

for n in tqdm(range(2,cc.N_MAX), desc="Convect", ascii=True):
    for s in range(1,cc.S_MAX+1):
        cell = cc.goto_HALOcell((s,n))
        convect.convect_hybrid(cell)

# —————————— 构建扩散项 ——————————————
# 在进行扩散项构建前,应该先确保各个cell和face存在梯度,再构建粘性的湍流模型参数.
# 请注意,也应该构建虚单元的湍流模型参数和梯度.
# 物理单元 SA 参数和虚单元 SA 参数将分别被构建,构建时跳过物面和远场

for n in tqdm(range(1,cc.N_MAX+1), desc="SA constants", ascii=True):
    for s in range(1,cc.S_MAX+1):
        cell = cc.goto_HALOcell((s,n))
        turbulence.SA_calc_constants(cell)

for s in tqdm(range(1,cc.S_MAX+1), desc="SA constants (ghost)", ascii=True):
    for n in (0,-1):                       
        turbulence.SA_calc_constants(cc.goto_HALOcell((s,n)))
    for n in (cc.N_MAX+1,cc.N_MAX+2):      
        turbulence.SA_calc_constants(cc.goto_HALOcell((s,n)))

for face in tqdm(cc.FaceList_NS, desc="Face diffusion (NS)", ascii=True):
    turbulence.diffusion_2nd_mid_SA(face)

for face in tqdm(cc.FaceList_WE, desc="Face diffusion (WE)", ascii=True):
    turbulence.diffusion_2nd_mid_SA(face)

for n in tqdm(range(2,cc.N_MAX), desc="Diffusion+Source", ascii=True):
    for s in range(1,cc.S_MAX+1):
        cell = cc.goto_HALOcell((s,n))
        turbulence.cell_diffusion(cell)
        turbulence.cell_source(cell)

print("All the Flux is OK.time:",time.time()-starttime)

# —————————— 组装大矩阵 ——————————————

for n in tqdm(range(1,cc.N_MAX+1), desc="Assemble S/T", ascii=True):
    for s in range(1,cc.S_MAX+1):
        formmat.formmat(cc.goto_HALOcell((s,n)))

S, T = formmat.build()
print("S:", S.shape, "nnz =", S.nnz)
print("T:", T.shape, "nnz =", T.nnz, ",number of I =", T.diagonal().sum())

# —————————— 特征值分析 ——————————————
# 这部分没有进度条,对于一个89*180*5维度的矩阵,求解特征值大约需要39s

eigmain.solve_eig(S, T)