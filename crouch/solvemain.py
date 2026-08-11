import classconfig as cc
import readrans
import boundary
import grad
import convect
import turbulence

# —————————— 阅读RANS数据 ——————————————
# 这一步之后,应该在cc建立起了cc.NMAX 和cc.SMAX,对于每个cell都应该建立了无粘通量F和G.
# cell 是按照二维数组结构化存储,但是face是摊开存储的,每个cell同别的face和cell链式连接.
# face应该建立起了面上物理量和链式连接关系.
# TODO: 日后可能提供C-Block网格的接口,因此以下函数内部均使用链式访问进行操作.

readrans.read_rans("ransdata.txt","edge.txt")

# —————————— 计算边界条件 ——————————————

for s in range(1,cc.S_MAX+1):
    cell_far = cc.goto_HALOcell((s,cc.N_MAX))
    cell_wing = cc.goto_HALOcell((s,1))
    boundary.far_boundary(cell_far)
    boundary.wing_boundary(cell_wing)

# —————————— 计算梯度 ——————————————

for n in range(1,cc.N_MAX+1):
    for s in range(1,cc.S_MAX+1):
        cell = cc.goto_HALOcell((s,n))
        grad.green_gauss_from_JST(cell,cell.north,cell.south,cell.east,cell.west)

for face in cc.FaceList_NS:
    face.grad_2nd_mid()

for face in cc.FaceList_WE:
    face.grad_2nd_mid()

# —————————— 构建对流项 ——————————————
# 构建时跳过物面和远场

for n in range(2,cc.N_MAX):
    for s in range(1,cc.S_MAX+1):
        cell = cc.goto_HALOcell((s,n))
        convect.convect_hybrid(cell)

# —————————— 构建扩散项 ——————————————
# 在进行扩散项构建前,应该先确保各个cell和face存在梯度,再构建粘性的湍流模型参数.
# 请注意,也应该构建虚单元的湍流模型参数和梯度.
# 物理单元 SA 参数和虚单元 SA 参数将分别被构建,构建时跳过物面和远场

for n in range(1,cc.N_MAX+1):
    for s in range(1,cc.S_MAX+1):
        cell = cc.goto_HALOcell((s,n))
        turbulence.SA_calc_constants(cell)

for s in range(1,cc.S_MAX+1):
    for n in (0,-1):                       
        turbulence.SA_calc_constants(cc.goto_HALOcell((s,n)))
    for n in (cc.N_MAX+1,cc.N_MAX+2):      
        turbulence.SA_calc_constants(cc.goto_HALOcell((s,n)))

for face in cc.FaceList_NS:
    turbulence.diffusion_2nd_mid_SA(face)

for face in cc.FaceList_WE:
    turbulence.diffusion_2nd_mid_SA(face)

for n in range(2,cc.N_MAX):
    for s in range(1,cc.S_MAX+1):
        cell = cc.goto_HALOcell((s,n))
        turbulence.cell_diffusion(cell)
        turbulence.cell_source(cell)