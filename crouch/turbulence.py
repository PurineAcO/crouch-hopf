import classconfig as cc
import numpy as np
import grad

def SA_calc_constants(cell:cc.cell_class):
    """计算Spalart-Allmaras湍流模型引起的有效粘度系数*μeff*,\n
    执行这个函数以前请确保`grad.green_gauss_constant`已执行"""

    # 计算有效粘度μeff的公式
    cell.mu= cc.mu0 * (cell.T/cc.T0)**1.5* (cc.T0+cc.Ts)/(cell.T+cc.Ts) # 计算分子粘度μ,基于Suthland公式
    cell.chi = cell.rho * cell.miubl / cell.mu                          # 计算χ,修正粘度比
    cell.fv1 = (cell.chi**3)/(cell.chi**3+cc.Cv1**3)                    # 计算阻尼函数fv1
    cell.mu_eff = cell.mu + cell.rho * cell.fv1 * cell.miubl            # 计算有效粘度μeff

    # 计算有效导热系数λeff的公式
    cell.lambda_eff = cell.mu/cc.Pr + (cell.rho * cell.miubl * cell.fv1)/cc.Prt

def cell_diffusion(cell:cc.cell_class):
    """计算单元扩散项矩阵,在进行这一函数之前,**必须**对**所有单元和面**执行`SA_calc_constants`"""

    # 初始化总的影响矩阵
    influence = [np.zeros((5,5)) for _ in range(13)]
    SA_calc_constants(cell)

    # 计算东面的结果
    directions = ["c", "e", "n", "ne", "s", "se", "ee", "w"]
    results = face_diffusion_WE(cell.east)
    for dire, val in zip(directions, results):
        influence[cc.dic[dire]] += val

    # 计算西面的结果
    directions = ["w","c","nw","n","sw","s","e","ww"]
    results = face_diffusion_WE(cell.west)
    for dire, val in zip(directions, results):
        influence[cc.dic[dire]] += val

    # 计算南面的结果
    ...

    # 计算北面的结果
    ...

def face_diffusion_WE(face:cc.face_class):
    """计算WE面的扩散项"""

    B0 = 2 * (face.ugrad[0]-1/3*(face.ugrad[0]+face.vgrad[1]))*(face.fv1*(4-3*face.fv1))
    B1 = B0 * face.miubl
    B2 = B0 * face.rho
    dic = grad.green_gauss_face_vari_WE(face)

    # 中心网格(6号),以下均以东侧网格为例,西侧网格的相对位置关系也是一致的.
    Dx = np.array([[0,0,0,0,0],
                   [4/3*face.mu_eff*dic["w"][0],-2/3*face.mu_eff*dic["w"][1],B1/2,0,B2/2],
                   [...],
                   [...],
                   [...],])
    Dy = np.array([[0,0,0,0,0],
                   [...],
                   [...],
                   [...],
                   [...]])
    D6 = face.jacobi(Dx,Dy)[0]

    # 东侧网格(7号)
    Dx = np.array([[0,0,0,0,0],
                   [4/3*face.mu_eff*dic["e"][0],-2/3*face.mu_eff*dic["e"][1],B1/2,0,B2/2],
                   [...],
                   [...],
                   [...],])
    Dy = np.array([[0,0,0,0,0],
                   [...],
                   [...],
                   [...],
                   [...]])
    D7 = face.jacobi(Dx,Dy)[0]

    # 北侧网格(2号)
    Dx = np.array([[0,0,0,0,0],
                   [4/3*face.mu_eff*dic["nw"][0],-2/3*face.mu_eff*dic["nw"][1],0,0,0],
                   [...],
                   [...],
                   [...],])
    Dy = np.array([[0,0,0,0,0],
                   [...],
                   [...],
                   [...],
                   [...]])
    D2 = face.jacobi(Dx,Dy)[0]

    # 东北网格(3号)
    Dx = np.array([[0,0,0,0,0],
                   [4/3*face.mu_eff*dic["ne"][0],-2/3*face.mu_eff*dic['ne'][1],0,0,0],
                   [...],
                   [...],
                   [...],])
    Dy = np.array([[0,0,0,0,0],
                   [...],
                   [...],
                   [...],
                   [...]])
    D3 = face.jacobi(Dx,Dy)[0]

    # 南侧网格(10号)
    Dx = np.array([[0,0,0,0,0],
                   [4/3*face.mu_eff*dic['sw'][0],-2/3*face.mu_eff*dic['sw'][1],0,0,0],
                   [...],
                   [...],
                   [...],])
    Dy = np.array([[0,0,0,0,0],
                   [...],
                   [...],
                   [...],
                   [...]])
    D10 = face.jacobi(Dx,Dy)[0]

    # 东南网格(11号)
    Dx = np.array([[0,0,0,0,0],
                   [4/3*face.mu_eff*dic['se'][0],-2/3*face.mu_eff*dic['se'][1],0,0,0],
                   [...],
                   [...],
                   [...],])
    Dy = np.array([[0,0,0,0,0],
                   [...],
                   [...],
                   [...],
                   [...]])
    D11 = face.jacobi(Dx,Dy)[0]

    # 东东网格(8号)
    Dx = np.array([[0,0,0,0,0],
                   [4/3*face.mu_eff*dic['ee'][0],-2/3*face.mu_eff*dic["ee"][1],0,0,0],
                   [...],
                   [...],
                   [...],])
    Dy = np.array([[0,0,0,0,0],
                   [...],
                   [...],
                   [...],
                   [...]])
    D8 = face.jacobi(Dx,Dy)[0]

    # 西侧网格(5号)
    Dx = np.array([[0,0,0,0,0],
                   [4/3*face.mu_eff*dic['ww'][0],-2/3*face.mu_eff*dic['ww'][1],0,0,0],
                   [...],
                   [...],
                   [...],])
    Dy = np.array([[0,0,0,0,0],
                   [...],
                   [...],
                   [...],
                   [...]])
    D5 = face.jacobi(Dx,Dy)[0]

    return D6,D7,D2,D3,D10,D11,D8,D5



# ————————————————————————————————————————————————————————————————————————————————————————————————————
# def form_face_diffusion_1stbounded(face:cc.face_class,cell_1:cc.cell_class,cell_2:cc.cell_class):
#     """根据相邻单元的湍流扩散项计算面上的湍流扩散项`DiffuTurb`.采用一阶中心差分"""
#     face_diff = (cell_1.DiffuTurb + cell_2.DiffuTurb) / 2.0
#     normal = np.array([face.nx, face.ny])
#     face.DiffuTurb = face_diff @ normal

# def form_source_term(cell:cc.cell_class):
#     """计算单元的湍流源项`S`"""
#     # the first term
#     ft2 = cc.Ct3 * math.exp(-cc.Ct4 * cell.chi**2)  # 生产项修正函数ft2
#     fv2 = 1-cell.chi/(1+cell.chi*cell.fv1)          # 涡量修正函数fv2
#     # BUGFIX: 二维涡量 ω = ∂v/∂x − ∂u/∂y,原式取的是 ½(∂u/∂y − ∂v/∂x) = −ω/2,
#     #         再乘 √2 得到 |ω|/√2,比正确的涡量模 S = √(2ΩᵢⱼΩᵢⱼ) = |ω| 小 √2 倍.
#     Omega = cell.vgrad[1] - cell.ugrad[2]           # 计算涡量Omega
#     S = cc.fv3 * abs(Omega)                         # 计算涡量模S
#     nu_tilde = cell.U[5]/cell.U[1]                  # ν̃
#     inv_kd2 = 1.0/(cc.kappa**2 * cell.sad**2)       # 1/(κ²d²)
#     Sbl = S + nu_tilde*inv_kd2*fv2                  # 计算修正涡量S̃
#     # BUGFIX: S̃ 可能为 0 甚至为负(fv2 < 0 时),下方 r = ν̃/(S̃κ²d²) 会除零/发散.
#     #         按 Allmaras 2012 的建议对 S̃ 做下限截断.
#     Sbl = max(Sbl, 1e-10)
#     P = cc.Cb1 * (1-ft2) * Sbl * cell.U[5]          # 计算生成项P
#     # the second term
#     r = min(nu_tilde/Sbl*inv_kd2, cc.rmax)          # 无量纲sad
#     g = r + cc.Cw2 * (r**6 - r)
#     fw = g * ((1+cc.Cw3**6)/(g**6 + cc.Cw3**6))**(1/6) # 壁面阻尼函数fw
#     D = ((cc.Cw1 * fw - cc.Cb1 /(cc.kappa**2) * ft2) *
#          cell.U[1] *(nu_tilde/cell.sad)**2)         # 计算破坏项D
#     # the third term
#     G = cc.Cb2 * cc.inv_sigma * cell.U[1] * float(cell.miublgrad @ cell.miublgrad)
#     # form the final source term
#     cell.S = np.array([0.0,0.0,0.0,0.0,0.0,P-D+G]) * cell.vol