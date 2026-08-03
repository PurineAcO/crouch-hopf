import classconfig as cc
import numpy as np
import math

def SA_calc_constants(cell:cc.cell_class):
    """计算Spalart-Allmaras湍流模型引起的有效粘度系数*μeff*"""

    # 计算有效粘度μeff的公式
    cell.mu= cc.mu0 * (cell.T/cc.T0)**1.5* (cc.T0+cc.Ts)/(cell.T+cc.Ts) # 计算分子粘度μ,基于Suthland公式
    cell.chi = cell.rho * cell.miubl / cell.mu                          # 计算χ,修正粘度比
    cell.fv1 = (cell.chi**3)/(cell.chi**3+cc.Cv1**3)                    # 计算阻尼函数fv1
    cell.mu_eff = cell.mu + cell.rho * cell.fv1 * cell.miubl            # 计算有效粘度μeff

    # 计算有效导热系数λeff的公式
    cell.lambda_eff = cell.mu/cc.Pr + (cell.rho * cell.miubl * cell.fv1)/cc.Prt

def face_diffusion(face:cc.face_class):
    """计算面上扩散项矩阵"""
    face.diffusion_2nd_min_SA()


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