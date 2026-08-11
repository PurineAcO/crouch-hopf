import classconfig as cc
import numpy as np
import grad
import math

def SA_calc_constants(cell:cc.cell_class):
    """计算Spalart-Allmaras湍流模型引起的有效粘度系数*μeff*,\n
    执行这个函数以前请确保`grad.green_gauss_from_JST`已执行"""

    # 计算有效粘度μeff的公式
    cell.mu= cc.mu0 * (cell.T/cc.T0)**1.5* (cc.T0+cc.Ts)/(cell.T+cc.Ts) # 计算分子粘度μ,基于Suthland公式
    cell.chi = cell.rho * cell.miubl / cell.mu                          # 计算χ,修正粘度比
    cell.fv1 = (cell.chi**3)/(cell.chi**3+cc.Cv1**3)                    # 计算阻尼函数fv1
    cell.mu_eff = cell.mu + cell.rho * cell.fv1 * cell.miubl            # 计算有效粘度μeff

    # 计算有效导热系数λeff的公式
    cell.lambda_eff = cell.mu/cc.Pr + (cell.rho * cell.miubl * cell.fv1)/cc.Prt

    # 计算源项部分参数:
    cell.ft2 = cc.Ct3 * math.exp(-cc.Ct4 * (cell.chi**2))
    cell.fv2 = 1 - (cell.chi)/(1+cell.chi * cell.fv1)
    cell.Omega = abs(cell.ugrad[1] - cell.vgrad[0])
    cell.Sbl = cell.Omega + cell.fv2 * cell.miubl /((cc.kappa*cell.sad)**2)
    cell.r = min(cell.miubl/(cell.Sbl * cc.kappa**2 * cell.sad**2),cc.rmax)
    cell.g = cell.r + cc.Cw2 * (cell.r**6 -cell.r)
    cell.fw = cell.g * ((1+cc.Cw3**6)/(cell.g**6+cc.Cw3**6))**(1/6)
    cell.S = math.sqrt(2*cell.ugrad[0]**2 + 2*cell.vgrad[1]**2 + (cell.ugrad[1]+cell.vgrad[0])**2)


def diffusion_2nd_mid_SA(face:cc.face_class):
    """对面上的湍流字典进行二阶中心插值并构建切应力"""
    face.mu_eff = (face.me.mu_eff+face.nei.mu_eff)/2
    face.lambda_eff = (face.me.lambda_eff+face.nei.lambda_eff)/2
    face.chi = (face.me.chi+face.nei.chi)/2
    face.fv1 = (face.me.fv1+face.nei.fv1)/2
    face.mu = (face.me.mu+face.nei.mu)/2
    face.tauxx = face.mu_eff * (face.ugrad[0]-1/3*(face.ugrad[0]+face.vgrad[1]))
    face.tauxy = face.mu_eff * (face.ugrad[1]+face.vgrad[0])
    face.tauyy = face.mu_eff * (face.vgrad[1]-1/3*(face.ugrad[0]+face.vgrad[1]))


def cell_diffusion(cell:cc.cell_class):
    """计算单元扩散项矩阵,在进行这一函数之前,**必须**对**所有单元和面**执行`SA_calc_constants`"""

    # 初始化总的影响矩阵
    influence = [np.zeros((5,5)) for _ in range(13)]

    # 计算东面的结果
    directions = ["c", "e", "n", "ne", "s", "se", "ee", "w"]
    results = face_diffusion(cell.east)
    for dire, val in zip(directions, results):
        influence[cc.dic[dire]] += val

    # 计算西面的结果
    directions = ["w","c","nw","n","sw","s","e","ww"]
    results = face_diffusion(cell.west)
    for dire, val in zip(directions, results):
        influence[cc.dic[dire]] += val

    # 计算南面的结果
    directions = ["s","c","se","sw","ss","w","n","e"]
    results = face_diffusion(cell.south)
    for dire, val in zip(directions, results):
        influence[cc.dic[dire]] += val

    # 计算北面的结果
    directions = ["c","n","e","w","s","nw","nn","ne"]
    results = face_diffusion(cell.north)
    for dire, val in zip(directions, results):
        influence[cc.dic[dire]] += val

    for i in range(13):
        cell.form_influence(i, -influence[i]/cell.vol)

def face_diffusion(face:cc.face_class):
    """计算面上的湍流扩散项"""

    F0 = (face.fv1*(4-3*face.fv1))
    B0 = 2 * (face.ugrad[0]-1/3*(face.ugrad[0]+face.vgrad[1]))* F0
    C0 = 2 * (face.ugrad[1] + face.vgrad[0]) * F0
    E0 = 2 * (face.vgrad[1]-1/3*(face.ugrad[0]+face.vgrad[1]))* F0
    G0 = face.Tgrad[0] * F0 / cc.Prt 
    B1 = B0 * face.miubl ; C1 = C0 * face.miubl ; E1 = E0 * face.miubl ; G1 = G0 * face.miubl
    B2 = B0 * face.rho ; C2 = C0 * face.rho ; E2 = E0 * face.rho ; G2 = G0 * face.rho
    if face.direction == "WE":
        dic = grad.green_gauss_face_vari(face)
        jacobiflag = 0
    elif face.direction == "NS":
        dic = grad.green_gauss_face_vari(face)
        jacobiflag = 1
    else:
        raise ValueError("face.direction must be WE or NS")

    # 中心网格(6号),以下均以东侧网格为例,西侧网格的相对位置关系也是一致的.//分割线后考虑北侧面
    dire = dic["w"] if face.direction == "WE" else dic["s"]
    Dx = np.array([[0,0,0,0,0],
                   [B1/2,4/3*face.mu_eff*dire[0],-2/3*face.mu_eff*dire[1],0,B2/2],
                   [C1/2,face.mu_eff*dire[1],face.mu_eff*dire[0],0,C2/2],
                   [face.u*B1/2+face.v*C1/2+G1/2,face.tauxx/2+4/3*face.u*face.mu_eff*dire[0]+face.v*face.mu_eff*dire[1],
                    face.tauxy/2-2/3*face.u*face.mu_eff*dire[1]+face.v*face.mu_eff*dire[0],face.lambda_eff*dire[0],
                    face.u*B2/2+face.v*C2/2+G2/2],
                   [1/2*face.miublgrad[0]*face.miubl*cc.inv_sigma,0,0,0,
                    1/2*face.miublgrad[0]*face.rho*cc.inv_sigma+face.mu_eff*dire[0]*cc.inv_sigma]])
    Dy = np.array([[0,0,0,0,0],
                   [C1/2,face.mu_eff*dire[1],face.mu_eff*dire[0],0,C2/2],
                   [E1/2,-2/3*face.mu_eff*dire[0],4/3*face.mu_eff*dire[1],0,E2/2],
                   [face.u*C1/2+face.v*E1/2+G1/2,face.tauxy/2+face.u*face.mu_eff*dire[1]-2/3*face.v*face.mu_eff*dire[0],
                   face.tauyy/2+face.u*face.mu_eff*dire[1]+4/3*face.v*face.mu_eff*dire[0],face.lambda_eff*dire[1],
                   face.u*C2/2+face.v*E2/2+G2/2],
                   [1/2*face.miublgrad[0]*face.miubl*cc.inv_sigma,0,0,0,
                    1/2*face.miublgrad[0]*face.rho*cc.inv_sigma+face.mu_eff*dire[0]*cc.inv_sigma]])
    D6 = face.jacobi(Dx,Dy)[jacobiflag]

    # 东侧网格(7号) // 北侧网格(2)
    dire = dic['e'] if face.direction == "WE" else dic['n']
    Dx = np.array([[0,0,0,0,0],
                   [B1/2,4/3*face.mu_eff*dire[0],-2/3*face.mu_eff*dire[1],0,B2/2],
                   [C1/2,face.mu_eff*dire[1],face.mu_eff*dire[0],0,C2/2],
                   [face.u*B1/2+face.v*C1/2+G1/2,face.tauxx/2+4/3*face.u*face.mu_eff*dire[0]+face.v*face.mu_eff*dire[1],
                    face.tauxy/2-2/3*face.u*face.mu_eff*dire[1]+face.v*face.mu_eff*dire[0],face.lambda_eff*dire[0],
                    face.u*B2/2+face.v*C2/2+G2/2],
                   [1/2*face.miublgrad[0]*face.miubl*cc.inv_sigma,0,0,0,
                    1/2*face.miublgrad[0]*face.rho*cc.inv_sigma+face.mu_eff*dire[0]*cc.inv_sigma]])
    Dy = np.array([[0,0,0,0,0],
                   [C1/2,face.mu_eff*dire[1],face.mu_eff*dire[0],0,C2/2],
                   [E1/2,-2/3*face.mu_eff*dire[0],4/3*face.mu_eff*dire[1],0,E2/2],
                   [face.u*C1/2+face.v*E1/2+G1/2,face.tauxy/2+face.u*face.mu_eff*dire[1]-2/3*face.v*face.mu_eff*dire[0],
                   face.tauyy/2+face.u*face.mu_eff*dire[1]+4/3*face.v*face.mu_eff*dire[0],face.lambda_eff*dire[1],
                   face.u*C2/2+face.v*E2/2+G2/2],
                   [1/2*face.miublgrad[1]*face.miubl*cc.inv_sigma,0,0,0,
                    1/2*face.miublgrad[1]*face.rho*cc.inv_sigma+face.mu_eff*dire[1]*cc.inv_sigma]])
    D7 = face.jacobi(Dx,Dy)[jacobiflag]

    # 北侧网格(2号) // 东侧网格(7)
    dire = dic["nw"] if face.direction == "WE" else dic["se"]
    Dx = np.array([[0,0,0,0,0],
                   [0,4/3*face.mu_eff*dire[0],-2/3*face.mu_eff*dire[1],0,0],
                   [0,face.mu_eff*dire[1],face.mu_eff*dire[0],0,0],
                   [0,4/3*face.u*face.mu_eff*dire[0]+face.v*face.mu_eff*dire[1],
                    -2/3*face.u*face.mu_eff*dire[1]+face.v*face.mu_eff*dire[0],face.lambda_eff*dire[0],0],
                   [0,0,0,0,face.mu_eff*dire[0]*cc.inv_sigma]])
    Dy = np.array([[0,0,0,0,0],
                   [0,face.mu_eff*dire[1],face.mu_eff*dire[0],0,0],
                   [0,-2/3*face.mu_eff*dire[0],4/3*face.mu_eff*dire[1],0,0],
                   [0,face.u*face.mu_eff*dire[1]-2/3*face.v*face.mu_eff*dire[0],
                   face.u*face.mu_eff*dire[1]+4/3*face.v*face.mu_eff*dire[0],face.lambda_eff*dire[1],0],
                   [0,0,0,0,face.mu_eff*dire[1]*cc.inv_sigma]])
    D2 = face.jacobi(Dx,Dy)[jacobiflag]

    # 东北网格(3号) // 西侧网格(5)
    dire = dic["ne"] if face.direction == "WE" else dic["sw"]
    Dx = np.array([[0,0,0,0,0],
                   [0,4/3*face.mu_eff*dire[0],-2/3*face.mu_eff*dire[1],0,0],
                   [0,face.mu_eff*dire[1],face.mu_eff*dire[0],0,0],
                   [0,4/3*face.u*face.mu_eff*dire[0]+face.v*face.mu_eff*dire[1],
                    -2/3*face.u*face.mu_eff*dire[1]+face.v*face.mu_eff*dire[0],face.lambda_eff*dire[0],0],
                   [0,0,0,0,face.mu_eff*dire[0]*cc.inv_sigma]])
    Dy = np.array([[0,0,0,0,0],
                   [0,face.mu_eff*dire[1],face.mu_eff*dire[0],0,0],
                   [0,-2/3*face.mu_eff*dire[0],4/3*face.mu_eff*dire[1],0,0],
                   [0,face.u*face.mu_eff*dire[1]-2/3*face.v*face.mu_eff*dire[0],
                   face.u*face.mu_eff*dire[1]+4/3*face.v*face.mu_eff*dire[0],face.lambda_eff*dire[1],0],
                   [0,0,0,0,face.mu_eff*dire[1]*cc.inv_sigma]])
    D3 = face.jacobi(Dx,Dy)[jacobiflag]

    # 南侧网格(10号) // 南侧网格(10)
    dire = dic["sw"] if face.direction == "WE" else dic["ss"]
    Dx = np.array([[0,0,0,0,0],
                   [0,4/3*face.mu_eff*dire[0],-2/3*face.mu_eff*dire[1],0,0],
                   [0,face.mu_eff*dire[1],face.mu_eff*dire[0],0,0],
                   [0,4/3*face.u*face.mu_eff*dire[0]+face.v*face.mu_eff*dire[1],
                    -2/3*face.u*face.mu_eff*dire[1]+face.v*face.mu_eff*dire[0],face.lambda_eff*dire[0],0],
                   [0,0,0,0,face.mu_eff*dire[0]*cc.inv_sigma]])
    Dy = np.array([[0,0,0,0,0],
                   [0,face.mu_eff*dire[1],face.mu_eff*dire[0],0,0],
                   [0,-2/3*face.mu_eff*dire[0],4/3*face.mu_eff*dire[1],0,0],
                   [0,face.u*face.mu_eff*dire[1]-2/3*face.v*face.mu_eff*dire[0],
                   face.u*face.mu_eff*dire[1]+4/3*face.v*face.mu_eff*dire[0],face.lambda_eff*dire[1],0],
                   [0,0,0,0,face.mu_eff*dire[1]*cc.inv_sigma]])
    D10 = face.jacobi(Dx,Dy)[jacobiflag]

    # 东南网格(11号) // 西北网格(1)
    dire = dic["se"] if face.direction == "WE" else dic["nw"]
    Dx = np.array([[0,0,0,0,0],
                   [0,4/3*face.mu_eff*dire[0],-2/3*face.mu_eff*dire[1],0,0],
                   [0,face.mu_eff*dire[1],face.mu_eff*dire[0],0,0],
                   [0,4/3*face.u*face.mu_eff*dire[0]+face.v*face.mu_eff*dire[1],
                    -2/3*face.u*face.mu_eff*dire[1]+face.v*face.mu_eff*dire[0],face.lambda_eff*dire[0],0],
                   [0,0,0,0,face.mu_eff*dire[0]*cc.inv_sigma]])
    Dy = np.array([[0,0,0,0,0],
                   [0,face.mu_eff*dire[1],face.mu_eff*dire[0],0,0],
                   [0,-2/3*face.mu_eff*dire[0],4/3*face.mu_eff*dire[1],0,0],
                   [0,face.u*face.mu_eff*dire[1]-2/3*face.v*face.mu_eff*dire[0],
                   face.u*face.mu_eff*dire[1]+4/3*face.v*face.mu_eff*dire[0],face.lambda_eff*dire[1],0],
                   [0,0,0,0,face.mu_eff*dire[1]*cc.inv_sigma]])
    D11 = face.jacobi(Dx,Dy)[jacobiflag]

    # 东东网格(8号)  // 北北网格(0)
    dire = dic["ee"] if face.direction == "WE" else dic["nn"]
    Dx = np.array([[0,0,0,0,0],
                   [0,4/3*face.mu_eff*dire[0],-2/3*face.mu_eff*dire[1],0,0],
                   [0,face.mu_eff*dire[1],face.mu_eff*dire[0],0,0],
                   [0,4/3*face.u*face.mu_eff*dire[0]+face.v*face.mu_eff*dire[1],
                    -2/3*face.u*face.mu_eff*dire[1]+face.v*face.mu_eff*dire[0],face.lambda_eff*dire[0],0],
                   [0,0,0,0,face.mu_eff*dire[0]*cc.inv_sigma]])
    Dy = np.array([[0,0,0,0,0],
                   [0,face.mu_eff*dire[1],face.mu_eff*dire[0],0,0],
                   [0,-2/3*face.mu_eff*dire[0],4/3*face.mu_eff*dire[1],0,0],
                   [0,face.u*face.mu_eff*dire[1]-2/3*face.v*face.mu_eff*dire[0],
                   face.u*face.mu_eff*dire[1]+4/3*face.v*face.mu_eff*dire[0],face.lambda_eff*dire[1],0],
                   [0,0,0,0,face.mu_eff*dire[1]*cc.inv_sigma]])
    D8 = face.jacobi(Dx,Dy)[jacobiflag]

    # 西侧网格(5号) // 东北网格(3)
    dire = dic["ww"] if face.direction == "WE" else dic["ne"]
    Dx = np.array([[0,0,0,0,0],
                   [0,4/3*face.mu_eff*dire[0],-2/3*face.mu_eff*dire[1],0,0],
                   [0,face.mu_eff*dire[1],face.mu_eff*dire[0],0,0],
                   [0,4/3*face.u*face.mu_eff*dire[0]+face.v*face.mu_eff*dire[1],
                    -2/3*face.u*face.mu_eff*dire[1]+face.v*face.mu_eff*dire[0],face.lambda_eff*dire[0],0],
                   [0,0,0,0,face.mu_eff*dire[0]*cc.inv_sigma]])
    Dy = np.array([[0,0,0,0,0],
                   [0,face.mu_eff*dire[1],face.mu_eff*dire[0],0,0],
                   [0,-2/3*face.mu_eff*dire[0],4/3*face.mu_eff*dire[1],0,0],
                   [0,face.u*face.mu_eff*dire[1]-2/3*face.v*face.mu_eff*dire[0],
                   face.u*face.mu_eff*dire[1]+4/3*face.v*face.mu_eff*dire[0],face.lambda_eff*dire[1],0],
                   [0,0,0,0,face.mu_eff*dire[1]*cc.inv_sigma]])
    D5 = face.jacobi(Dx,Dy)[jacobiflag]

    return D6,D7,D2,D3,D10,D11,D8,D5

def cell_source(cell:cc.cell_class):
    """邢程单元上的源项矩阵,直接构建在了`cell.influence`\n
    *这一段十分繁复,作者于2026年8月5日推了一整天,最终发现了原文的3处错误.*"""

    influent = [None,None,None,None,None]  # 顺序:c,n,s,e,w

    # 用到的几个导数项
    dfv2 = (3*cell.chi*cell.fv1*(1-cell.fv1) - 1)/((1+cell.chi*cell.fv1)**2)
    dft2 = -2 * cell.chi * cc.Ct4 * cell.ft2
    dfw = cell.fw/cell.g * (cc.Cw3**6)/(cell.g**6 + cc.Cw3**6) * (1-cc.Cw2+6*cc.Cw2*cell.r**5)

    # 恒定部分,本家网格承担了密度、温度、湍流粘度的全部内容和u,v的部分内容
    R1 = cc.Cb1 * (1- cell.ft2 - cell.chi * dft2 )
    R2 = cc.Cb1 * cell.r * (cell.chi * dfv2 *(1-cell.ft2) + cell.ft2 + cell.chi * dft2)
    R3 = -cc.Cw1 * cc.kappa**2 * cell.r * (cell.fw - dfw * dfv2 * cell.r**2 * cell.chi)
    R4 = cc.Cb2 * cc.inv_sigma * (cell.miublgrad[0]**2 + cell.miublgrad[1]**2)
    R5 = -cc.C5 * cell.miubl**2 * cell.S**2 /(cc.gamma * cc.R * cell.T)
    R = (R1+R2+R3) * cell.Sbl * cell.miubl + R4 + R5

    M1 = cc.Cb1 * (1- cell.ft2 - cell.chi * dft2)
    M2 = cc.Cb1 * cell.r * ((cell.fv2 + cell.chi * dfv2)*(1-cell.ft2) + 2* cell.ft2 + cell.chi * dft2)
    M3 = -cc.Cw1 * cc.kappa**2 * cell.r * (2*cell.fw + cell.r * dfw - cell.r**2 * dfw *(cell.fv2 + cell.chi * dfv2))
    M4 = -2* cc.C5 * cell.rho *cell.miubl *cell.S**2 /(cc.gamma * cc.R * cell.T)
    M0 = (M1+M2+M3) * cell.Sbl * cell.rho + M4

    T = cc.C5 * cell.rho * cell.miubl**2 * cell.S**2 /(cc.gamma * cc.R * cell.T**2)

    # 梯度部分常数
    O1 = cell.rho *cell.miubl /cell.Omega * (cc.Cb1*(1-cell.ft2) + cc.Cw1*cc.kappa**2*cell.r**2*dfw) * (cell.ugrad[1]-cell.vgrad[0])
    O2 = 2 * cc.Cb2 * cc.inv_sigma * cell.rho
    O3 = -2 * cc.C5 * cell.rho *cell.miubl**2 /(cc.gamma * cc.R * cell.T)
    O4 = O3 * (cell.ugrad[1] + cell.vgrad[0])
    dic = grad.green_gauss_cell_vari(cell)

    # 所有网格的源项区
    directions = ["c","n","s","e","w"]
    for dire in directions:
        U = O1 * dic[dire][1] + O4 * dic[dire][1] + O3 * 2 * cell.ugrad[0]*dic[dire][0]
        V = -O1 * dic[dire][0] + O4 * dic[dire][0] + O3 * 2 * cell.vgrad[1]*dic[dire][1]
        M = O2 * (cell.miublgrad[0] * dic[dire][0] + cell.miublgrad[1] * dic[dire][1])
        if dire == "c":
            cell.form_influence(cc.dic["c"], -np.array([R,U,V,T,M+M0]))
        else:
            cell.form_influence(cc.dic[dire], -np.array([0,U,V,0,M]))
    
