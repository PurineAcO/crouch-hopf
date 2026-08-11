import classconfig as cc
import numpy as np
import math


def wing_boundary(cell:cc.cell_class):
    """处理壁面处的边界条件,要求提供的网格必须是壁面处的"""
    if cell.index[1] != 1 : raise ValueError("Not wing-up cell")
    dx1 = cell.north.north.x - cell.x
    dy1 = cell.north.north.y - cell.y
    dx2 = cell.north.north.north.north.x - cell.north.north.x
    dy2 = cell.north.north.north.north.y - cell.north.north.y
    det = dx1*dy2-dx2*dy1
    C0 = ((dy1-dy2)*cell.south.nx + (dx2-dx1)*cell.south.ny)/det
    C1 = (dy2*cell.south.nx - dx2*cell.south.ny)/det
    C2 = -1 * (dy1*cell.south.nx - dx1*cell.south.ny)/det
    A = np.array([[C0,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,C0,0],[0,0,0,0,1]])
    B = np.array([[C1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,C1,0],[0,0,0,0,0]])
    D = np.array([[C2,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,C2,0],[0,0,0,0,0]])
    cell.form_influence(cc.dic["c"],A)
    cell.form_influence(cc.dic["n"],B)
    cell.form_influence(cc.dic["nn"],D)

def far_boundary(cell:cc.cell_class):
    if cell.index[1] != cc.N_MAX : raise ValueError("Not Far-in cell")
    CT = (math.sqrt(cc.gamma * cc.R))/((cc.gamma-1) * cell.T)
    KR = -1 * (cc.R * (cc.gamma -1) * cell.T)/(cell.rho ** cc.gamma)
    KT = cc.R / (cell.rho ** (cc.gamma -1))
    dx1 = cell.south.south.x - cell.x
    dy1 = cell.south.south.y - cell.y
    dx2 = cell.south.south.south.south.x - cell.south.south.x
    dy2 = cell.south.south.south.south.y - cell.south.south.y
    det = dx1*dy2-dx2*dy1
    C0 = ((dy1-dy2)*cell.north.nx + (dx2-dx1)*cell.north.ny)/det
    C1 = (dy2*cell.north.nx - dx2*cell.north.ny)/det
    C2 = -1 * (dy1*cell.north.nx - dx1*cell.north.ny)/det
    kx = cell.north.nx/(math.sqrt(cell.north.nx**2+cell.north.ny**2))
    ky = cell.north.ny/(math.sqrt(cell.north.nx**2+cell.north.ny**2))
    if cell.north.vn <= 0: # 入流边界
        A = np.array([[0,kx,ky,CT,0],[0,C0*kx,C0*ky,-C0*CT,0],[0,ky,-kx,0,0],[KR,0,0,KT,0],[0,0,0,0,1]])
        B = np.array([[0,0,0,0,0],[0,C1*kx,C1*ky,-C1*CT,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
        D = np.array([[0,0,0,0,0],[0,C2*kx,C2*ky,-C2*CT,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    else: # 出流边界
        A = np.array([[0,C0*kx,C0*ky,C0*CT,0],[0,kx,ky,-CT,0],[0,C0*ky,-C0*kx,0,0],[C0*KR,0,0,C0*KT,0],[0,0,0,0,C0]])
        B = np.array([[0,C1*kx,C1*ky,C1*CT,0],[0,0,0,0,0],[0,C1*ky,-C1*kx,0,0],[C1*KR,0,0,C1*KT,0],[0,0,0,0,C1]])
        D = np.array([[0,C2*kx,C2*ky,C2*CT,0],[0,0,0,0,0],[0,C2*ky,-C2*kx,0,0],[C2*KR,0,0,C2*KT,0],[0,0,0,0,C2]])
    cell.form_influence(cc.dic["c"],A)
    cell.form_influence(cc.dic["s"],B)
    cell.form_influence(cc.dic["ss"],D)