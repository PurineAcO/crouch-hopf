import classconfig as cc
import numpy as np

# 本篇对s,n随体向量的要求是,必须指向北方和东方作为正

def face_convect_mat_4th_mid(face:cc.face_class):
    """依照四阶中心差分格式,构造四阶中心差分对流项行向量,依据论文(3.1.12-13,16-18)"""
    i,j = face.me.index
    mid_4th = [np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),
               np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),
               np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5))]
    if face.direction == 'N':
        G_list = [face.jacobi(cc.CellList[i][dj].F, cc.CellList[i][dj].G)[1] for dj in [j+2, j+1, j, j-1]]
        for pos, g, w in zip([1,3,7,11], G_list, [-1/12, 7/12, 7/12, -1/12]):
            mid_4th[pos-1] = w*g
    elif face.direction == 'S':
        G_list = [face.jacobi(cc.CellList[i][dj].F, cc.CellList[i][dj].G)[1] for dj in [j-2, j-1, j, j+1]]
        for pos, g, w in zip([13,11,7,3], G_list, [-1/12, 7/12, 7/12, -1/12]):
            mid_4th[pos-1] = w*g*(-1)
    elif face.direction == 'E':
        F_list = [face.jacobi(cc.CellList[di][j].F, cc.CellList[di][j].G)[0] for di in [i+2, i+1, i, i-1]]
        for pos, g, w in zip([9,8,7,6], F_list, [-1/12, 7/12, 7/12, -1/12]):
            mid_4th[pos-1] = w*g
    elif face.direction == 'W':
        F_list = [face.jacobi(cc.CellList[di][j].F, cc.CellList[di][j].G)[0] for di in [i-2, i-1, i, i+1]]
        for pos, g, w in zip([5,6,7,8], F_list, [-1/12, 7/12, 7/12, -1/12]):
            mid_4th[pos-1] = w*g*(-1)
    else:raise ValueError("incorrect direction")

    return mid_4th


def face_convect_mat_3rd_upwind(face:cc.face_class):
    """依照三阶迎风格式,构造三阶迎风格式对流项行向量,依据论文(3.1.12,14-18)"""
    i,j = face.me.index
    upwind_3rd = [np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),
                  np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),
                  np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5))]
    v_n = face.jacobi(face.u,face.v)[0]
    if face.direction == 'N':
        if v_n <= 0:
            G_list = [face.jacobi(cc.CellList[i][dj].F, cc.CellList[i][dj].G)[1] for dj in [j+2, j+1, j]]
            for pos,g,w in zip([1,3,7],G_list,[-1/6,5/6,1/3]):
                upwind_3rd[pos-1] = w*g
        else:
            G_list = [face.jacobi(cc.CellList[i][dj].F, cc.CellList[i][dj].G)[1] for dj in [j-1,j,j+1]]
            for pos,g,w in zip([11,7,3],G_list,[-1/6,5/6,1/3]):
                upwind_3rd[pos-1] = w*g
    elif face.direction == 'S':
        if v_n >= 0:
            G_list = [face.jacobi(cc.CellList[i][dj].F, cc.CellList[i][dj].G)[1] for dj in [j-2,j-1,j]]
            for pos,g,w in zip([13,11,7],G_list,[-1/6,5/6,1/3]):
                upwind_3rd[pos-1] = w*g*(-1)
        else:
            G_list = [face.jacobi(cc.CellList[i][dj].F, cc.CellList[i][dj].G)[1] for dj in [j+1,j,j-1]]
            for pos,g,w in zip([3,7,11],G_list,[-1/6,5/6,1/3]):
                upwind_3rd[pos-1] = w*g*(-1)
    elif face.direction == 'E':
        if v_n <= 0:
            F_list = [face.jacobi(cc.CellList[di][j].F, cc.CellList[di][j].G)[0] for di in [i+2,i+1,i]]
            for pos,g,w in zip([9,8,7],F_list,[-1/6,5/6,1/3]):
                upwind_3rd[pos-1] = w*g
        else:
            F_list= [face.jacobi(cc.CellList[di][j].F, cc.CellList[di][j].G)[0] for di in [i-1,i,i+1]]
            for pos,g,w in zip([6,7,8],F_list,[-1/6,5/6,1/3]):
                upwind_3rd[pos-1] = w*g
    elif face.direction == 'W':
        if v_n >= 0:
            F_list = [face.jacobi(cc.CellList[di][j].F, cc.CellList[di][j].G)[0] for di in [i-2,i-1,i]]
            for pos,g,w in zip([5,6,7],F_list,[-1/6,5/6,1/3]):
                upwind_3rd[pos-1] = w*g*(-1)
        else:
            F_list = [face.jacobi(cc.CellList[di][j].F, cc.CellList[di][j].G)[0] for di in [i+1,i,i-1]]
            for pos,g,w in zip([8,7,6],F_list,[-1/6,5/6,1/3]):
                upwind_3rd[pos-1] = w*g*(-1)
    else:raise ValueError("incorrect direction")

    return upwind_3rd

def convect_hybrid(cell:cc.cell_class):
    """混合格式,依据论文(3.1.19)"""
    for i in range(4):
        face = cell.face[i]
        upwind = face_convect_mat_3rd_upwind(face)
        mid = face_convect_mat_4th_mid(face)
        for j in range(13):
            cell.form_influence(j, cc.alpha_H*upwind[j]+(1-cc.alpha_H)*mid[j])