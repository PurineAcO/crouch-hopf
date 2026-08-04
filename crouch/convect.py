import classconfig as cc
import numpy as np

# 本篇对s,n随体向量的要求是,必须指向北方和东方作为正.

def face_convect_mat_4th_mid(cell:cc.cell_class):
    """四阶中心差分格式"""
    influence = [np.zeros((5,5)) for _ in range(13)]
    influence[cc.dic['nn']] = 1/12*cell.north.north.north.north.convect_jacobi()[1]
    influence[cc.dic['n']] = -2/3*cell.north.north.convect_jacobi()[1]
    influence[cc.dic['s']] = 2/3*cell.south.south.convect_jacobi()[1]
    influence[cc.dic['ss']] = -1/12*cell.south.south.south.south.convect_jacobi()[1]
    influence[cc.dic['ee']] = 1/12*cell.east.east.east.east.convect_jacobi()[0]
    influence[cc.dic['e']] = -2/3*cell.east.east.convect_jacobi()[0]
    influence[cc.dic['w']] = 2/3*cell.west.west.convect_jacobi()[0]
    influence[cc.dic['ww']] = -1/12*cell.west.west.west.west.convect_jacobi()[0]
    return influence

def face_convect_mat_3rd_upwind(cell:cc.cell_class):
    """三阶迎风格式"""
    influence = [np.zeros((5,5)) for _ in range(13)]

    if cell.north.vn <= 0:
        influence[cc.dic['nn']] += -1/6*cell.north.north.north.north.convect_jacobi()[1]
        influence[cc.dic['n']] += 5/6*cell.north.north.convect_jacobi()[1]
        influence[cc.dic['c']] += 1/3*cell.convect_jacobi()[1]
    else:
        influence[cc.dic['s']] += -1/6*cell.south.south.convect_jacobi()[1]
        influence[cc.dic['c']] += 5/6*cell.convect_jacobi()[1]
        influence[cc.dic['n']] += 1/3*cell.north.north.convect_jacobi()[1]

    if cell.south.vn >= 0:
        influence[cc.dic['ss']] += 1/6*cell.south.south.south.south.convect_jacobi()[1]
        influence[cc.dic['s']] += -5/6*cell.south.south.convect_jacobi()[1]
        influence[cc.dic['c']] += -1/3*cell.convect_jacobi()[1]
    else:
        influence[cc.dic['n']] += 1/6*cell.north.north.convect_jacobi()[1]
        influence[cc.dic['c']] += -5/6*cell.convect_jacobi()[1]
        influence[cc.dic['s']] += -1/3*cell.south.south.convect_jacobi()[1]

    if cell.east.vn <= 0:
        influence[cc.dic['ee']] += -1/6*cell.east.east.east.east.convect_jacobi()[0]
        influence[cc.dic['e']] += 5/6*cell.east.east.convect_jacobi()[0]
        influence[cc.dic['c']] += 1/3*cell.convect_jacobi()[0]
    else:
        influence[cc.dic['w']] += -1/6*cell.west.west.convect_jacobi()[0]
        influence[cc.dic['c']] += 5/6*cell.convect_jacobi()[0]
        influence[cc.dic['e']] += 1/3*cell.east.east.convect_jacobi()[0]

    if cell.west.vn >= 0:
        influence[cc.dic['ww']] += 1/6*cell.west.west.west.west.convect_jacobi()[0]
        influence[cc.dic['w']] += -5/6*cell.west.west.convect_jacobi()[0]
        influence[cc.dic['c']] += -1/3*cell.convect_jacobi()[0]
    else:
        influence[cc.dic['e']] += 1/6*cell.east.east.convect_jacobi()[0]
        influence[cc.dic['c']] += -5/6*cell.convect_jacobi()[0]
        influence[cc.dic['w']] += -1/3*cell.west.west.convect_jacobi()[0]

    return influence

def viscous_convect_1st_upwind(cell:cc.cell_class):
    """一阶迎风格式"""
    influence = [np.zeros((5,5)) for _ in range(13)]

    if cell.north.vn <= 0:
        influence[cc.dic['n']] = cell.north.north.viscous_convect_jacobi()[1]
    else:
        influence[cc.dic['c']] = cell.convect_jacobi()[1]

    if cell.south.vn >= 0:
        influence[cc.dic['s']] = cell.south.south.viscous_convect_jacobi()[1]
    else:
        influence[cc.dic['c']] = cell.convect_jacobi()[1]

    if cell.east.vn <= 0:
        influence[cc.dic['e']] = cell.east.east.viscous_convect_jacobi()[0]
    else:
        influence[cc.dic['c']] = cell.convect_jacobi()[0]

    if cell.west.vn >= 0:
        influence[cc.dic['w']] = cell.west.west.viscous_convect_jacobi()[0]
    else:
        influence[cc.dic['c']] = cell.convect_jacobi()[0]

    return influence

def convect_hybrid(cell:cc.cell_class):
    """邢程对流项"""
    upwind = face_convect_mat_3rd_upwind(cell)
    mid = face_convect_mat_4th_mid(cell)
    viscous = viscous_convect_1st_upwind(cell)
    for j in range(13):
        cell.form_influence(j, cc.alpha_H*upwind[j]+(1-cc.alpha_H)*mid[j]+viscous[j])