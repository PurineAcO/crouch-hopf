import classconfig as cc
import numpy as np

# 本篇对s,n随体向量的要求是,必须指向北方和东方作为正.
# BUG:Jacobi 的位置错误了!!!

def convect_sum_jacobi(cell_me:cc.cell_class,cell_nei:cc.cell_class):
    """将`cell_nei`根据`cell_me`进行jacobi"""
    return cell_me.jacobi(cell_nei.F,cell_nei.G)

def viscous_convect_sum_jacobi(cell_me:cc.cell_class,cell_nei:cc.cell_class):
    return cell_me.jacobi(cell_nei.viscous_convect_vec()[0],cell_nei.viscous_convect_vec()[1])

def face_convect_mat_4th_mid(cell:cc.cell_class):
    """四阶中心差分格式"""
    influence = [np.zeros((5,5)) for _ in range(13)]
    influence[cc.dic['nn']] = 1/12*convect_sum_jacobi(cell,cell.north.north.north.north)[1]
    influence[cc.dic['n']] = -2/3*convect_sum_jacobi(cell,cell.north.north)[1]
    influence[cc.dic['s']] = 2/3*convect_sum_jacobi(cell,cell.south.south)[1]
    influence[cc.dic['ss']] = -1/12*convect_sum_jacobi(cell,cell.south.south.south.south)[1]
    influence[cc.dic['ee']] = 1/12*convect_sum_jacobi(cell,cell.east.east.east.east)[0]
    influence[cc.dic['e']] = -2/3*convect_sum_jacobi(cell,cell.east.east)[0]
    influence[cc.dic['w']] = 2/3*convect_sum_jacobi(cell,cell.west.west)[0]
    influence[cc.dic['ww']] = -1/12*convect_sum_jacobi(cell,cell.west.west.west.west)[0]
    return influence

def face_convect_mat_3rd_upwind(cell:cc.cell_class):
    """三阶迎风格式"""
    influence = [np.zeros((5,5)) for _ in range(13)]

    if cell.north.vn <= 0:
        influence[cc.dic['nn']] += -1/6*convect_sum_jacobi(cell,cell.north.north.north.north)[1]
        influence[cc.dic['n']] += 5/6*convect_sum_jacobi(cell,cell.north.north)[1]
        influence[cc.dic['c']] += 1/3*convect_sum_jacobi(cell,cell)[1]
    else:
        influence[cc.dic['s']] += -1/6*convect_sum_jacobi(cell,cell.south.south)[1]
        influence[cc.dic['c']] += 5/6*convect_sum_jacobi(cell,cell)[1]
        influence[cc.dic['n']] += 1/3*convect_sum_jacobi(cell,cell.north.north)[1]

    if cell.south.vn >= 0:
        influence[cc.dic['ss']] += 1/6*convect_sum_jacobi(cell,cell.south.south.south.south)[1]
        influence[cc.dic['s']] += -5/6*convect_sum_jacobi(cell,cell.south.south)[1]
        influence[cc.dic['c']] += -1/3*convect_sum_jacobi(cell,cell)[1]
    else:
        influence[cc.dic['n']] += 1/6*convect_sum_jacobi(cell,cell.north.north)[1]
        influence[cc.dic['c']] += -5/6*convect_sum_jacobi(cell,cell)[1]
        influence[cc.dic['s']] += -1/3*convect_sum_jacobi(cell,cell.south.south)[1]

    if cell.east.vn <= 0:
        influence[cc.dic['ee']] += -1/6*convect_sum_jacobi(cell,cell.east.east.east.east)[0]
        influence[cc.dic['e']] += 5/6*convect_sum_jacobi(cell,cell.east.east)[0]
        influence[cc.dic['c']] += 1/3*convect_sum_jacobi(cell,cell)[0]
    else:
        influence[cc.dic['w']] += -1/6*convect_sum_jacobi(cell,cell.west.west)[0]
        influence[cc.dic['c']] += 5/6*convect_sum_jacobi(cell,cell)[0]
        influence[cc.dic['e']] += 1/3*convect_sum_jacobi(cell,cell.east.east)[0]

    if cell.west.vn >= 0:
        influence[cc.dic['ww']] += 1/6*convect_sum_jacobi(cell,cell.west.west.west.west)[0]
        influence[cc.dic['w']] += -5/6*convect_sum_jacobi(cell,cell.west.west)[0]
        influence[cc.dic['c']] += -1/3*convect_sum_jacobi(cell,cell)[0]
    else:
        influence[cc.dic['e']] += 1/6*convect_sum_jacobi(cell,cell.east.east)[0]
        influence[cc.dic['c']] += -5/6*convect_sum_jacobi(cell,cell)[0]
        influence[cc.dic['w']] += -1/3*convect_sum_jacobi(cell,cell.west.west)[0]

    return influence

def viscous_convect_1st_upwind(cell:cc.cell_class):
    """一阶迎风格式"""
    influence = [np.zeros(5) for _ in range(13)]

    if cell.north.vn <= 0:
        influence[cc.dic['n']] = viscous_convect_sum_jacobi(cell,cell.north.north)[1]
    else:
        influence[cc.dic['c']] = viscous_convect_sum_jacobi(cell,cell)[1]

    if cell.south.vn >= 0:
        influence[cc.dic['s']] = viscous_convect_sum_jacobi(cell,cell.south.south)[1]
    else:
        influence[cc.dic['c']] = viscous_convect_sum_jacobi(cell,cell)[1]

    if cell.east.vn <= 0:
        influence[cc.dic['e']] = viscous_convect_sum_jacobi(cell,cell.east.east)[0]
    else:
        influence[cc.dic['c']] = viscous_convect_sum_jacobi(cell,cell)[0]

    if cell.west.vn >= 0:
        influence[cc.dic['w']] = viscous_convect_sum_jacobi(cell,cell.west.west)[0]
    else:
        influence[cc.dic['c']] = viscous_convect_sum_jacobi(cell,cell)[0]

    return influence

def convect_hybrid(cell:cc.cell_class):
    """邢程对流项"""
    upwind = face_convect_mat_3rd_upwind(cell)
    mid = face_convect_mat_4th_mid(cell)
    viscous = viscous_convect_1st_upwind(cell)
    for j in range(13):
        cell.form_influence(j, cc.alpha_H*upwind[j]+(1-cc.alpha_H)*mid[j])
        cell.form_influence(j,np.array([np.zeros((4,5)),viscous[j]]))