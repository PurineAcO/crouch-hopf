import classconfig as cc
import numpy as np

def face_convect_mat_4rdmid(face:cc.face_class):
    i,j = face.me.index
    if face.direction == 'N':
        G_list = [face.jacobi(cc.CellList[i][dj].F, cc.CellList[i][dj].G)[1] for dj in [j+2, j+1, j, j-1]]
        for pos, g, w in zip([1,3,7,11], G_list, [-1/12, 7/12, 7/12, -1/12]):
            face.me.form_influence(pos, w * g)
    if face.direction == 'S':
        G_list = [face.jacobi(cc.CellList[i][dj].F, cc.CellList[i][dj].G)[1] for dj in [j-2, j-1, j, j+1]]
        for pos, g, w in zip([13,11,7,3], G_list, [-1/12, 7/12, 7/12, -1/12]):
            face.nei.form_influence(pos, -1 * w * g)
    if face.direction == 'E':
        F_list = [face.jacobi(cc.CellList[di][j].F, cc.CellList[di][j].G)[0] for di in [i+2, i+1, i, i-1]]
        for pos, g, w in zip([6,7,8,9], G_list, [-1/12, 7/12, 7/12, -1/12]):
            face.nei.form_influence(pos, w * g)
    if face.direction == 'W':
        F_list = [face.jacobi(cc.CellList[di][j].F, cc.CellList[di][j].G)[0] for di in [i-2, i-1, i, i+1]]
        for pos, g, w in zip([5,6,7,8], G_list, [-1/12, 7/12, 7/12, -1/12]):
            face.me.form_influence(pos, -1 * w * g)
        
