import classconfig as cc
import numpy as np

def green_gauss_constant(face:cc.face_class):
    """基本流的*Green-Gauss*重构方法,对面上的梯度进行差分."""
    green_gauss_from_JST(face.me,face.me.north,face.me.south,face.me.east,face.me.west)
    green_gauss_from_JST(face.nei,face.nei.north,face.nei.south,face.nei.east,face.nei.west)
    face.grad_2nd_mid()

def green_gauss_face_vari(face:cc.face_class):
    """计算某个`face`的梯度影响矩阵,根据green-gauss方法,会和8个网格挂钩\n
       为了看起来舒服,会返回一个字典."""

    if face.direction == "WE" :
        grad_dic = {}
        east = face.east;west = face.west
        grad_dic['w'] = ((west.north.jacobian[0]-west.south.jacobian[0]+
                        west.east.jacobian[0]-west.west.jacobian[0])/west.vol/4 -
                        east.west.jacobian[0]/east.vol/4) 
        grad_dic['e'] = ((east.north.jacobian[0]-east.south.jacobian[0]+
                        east.east.jacobian[0]-east.west.jacobian[0])/east.vol/4 +
                        west.east.jacobian[0]/west.vol/4)
        grad_dic['nw'] = (west.north.jacobian[0])/west.vol/4
        grad_dic['ne'] = (east.north.jacobian[0])/east.vol/4
        grad_dic['sw'] = -(west.south.jacobian[0])/west.vol/4
        grad_dic['se'] = -(east.south.jacobian[0])/east.vol/4
        grad_dic['ee'] = (east.east.jacobian[0])/east.vol/4
        grad_dic['ww'] = -(west.west.jacobian[0])/west.vol/4
        return grad_dic

    elif face.direction != "NS" : 
        grad_dic = {}
        north = face.north; south = face.south
        grad_dic['n'] = ((north.north.jacobian[0]-north.south.jacobian[0]+
                        north.east.jacobian[0]-north.west.jacobian[0])/north.vol/4 +
                        south.north.jacobian[0]/south.vol/4)
        grad_dic['s'] = ((south.north.jacobian[0]-south.south.jacobian[0]+
                        south.east.jacobian[0]-south.west.jacobian[0])/south.vol/4 -
                        north.south.jacobian[0]/north.vol/4)
        grad_dic['ne'] = (north.east.jacobian[0])/north.vol/4
        grad_dic['se'] = (south.east.jacobian[0])/south.vol/4
        grad_dic['nw'] = -(north.west.jacobian[0])/north.vol/4
        grad_dic['sw'] = -(south.west.jacobian[0])/south.vol/4
        grad_dic['nn'] = (north.north.jacobian[0])/north.vol/4
        grad_dic['ss'] = -(south.south.jacobian[0])/south.vol/4
        return grad_dic

    else: raise ValueError("face.direction must be 'WE' or 'NS'")

def green_gauss_from_JST(cell:cc.cell_class,face1:cc.face_class,face2:cc.face_class,
                    face3:cc.face_class,face4:cc.face_class):
        """基于Green-Guass的梯度构建""" 
        u_vec = np.array([face1.u,face2.u,face3.u,face4.u])
        v_vec = np.array([face1.v,face2.v,face3.v,face4.v])
        miubl_vec = np.array([face1.miubl,face2.miubl,face3.miubl,face4.miubl])
        T_vec = np.array([face1.T,face2.T,face3.T,face4.T])
        nx_vec = np.array([face1.nx,-face2.nx,face3.nx,-face4.nx])
        ny_vec = np.array([face1.ny,-face2.ny,face3.ny,-face4.ny])
        cell.ugrad = np.array([np.dot(u_vec,nx_vec),np.dot(u_vec,ny_vec)])/cell.vol
        cell.vgrad = np.array([np.dot(v_vec,nx_vec),np.dot(v_vec,ny_vec)])/cell.vol
        cell.miublgrad = np.array([np.dot(miubl_vec,nx_vec),np.dot(miubl_vec,ny_vec)])/cell.vol
        cell.Tgrad = np.array([np.dot(T_vec,nx_vec),np.dot(T_vec,ny_vec)])/cell.vol