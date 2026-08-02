import classconfig as cc
import numpy as np

def green_gauss_constant(face:cc.face_class):
    """基本流的*Green-Gauss*重构方法,对面上的梯度进行差分."""
    green_gauss_from_JST(face.me,face.me.north,face.me.south,face.me.east,face.me.west)
    green_gauss_from_JST(face.nei,face.nei.north,face.nei.south,face.nei.east,face.nei.west)
    face.grad_2nd_mid()

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