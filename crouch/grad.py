import classconfig as cc
import numpy as np

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

    elif face.direction == "NS" : 
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

def green_gauss_cell_vari(cell:cc.cell_class):
    """计算某个`cell`的梯度影响矩阵,根据green-gauss方法,会和5个网格挂钩\n
       为了看起来舒服,会返回一个字典."""

    grad_dic = {}
    grad_dic['c'] = (cell.north.jacobian[0]+cell.south.jacobian[0]+
                    cell.east.jacobian[0]+cell.west.jacobian[0])/cell.vol/2
    grad_dic['n'] = (cell.north.jacobian[0])/cell.vol/2
    grad_dic['s'] = -(cell.south.jacobian[0])/cell.vol/2
    grad_dic['e'] = (cell.east.jacobian[0])/cell.vol/2
    grad_dic['w'] = -(cell.west.jacobian[0])/cell.vol/2
    return grad_dic
    

def green_gauss_from_JST(cell:cc.cell_class,facenorth:cc.face_class,facesouth:cc.face_class,
                    faceeast:cc.face_class,facewest:cc.face_class):
        """基于Green-Guass的梯度构建,一般按照北、南、东、西来写""" 
        u_vec = np.array([facenorth.u,facesouth.u,faceeast.u,facewest.u])
        v_vec = np.array([facenorth.v,facesouth.v,faceeast.v,facewest.v])
        miubl_vec = np.array([facenorth.miubl,facesouth.miubl,faceeast.miubl,facewest.miubl])
        T_vec = np.array([facenorth.T,facesouth.T,faceeast.T,facewest.T])
        nx_vec = np.array([facenorth.nx,-facesouth.nx,faceeast.nx,-facewest.nx])
        ny_vec = np.array([facenorth.ny,-facesouth.ny,faceeast.ny,-facewest.ny])
        cell.ugrad = np.array([np.dot(u_vec,nx_vec),np.dot(u_vec,ny_vec)])/cell.vol
        cell.vgrad = np.array([np.dot(v_vec,nx_vec),np.dot(v_vec,ny_vec)])/cell.vol
        cell.miublgrad = np.array([np.dot(miubl_vec,nx_vec),np.dot(miubl_vec,ny_vec)])/cell.vol
        cell.Tgrad = np.array([np.dot(T_vec,nx_vec),np.dot(T_vec,ny_vec)])/cell.vol