import numpy as np

# 常数定义
R = 287.06
cp = ...#TODO
#TODO 常数定义一般需要从config.json中来

class cell_class:
    def __init__(self,index:tuple,x,y,rho,u,v,T,miubl):
        """单元中心型的类,在创建时需要给出全部物理量"""
        self.index = index
        self.x = x
        self.y = y
        self.rho = rho
        self.u = u
        self.v = v
        self.T = T
        self.H = cp*T + 0.5*(self.u**2 + self.v**2)
        self.miubl = miubl
        self.convect_x = np.zeros((5,5))
        self.convect_y = np.zeros((5,5))

class face_class(cell_class):
    def __init__(self,index:tuple,x,y,rho,u,v,T,miubl):
        super().__init__(index,x,y,rho,u,v,T,miubl)