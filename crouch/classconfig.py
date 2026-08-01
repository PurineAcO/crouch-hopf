import numpy as np

# 常数定义

# ————————————————————physics constants——————————————————
R = 287.06
cp = 1004.71               # 空气定压比热

# ————————————————————simulation params——————————————————
alpha_H = 0.2                   # 混合格式系数
HALO = 2                        # 虚单元层数

#TODO 常数定义一般需要从config.json中来

class cell_class:
    def __init__(self,index:tuple[int,int],x,y,rho,u,v,T,miubl):
        """单元中心型的类,在创建时需要给出全部物理量,索引从1开始,i=1是最左侧,j=1是物面上方首层网格"""
        self.index = index                              # 单元编号,为结构化网格元组形式(i,j)
        self.x = x                                      # 单元中心x坐标
        self.y = y                                      # 单元中心y坐标     
        self.rho = rho                                  # 密度
        self.u = u                                      # x速度
        self.v = v                                      # y速度
        self.T = T                                      # 静温
        self.H = cp*T + 0.5*(self.u**2 + self.v**2)     # 焓
        self.miubl = miubl                              # 修正湍流粘度(̃ν)

        # 单元的全部邻接面, 槽位顺序 [W, E, S, N] (见 FACE_W/E/S/N)
        self.west : face_class = None
        self.east : face_class = None
        self.south : face_class = None
        self.north : face_class = None

        # 本单元的流动量能够写为13个矩阵的线性组合,他们的位置下
        self.influence = [                                np.zeros((5,5)),
                                          np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),
                          np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),
                                          np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5)),
                                                          np.zeros((5,5))
                          ]  

        # 以下是各个对流项矩阵,我们约定,所有矩阵全部写在=0方程的左侧
        self.F = np.zeros((5,5))                        # x对流项(F)
        self.G = np.zeros((5,5))                        # y对流项(G)

    def form_influence(self,index,A):
        """对本单元的Nq离散化后是13个矩阵的线性组合,有13个网格对本网格造成了影响,他们的空间分布是\n
        [ \\ ][ \\ ][ **1** ][ \\ ][ \\ ]\n
        [ \\ ][ **2** ][ **3** ][ **4** ][ \\ ]\n
        [ **5** ][ **6** ][ **7** ][ **8** ][ **9** ]\n
        [ \\ ][**10** ][**11** ][**12** ][ \\ ]\n
        [ \\ ][ \\ ][**13** ][ \\ ][ \\ ]\n
        """    
        self.influence[index] = self.influence[index]+A

    def cell_convect_mat(self):
        # 构建对流项矩阵,只保留前4*4个区间
        self.F = np.array([[self.u,self.rho,0,0,0],
                        [self.u**2+ R*self.T,2*self.rho*self.u,0,self.rho*R,0],
                        [self.u*self.v,self.rho*self.v,self.rho*self.u,0,0],
                        [self.u*self.H,self.rho*(self.H+self.u**2),self.rho*self.u*self.v,self.rho*self.u*cp,0],
                        [0,0,0,0,0]])
        self.G = np.array([[self.v,0,self.rho,0,0],
                        [self.u*self.v,self.rho*self.v,self.rho*self.u,0,0],
                        [self.v**2+ R*self.T,0,2*self.rho*self.v,self.rho*R,0],
                        [self.v*self.H,self.rho*self.u*self.v,self.rho*(self.H+self.v**2),self.rho*self.v*cp,0],
                        [0,0,0,0,0]])

    def source_mat(self):
        ...


class face_class():
    def __init__(self,direction:str,mid:tuple,jacobi,
                 me:cell_class,nei:cell_class):
        """`me`是本网格,`nei`指的是邻居网格,`direction`指的是邻居网格在本网格的区位\n
        'N'表示北面,`S`表示南面,`E`表示东面,`W`表示西面\n"""
        self.direction = direction
        self.me = me            # 一般一个面的高侧为me网格
        self.nei = nei          # 一般一个面的低侧为nei网格
        self.mid = mid
        self.jacobian = jacobi  # 形式必须是(Xn,Yn;Xs,Ys)
        self.form_physics()     # 形成面上物理量,二阶中心差分
        self.recognize_direction()  # 将面与单元的槽位对应起来

    def recognize_direction(self):
        """根据`direction`属性,将面与单元的槽位对应起来"""
        if self.direction == "NS":
            self.south = self.nei
            self.north = self.me
            self.east = self.west = None
        elif self.direction == "WE":
            self.west = self.nei
            self.east = self.me
            self.north = self.south = None
        else:
            raise ValueError("face_class: direction must be 'NS' or 'WE'")

    def form_physics(self):
        self.rho = (self.me.rho+self.nei.rho)/2
        self.u = (self.me.u+self.nei.u)/2
        self.v = (self.me.v+self.nei.v)/2
        self.T = (self.me.T+self.nei.T)/2
        self.H = (self.me.H+self.nei.H)/2
        self.miubl = (self.me.miubl+self.nei.miubl)/2

    def jacobi(self,A,B):
        """对输入的两个量`A`和`B`进行Jacobi变换"""
        A1 = A * self.jacobian[0][0] + B * self.jacobian[0][1]
        A2 = A * self.jacobian[1][0] + B * self.jacobian[1][1]
        return A1,A2


CellList : list[list[cell_class]] = []
FaceList_WE : list[face_class] = []
FaceList_NS : list[face_class] = []

BigMatrix = []