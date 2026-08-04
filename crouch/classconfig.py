import json
import os

import numpy as np

# 常数定义

# ————————————————————configuration——————————————————
# 全部常数从项目根目录的 config.json 中读取,按湍流模型参数/物理参数/求解器参数分类
with open(os.path.join(os.path.dirname(__file__), "..", "config.json"),
          encoding="utf-8") as _cfg_file:
    _CONFIG = json.load(_cfg_file)

# ————————————————————turbulence model constants(S-A模型)——————————————————
inv_sigma = _CONFIG["turbulence"]["inv_sigma"]  # 1/σ,湍流扩散项系数(代码中使用1/σ形式)
Cv1 = _CONFIG["turbulence"]["Cv1"]          # 阻尼函数常数Cv1
Ct3 = _CONFIG["turbulence"]["Ct3"]          # 转捩修正常数Ct3
Ct4 = _CONFIG["turbulence"]["Ct4"]          # 转捩修正常数Ct4
fv3 = _CONFIG["turbulence"]["fv3"]          # 涡量模修正系数fv3
kappa = _CONFIG["turbulence"]["kappa"]      # von Kármán常数κ
Cb1 = _CONFIG["turbulence"]["Cb1"]          # 生成项常数Cb1
Cb2 = _CONFIG["turbulence"]["Cb2"]          # 扩散项常数Cb2
Cw1 = _CONFIG["turbulence"]["Cw1"]          # 破坏项常数Cw1
Cw2 = _CONFIG["turbulence"]["Cw2"]          # 壁面阻尼函数常数Cw2
Cw3 = _CONFIG["turbulence"]["Cw3"]          # 壁面阻尼函数常数Cw3
rmax = _CONFIG["turbulence"]["rmax"]        # 无量纲距离r的上限

# ————————————————————physics constants(Sutherland等空气性质)——————————————————
R = _CONFIG["physics"]["R"]                 # 气体常数
cp = _CONFIG["physics"]["cp"]               # 空气定压比热
mu0 = _CONFIG["physics"]["mu0"]             # Sutherland公式参考粘度
T0 = _CONFIG["physics"]["T0"]               # Sutherland公式参考温度
Ts = _CONFIG["physics"]["Ts"]               # Sutherland公式Sutherland温度
Pr = _CONFIG["physics"]["Pr"]               # 层流Prandtl数
Prt = _CONFIG["physics"]["Prt"]             # 湍流Prandtl数

# ————————————————————solver params——————————————————
alpha_H = _CONFIG["solver"]["alpha_H"]      # 混合格式系数
HALO = _CONFIG["solver"]["HALO"]            # 虚单元层数

# 在实际的计算中,程序一共会和本家网格的周围13个单元发生关系,这是他们在cell.influence的编号
dic = {'n':2,'nn':0,'s':10,'ss':12,'e':7,'ee':8,'w':5,'ww':4,'c':6,'ne':3,'nw':1,'se':11,'sw':9}

class cell_class:
    def __init__(self,index:tuple[int,int],x,y,rho,u,v,T,miubl,vol):
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
        self.vol = vol                                  # 单元体积

        # 单元的梯度信息
        self.ugrad = np.zeros(2)                        # u梯度
        self.vgrad = np.zeros(2)                        # v梯度
        self.Tgrad = np.zeros(2)                        # T梯度
        self.miublgrad = np.zeros(2)                    # ̃ν梯度

        # 单元的湍流字典
        self.mu_eff = None                               # 有效粘度μeff
        self.lambda_eff = None                           # 有效导热系数λeff
        self.chi = None                                  # 修正粘度比χ
        self.fv1 = None                                  # 阻尼函数fv1
        self.fv2 = None                                  # 涡量修正函数fv2
        self.fw = None                                   # 壁面阻尼函数fw
        self.ft2 = None                                  # 生产项修正函数ft2
        self.S = None                                    # 涡量模S
        self.r = None                                    # 无量纲距离r
        self.mu = None                                   # 分子粘度μ

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

        # 在邢程期调用的方法
        self.cell_convect_mat()                         # 构建对流项矩阵

    def form_influence(self,index,A):
        """对本单元的Nq离散化后是13个矩阵的线性组合,有13个网格对本网格造成了影响,他们的空间分布是\n
        [ \\ ][ \\ ][ **0** ][ \\ ][ \\ ]\n
        [ \\ ][ **1** ][ **2** ][ **3** ][ \\ ]\n
        [ **4** ][ **5** ][ **6** ][ **7** ][ **8** ]\n
        [ \\ ][**9** ][**10** ][**11** ][ \\ ]\n
        [ \\ ][ \\ ][**12** ][ \\ ][ \\ ]\n
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

    def cell_jacobi(self):
        m_w, m_e = self.west.mid, self.east.mid
        m_s, m_n = self.south.mid, self.north.mid
        s_vec = (m_e[0] - m_w[0], m_e[1] - m_w[1])
        n_vec = (m_n[0] - m_s[0], m_n[1] - m_s[1])
        self.jacobian = np.array([list(s_vec), list(n_vec)])

    def jacobi(self,A,B):
        A1 = A * self.jacobian[0][0] + B * self.jacobian[0][1]
        A2 = A * self.jacobian[1][0] + B * self.jacobian[1][1]
        return A1,A2

    def convect_jacobi(self):
        return self.jacobi(self.F, self.G)

    def viscous_convect_jacobi(self):
        vsF,vsG = self.jacobi(np.array([self.u*self.miubl,self.rho*self.miubl,0,0,self.rho*self.u]),
                           np.array([self.v*self.miubl,0,self.rho*self.miubl,0,self.rho*self.v]))
        return (np.array([np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5),vsF]), 
                np.array([np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5),vsG]))


class face_class():
    def __init__(self,direction:str,mid:tuple,jacobi,
                 me:cell_class,nei:cell_class):
        """`me`是本网格,`nei`指的是邻居网格,`direction`指的是邻居网格在本网格的区位\n
        'N'表示北面,`S`表示南面,`E`表示东面,`W`表示西面\n"""
        self.direction = direction
        self.me = me            # 一般一个面的高侧为me网格
        self.nei = nei          # 一般一个面的低侧为nei网格
        self.mid = mid
        self.jacobian = np.array(jacobi)  # 形式必须是(Xn,Yn;Xs,Ys)

        # 梯度
        self.ugrad = np.zeros(2)                        # u梯度
        self.vgrad = np.zeros(2)                        # v梯度
        self.Tgrad = np.zeros(2)                        # T梯度
        self.miublgrad = np.zeros(2)                    # ̃ν梯度

        # 面上的湍流字典
        self.mu_eff = None                               # 有效粘度μeff
        self.lambda_eff = None                           # 有效导热系数λeff
        self.chi = None                                  # 修正粘度比χ
        self.fv1 = None                                  # 阻尼函数fv1
        self.fv2 = None                                  # 涡量修正函数fv2
        self.fw = None                                   # 壁面阻尼函数fw
        self.ft2 = None                                  # 生产项修正函数ft2
        self.S = None                                    # 涡量模S
        self.r = None                                    # 无量纲距离r
        self.mu = None                                   # 分子粘度μ
        self.tauxx = None                                # 切应力tauxx
        self.tauxy = None                                # 切应力tauxy
        self.tauyy = None                                # 切应力tauyy

        # 构建面上的参数
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

    def grad_2nd_mid(self):
        """对面上的梯度进行二阶中心插值"""
        self.ugrad = (self.me.ugrad+self.nei.ugrad)/2
        self.vgrad = (self.me.vgrad+self.nei.vgrad)/2
        self.Tgrad = (self.me.Tgrad+self.nei.Tgrad)/2
        self.miublgrad = (self.me.miublgrad+self.nei.miublgrad)/2

    def diffusion_2nd_mid_SA(self):
        """对面上的湍流字典进行二阶中心插值并构建切应力"""
        self.mu_eff = (self.me.mu_eff+self.nei.mu_eff)/2
        self.lambda_eff = (self.me.lambda_eff+self.nei.lambda_eff)/2
        self.chi = (self.me.chi+self.nei.chi)/2
        self.fv1 = (self.me.fv1+self.nei.fv1)/2
        self.fv2 = (self.me.fv2+self.nei.fv2)/2
        self.fw = (self.me.fw+self.nei.fw)/2
        self.ft2 = (self.me.ft2+self.nei.ft2)/2
        self.S = (self.me.S+self.nei.S)/2
        self.r = (self.me.r+self.nei.r)/2
        self.mu = (self.me.mu+self.nei.mu)/2
        self.tauxx = self.mu_eff * (self.ugrad[0]-1/3*(self.ugrad[0]+self.vgrad[1]))
        self.tauxy = self.mu_eff * (self.ugrad[1]+self.vgrad[0])
        self.tauyy = self.mu_eff * (self.vgrad[1]-1/3*(self.ugrad[0]+self.vgrad[1]))

    @property
    def vn(self):
        return self.jacobi(self.u, self.v)[0]

    @property
    def nx(self):
        return self.jacobian[0][0]

    @property
    def ny(self):
        return self.jacobian[0][1]


CellList : list[list[cell_class]] = []
FaceList_WE : list[face_class] = []
FaceList_NS : list[face_class] = []

BigMatrix = []

def HALO_cellinit(S_MAX:int,N_MAX:int):
    """halo 化下标空间: s=-HALO⋯S_MAX+HALO, n=-HALO⋯N_MAX+HALO, 角落保持 None"""
    CellList.clear()
    CellList.extend([[None] * (N_MAX + 2 * HALO + 1) for _ in range(S_MAX + 2 * HALO + 1)])