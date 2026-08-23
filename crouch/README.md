# Crouch-Py 函数和方法说明

使用URANS进行仿真时,常常受到时间步长 $\Delta t$ 的限制,有可能造成伪数值震荡,进而影响结果准确性.为了避免这一问题的发生,可以将求解器投射到频域上,形成一种基于模态的计算流体力学方法.由于频域上的解法并不显含时间,因此被视为无需考虑时间步长无关性的CFD方法

*Crouch-Py* 基于论文 [*JCP 2007 Crouch*](https://doi.org/10.1016/j.jcp.2006.10.035) 完成,是一种基于有限差分的隐式求解器,使用[*Spalart-Allmaras*湍流模型](),依赖ARPACK大型稀疏矩阵求解器,采用高阶精度的格式和链式网格访问,~~理论上可支持非结构网格~~(目前仅可以支持O-Block).

使用时,需提供一个[标准格式]()的**RANS数据文件和网格映射表**.上述RANS数据可以从商业软件或者自研的流体力学求解器中得到,而网格映射表则可以通过`hopf`中的脚本从 *Fluent msh(Binary)* 中转换出来.总之,使用可被读取的正确格式是使用好本求解器的关键因素.

本求解器先读取RANS数据和网格邻接关系,随后建立影响矩阵 $\boldsymbol{T}$ ,最后进行求解.下文称为**读取期**、**重构期**、**求解期**.

> 本文中,所有未使用的设计点但仍在源码中得以保留的,用~~删除线~~表示

<!-- ## 目录
 -->

## 网格的存储

本段的描述主要基于`classconfig.py`

对所有的网格单元`cell_class`,均采用链式存储.网格链接其4个邻接边,邻接边链接2个网格.对于所有的网格,其存储的区域分为以下4个部分:

1. 网格编号和几何

    - 每个网格在形成期的**索引**为`index`,其是一个元组`(s,n)`,其中`s`表示的是在每一个O形圈中该网格所处的位置,`n`表示的是网格所处的O型层数.随着`s`的变大,网格被定义为东向(`east`),随着`n`的变大,网格被定义为北向(`north`),从O-Block的视角看,东西方向是环形的(一般以顺时针为东),而北向总是指向远场.

    - 每个网格还在中心定义了位置坐标`x`,`y`,以及其体积`vol`和到壁面的距离`sad`(取自英文 *S-A distance* ),网格的Jacobi矩阵`Jacobi`定义为网格边中点的连线的Jacobi矩阵.

    - 邻接面:按照几何位置,对4个围成的面进行邻接.分别以`north`,`south`,`east`,`west`命名.如果需要邻接网格访问,应当使用`thiscell.north.north`.源码中落实了类型注释,以方便观察.
    
        > 下一个版本将修改邻接方式,改为不基于几何方位的邻接描述.

2. 物理量

    - 存储 $q = (\rho,u,v,T,\tilde{\nu})$ 五个物理量,这五个物理量分别被命名为`rho`,`u`,`v`,`T`,`miubl`,为了计算方便,还引入了焓 $H = C_p T + \frac{1}{2}(u^2 + v^2)$ 这个概念,其名称为`H`.以上物理量均随RANS数据读入.

    - 存储 $q$ 的梯度,分别命名为`ugrad`,`vgrad`,`Tgrad`,`miublgrad`

        > 如无特殊声明,所有元组、列表均为0基

    - 存储湍流变量(湍流字典).由于进行URANS求导时,涉及到了很多中间变量,所以这里存储了有效粘度系数 $\mu_{\mathrm{eff}}$ ,有效导热系数 $\lambda_{\mathrm{eff}}$ ,湍流粘度比 $\chi$ ,阻尼函数 $f_{v1}$ ,涡量修正函数 $f_{v2}$ ,壁面阻尼函数 $f_w$ ,生产项修正函数 $f_{t2}$ ,涡量 $\Omega$, 应变力张量 $S$, 修正涡量参数 $\tilde{S}$ ,无量纲壁面距离 $r$ ,粘度(基于Sutherland) $\mu$ ,壁面距离因子 $g$ .这些量的计算方法将于[S-A湍流模型]()给出

3. 无粘对流项矩阵`F`,`G`及其形成函数`self.cell_convect_mat`.在形成网格时,即调用了后者.其中, $F,G$ 形式如下(需要注意的是,对流项矩阵定义在直角坐标系下):

    $$ 
    F = \begin{pmatrix}
    u & \rho & 0 & 0 & 0 \\
    u^2 + RT & 2\rho u & 0 & \rho R & 0 \\
    u v & \rho v & \rho u & 0 & 0 \\
    u H & \rho(H + u^2) & \rho u v & \rho u c_p & 0 \\
    0 & 0 & 0 & 0 & 0
    \end{pmatrix}
    $$

    $$
    G = \begin{pmatrix}
    v & 0 & \rho & 0 & 0 \\
    u v & \rho v & \rho u & 0 & 0 \\
    v^2 + RT & 0 & 2\rho v & \rho R & 0 \\
    v H & \rho u v & \rho(H + v^2) & \rho v c_p & 0 \\
    0 & 0 & 0 & 0 & 0
    \end{pmatrix}
    $$ 

    粘性对流项矩阵由`self.viscous_convect_vec`形成,该方法**返回2个1x5向量** $\boldsymbol{F_V}$ 和 $\boldsymbol{G_V}$

    $$
    \boldsymbol{F_V} = \left( u \tilde{\nu},\rho \tilde{\nu},0,0,\rho u \right)^\top
    $$

    $$
    \boldsymbol{G_V} = \left( v \tilde{\nu},0,\rho \tilde{\nu},0,\rho v \right)^\top
    $$


4. 影响矩阵`influence`:每个中心网格(即6号)会与周围13个网格发生关系,其位置如下图所示.也就是说在最后的影响矩阵 $\boldsymbol{T}$ 中,每一行只有13个元素.矩阵重构期的任务是先对中心网格求出13个影响矩阵和 $\hat{q}$ 向量的线性组合,随后组装到对应位置上去. 为了调用方便,`classconfig.dic`提供了使用`n`,`nn`,`ne`等调用对应网格影响矩阵位置的方法,推荐使用字典.

    <table style="border-collapse: collapse; text-align: center; font-size: 1.2em; margin: 1em auto;">
    <tr>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #e0e0e0;"></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #e0e0e0;"></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>0</b></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #e0e0e0;"></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #e0e0e0;"></td>
    </tr>
    <tr>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #e0e0e0;"></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>1</b></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>2</b></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>3</b></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #e0e0e0;"></td>
    </tr>
    <tr>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>4</b></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>5</b></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>6</b></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>7</b></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>8</b></td>
    </tr>
    <tr>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #e0e0e0;"></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>9</b></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>10</b></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>11</b></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #e0e0e0;"></td>
    </tr>
    <tr>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #e0e0e0;"></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #e0e0e0;"></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #fff;"><b>12</b></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #e0e0e0;"></td>
        <td style="width: 44px; height: 44px; padding: 0; border: 1px solid #ccc; background: #e0e0e0;"></td>
    </tr>
    </table>

邻接边`face_class`在形成期被分化为`NS`和`WE`两种类型,~~尽管在后面的操作中没有显式使用其几何方向~~,其存储的内容被划分为以下2个部分:

1. 面编号与几何

    - 在面的形成期,需要给定面的`direction`,这个字段必须为`NS`或者`WE`(如果不是,似乎也不会报错,只是不会执行),同时,会初步行车面的邻接关系`me`和`nei`,其中,`me`总是指向当前面的高侧,`nei`总是指向当前面的低侧.此后,再根据`direction`决定`me`和`nei`的朝向,调用`self.recognize_direction`将其分配到南北或者东西方向.

    - 面上的jacobi矩阵 $\boldsymbol{J}$ 被定义在`jacobian`,其中,Jacobi矩阵所有元素均遵循向东、北为正的原则.该矩阵的形状为

    $$
    \boldsymbol{J} = \begin{pmatrix} X_n & Y_n \\ X_s & Y_s \end{pmatrix}
    $$

2. 物理量

    - 面上的 $\hat{q}$ 同网格中心,其根据相邻网格进行中心差分重构,由方法`self.form_physics`在形成期建立.梯度也根据相邻网格进行中心差分重构,由方法`self.grad_2nd_mid`在求解期建立.

    - 湍流字典.有效粘度系数 $\mu_{\mathrm{eff}}$ ,有效导热系数 $\lambda_{\mathrm{eff}}$ ,湍流粘度比 $\chi$ ,阻尼函数 $f_{v1}$ ,粘度 $\mu$ 均由相邻面的中心差分重构,但是这部分在`turbulence.diffusion_2nd_mid_SA`中完成.计算完毕这些内容后,**再**计算面上的Reynold应力和热流

    - 法向速度作为特别的参数,以 *@property* 形式给出,不受形成期限制.

    <!-- $$
    \tau_{xx} = \mu_{\mathrm{eff}} \left( \frac{2}{3} \frac{\partial u}{\partial x} - \frac{1}{3} \frac{\partial v}{\partial y} \right)
    $$

    $$
    \tau_{xy} = \mu_{\mathrm{eff}} \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right)
    $$

    $$
    \tau_{yy} = \mu_{\mathrm{eff}} \left( \frac{2}{3} \frac{\partial v}{\partial y} - \frac{1}{3} \frac{\partial u}{\partial x} \right)
    $$ -->`

网格和邻接面还具有`self.jacobi(a,b)`的方法,其中`a,b`必须是定义了加法的量.该函数可以返回 $A,B$ 经过jacobi变换后的 $\tilde{A},\tilde{B}$

全体网格被存储在`CellList`里面,存储结构为**双层存储**,可以直接通过索引访问.面按照方向分别存储在`FaceList_WE`和`FaceList_NS`里面,存储结构为**展平存储**,一般只推荐遍历.

由于采用了高阶格式,为了方便编程,在壁面、远场、~~周期边界~~处设置了层数为`HALO`的虚拟网格,因此,物理的网格在`CellList`的存储中存在偏移量.为此,建议使用`goto_HALOcell`访问.

在仿真中使用的所有物理常数、湍流模型常数、求解器控制参数均存储在`config.json`中,应保证`config.json`具有正确形式,并可被打开.

## 空间离散

本部分基于`convect.py`、`turbulence.py`、`grad.py`描述

对流项采取三阶迎风格式和四阶中心差分的混合格式,其具体形式已在上一级文档给出.在各个网格形成无粘对流项矩阵后,经过`convect_sum_jacobi`和`viscous_convect_sum_jacobi`变换为随体坐标,随后用`face_convect_4th_mid`和`face_convect_3rd_upwind`计算出无粘通量,用`viscous_convect_1st_upwind`计算粘性通量,最后用混合参数 $\alpha_H$ 调用`convect_hybrid`进行混合并写入影响矩阵.

