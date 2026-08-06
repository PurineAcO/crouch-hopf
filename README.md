# crouch-hopf

参考了*JCP 2007* J.D.Crouch 的论文. 

在本篇README中,**只介绍**该论文的理论部分,具体代码实现将会在对应文件夹下进行实现.该仓库代码归属北邮客运集团蓟门南客运段码家溪客运技术所,有极大概率在未来某个时间开始彻底消失.以下内容不会在公众号发布.

## 1. 从NS到RANS

在PurineCFD中已经阐述了什么是NS方程以及它是怎么来的,下面阐述一种对NS方程解析时候的近似RANS.RANS全称雷诺时均NS,它假设某个某种流动物理量可以分裂为基本量和脉动量,即:

$$ \phi = \bar{\phi} + \phi' $$

其中人们对 $\phi'$ 做出假设,譬如令其各种平均值为0,由此进行了一次很彻底的滤波,而消失的 $\phi'$ 则主要诉诸于湍流模型,以平衡其影响.

> 这种看似不负责任的手段,统治了CFD界很大的领域.个人不认为URANS同RANS有什么本质的区别,无非是把时间的导数打开了而已,终究都不是尺度的模拟.

那么我们就要追问一个问题,如果不对 $\phi'$ 做假设会怎样？这样我们就去改写URANS方程,连续性方程变成了这样:

$$ \frac{\partial (\bar{\rho} + \rho')}{\partial t} + \frac{\partial ((\bar{\rho} + \rho')(\bar{u} + u'))}{\partial x} + \frac{\partial ((\bar{\rho}+ \rho')(\bar{v} + v'))}{\partial y} = 0 $$

考虑到基流 $\rho,u,v$ 已经满足了RANS方程,由此我们展开各个乘积并且消除所有基流项,就像

$$(\bar{\rho} + \rho')(\bar{u} + u') \rightarrow \rho' \bar{u} + \bar{\rho} u' + \rho' u' \approx \rho' \bar{u} + \bar{\rho} u'$$

在最后一步,我们消去了双重脉动项,就正如物理上只取一阶小量一样.

> 这一个步骤和**乘积的求导法则**十分相像,是的,我们在URANS方程取了一个导数!

由此,我们改写连续性方程:

$$\frac{\partial \rho'}{\partial t} + \frac{\partial (\bar{\rho} u' + \rho' \bar{u})}{\partial x} + \frac{\partial (\bar{\rho} v' + \rho' \bar{v})}{\partial y} = 0$$

动量方程:

$$\frac{\partial (\rho u' + \rho' \bar{u})}{\partial t} + \frac{\partial (\rho' \bar{u}^2 + 2 \bar{\rho} \bar{u} u' + \bar{\rho} R T' + \rho' R \bar{T})}{\partial x} + \frac{\partial (\bar{\rho} \bar{u} v' + \bar{\rho} u' \bar{v} + \rho' \bar{u} \bar{v})}{\partial y} \\ = \frac{\partial}{\partial x} (\tau'_{xx}) + \frac{\partial}{\partial y} (\tau'_{xy})$$

$$\frac{\partial (\rho v' + \rho' \bar{v})}{\partial t} + \frac{\partial (\bar{\rho} \bar{u} v' + \bar{\rho} u' \bar{v} + \rho' \bar{u} \bar{v})}{\partial x} + \frac{\partial (\rho' \bar{v}^2 + 2 \bar{\rho} \bar{v} v' + \bar{\rho} R T' + \rho' R \bar{T})}{\partial y} \\ = \frac{\partial}{\partial x} (\tau'_{xy}) + \frac{\partial}{\partial y} (\tau'_{yy})$$

能量方程:

$$\frac{\partial (\rho' (C_v \bar{T} + 0.5 (\bar{u}^2 + \bar{v}^2)) + \bar{\rho} (C_v T' + \bar{u} u' + \bar{v} v'))}{\partial t} \\ + \frac{\partial ((\bar{\rho} u' + \rho' \bar{u}) (C_p \bar{T} + 0.5 (\bar{u}^2 + \bar{v}^2)) + \bar{\rho} \bar{u} (C_p T' + \bar{u} u' + \bar{v} v'))}{\partial x} \\ + \frac{\partial ((\bar{\rho} v' + \rho' \bar{v}) (C_p \bar{T} + 0.5 (\bar{u}^2 + \bar{v}^2)) + \bar{\rho} \bar{v} (C_p T' + \bar{u} u' + \bar{v} v'))}{\partial y} \\ = \frac{\partial}{\partial x} (u' \tau_{xx} + v' \tau_{xy} + \bar{u} \tau'_{xx} + \bar{v} \tau'_{xy} + \phi'_{x}) + \frac{\partial}{\partial y} (u' \tau_{xy} + v' \tau_{yy} + \bar{u} \tau'_{xy} + \bar{v} \tau'_{yy} + \phi'_{y}) $$

湍流方程将会在第三章进行全面的推导,这里省略.这里面产生的雷诺应力项、热流,也在第三章加以阐述其进一步形式.

## 2. 模态化的URANS

> 在CFD计算中,时间推进的时间步长十分重要,他决定了时间推进时的耗散程度,如果将URANS模态化,将不再有显式的推进时间 $\Delta t$ ,这就是一种隐式的计算手法.

我们令

$$\boldsymbol{q} = (\rho,u,v,T,\tilde{\nu})^\top$$

定义

$$\boldsymbol{\bar{q}} = (\bar{\rho},\bar{u},\bar{v},\bar{T},\bar{\tilde{\nu}})^\top,\boldsymbol{q'} = (\rho',u',v',T',\tilde{\nu}')^\top$$

考察方程左侧的时间导数项,可以写成:

$$\frac{\partial (\boldsymbol{M q'} )}{\partial t} = \frac{\partial}{\partial t} \begin{pmatrix}
1 & 0 & 0 & 0 & 0 \\
\bar{u} & \bar{\rho} & 0 & 0  & 0 \\
\bar{v} & 0 & \bar{\rho} & 0 & 0 \\
C_V \bar{T} + \frac{\bar{u}^2 + \bar{v}^2}{2} & \bar{\rho} \bar{u} & \bar{\rho} \bar{v} & \bar{\rho} C_V & 0 \\
\bar{\tilde{\nu}} & 0 & 0 & 0 & \bar{\rho}
\end{pmatrix}\boldsymbol{ q'} $$

> 为了方便起见,以后的式子里不带上划线的也表示基流.带上划线仅限于和脉动项需要显著区分的情形

所有无粘对流项、粘性对流项、粘性扩散项、粘性源项都可以写成 $\boldsymbol{q'}$ 的函数,不妨记为算子 $\cal{N}$ ,这样就得到了:

$$ \frac{\partial \boldsymbol{M q'}}{\partial t} + \cal{N} (\boldsymbol{q'})= 0$$

算子 $\cal{N}$ 未免太难算了,我们可以通过适当的手段将其进行离散,以得到一种线化近似 $\cal{N}(\boldsymbol{q'}) = \boldsymbol{N_{\bar{q}} q'}$,其中,$\boldsymbol{N_{\bar{q}}}$ 只关于 $\boldsymbol{\bar{q}}$,即 $\boldsymbol{N_{\bar{q}}} =\boldsymbol{N(\bar{q})} $,我们先承认这个事实,接下来几章将会用相当的篇幅进行阐述.由此,整个改写为

$$\frac{\partial \boldsymbol{M q'}}{\partial t} + \boldsymbol{N(\bar{q})q'}= 0 $$

注意到 $\boldsymbol{M}$ 是无关于时间的,这样可以提取出来,并左乘 $\boldsymbol{M^{-1}}$,得到:

$$\frac{\partial \boldsymbol{q'}}{\partial t} + \boldsymbol{L(\bar{q})q'}= 0 $$

现在对 $\boldsymbol{q'}$ 进行模态化,将其时间部分和空间部分分开,令

$$ \boldsymbol{q'} = \hat{\boldsymbol{q}} e^{- \mathrm{i} \omega t} $$

带入上面方程,这瞬间变成一个特征值问题:

$$  (-\mathrm{i} \omega +\boldsymbol{L(\bar{q})}) \hat{\boldsymbol{q}} = 0 $$

考察 $\boldsymbol{L}$,假设求解区域的网格数量是 $N$,那么其维度是 $5N$,就有 $5N$ 个特征值.我们必然无法找出所有特征值,但只需要找到那些 $\Im \omega >0$ 的,就是找到了可以**发散**的模态,这部分就是节律性流动失稳的主导模态.

> 如此下来,任务就十分清楚,我们需要:
> - (0)写对影响方程
> - (1)构建空间离散
> - (2)求解稀疏矩阵.