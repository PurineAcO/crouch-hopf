# crouch-hopf

参考了*JCP 2007* J.D.Crouch 的论文. 

在本篇README中,**只介绍**该论文的理论部分,具体代码实现将会在对应文件夹下进行实现.该仓库代码归属北邮客运集团蓟门南客运段码家溪客运技术所,有极大概率在未来某个时间开始彻底消失.以下内容不会在公众号发布.

## 1. 从NS到RANS

在PurineCFD中已经阐述了什么是NS方程以及它是怎么来的,下面阐述一种对NS方程解析时候的近似RANS.RANS全称雷诺时均NS,它假设某个某种流动物理量可以分裂为基本量和脉动量,即:

$$ \phi = \bar{\phi} + \phi' $$

其中人们对 $\phi'$ 做出假设,譬如令其各种平均值为0,由此进行了一次很彻底的滤波,而消失的 $\phi'$ 则主要诉诸于湍流模型,以平衡其影响.

> 这种手段,统治了CFD界很大的领域.个人不认为URANS同RANS有什么本质的区别,无非是把时间的导数打开了而已.

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

## 2. 模态化

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

算子 $\cal{N}$ 未免太难算了,我们可以通过适当的手段将其进行离散,以得到一种线化近似 $\cal{N}(\boldsymbol{q'}) = \boldsymbol{N_{\bar{q}} q'}$,其中, $\boldsymbol{N_{\bar{q}}}$ 只关于 $\boldsymbol{\bar{q}}$,即 $\boldsymbol{N_{\bar{q}}} =\boldsymbol{N(\bar{q})} $,我们先承认这个事实,接下来几章将会用相当的篇幅进行阐述.由此,整个改写为

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

## 3. 影响方程展开

### 3.1 雷诺应力和热流

首先考虑雷诺应力项 $\tau_{xx},\tau_{xy},\tau_{yy}$ ,考虑其求导的结果

$$\tau_{xx}' =  2 \bar{\mu}_\mathrm{eff} \left(\frac{2}{3} \frac{\partial u'}{\partial x} - \frac{1}{3} \frac{\partial v'}{\partial y} \right) + 2 \mu_\mathrm{eff}'  \left(\frac{2}{3} \frac{\partial \bar{u}}{\partial x} - \frac{1}{3} \frac{\partial \bar{v}}{\partial y} \right) $$

$$\tau'_{xy} = \bar{\mu}_{\text{eff}} \left( \frac{\partial u'}{\partial y} + \frac{\partial v'}{\partial x} \right) + \mu'_{\text{eff}} \left( \frac{\partial \bar{u}}{\partial y} + \frac{\partial \bar{v}}{\partial x} \right)$$

$$\tau'_{yy} = 2 \bar{\mu}_{\text{eff}} \left( \frac{2}{3} \frac{\partial v'}{\partial y} - \frac{1}{3}  \frac{\partial u'}{\partial x}  \right) + 2 \mu'_{\text{eff}} \left( \frac{2}{3} \frac{\partial \bar{v}}{\partial y} - \frac{1}{3}  \frac{\partial \bar{u}}{\partial x} \right)$$

考虑热流 $\phi_x,\phi_y$ 的导数:

$$\phi'_x = \bar{\lambda}_{\text{eff}} \frac{\partial T'}{\partial x} + \lambda'_{\text{eff}} \frac{\partial \bar{T}}{\partial x},\phi'_y = \bar{\lambda}_{\text{eff}} \frac{\partial T'}{\partial y} + \lambda'_{\text{eff}} \frac{\partial \bar{T}}{\partial y}$$

这里我们遇到了第一处难点,就是如何对 $\mu_\mathrm{eff}'$ 和 $\lambda_\mathrm{eff}'$ 划归到 $\boldsymbol{q}'$ 上?后者实际上是前者除以湍流普朗特数 $Pr_t$ 的结果

$$ \lambda_\mathrm{eff}' = \frac{\mu_\mathrm{eff}'}{Pr_t} $$

因此考虑有效粘度系数的导数.

$$\mu_\mathrm{eff}' = \mu(\bar{T})' + \rho' \bar{f}_{v1} \bar{\tilde{\nu}} + \bar{\rho} f_{v1}' \bar{\tilde{\nu}} + \bar{\rho} \bar{f}_{v1} \tilde{\nu}' $$

其中, $\mu(\bar{T})' \equiv 0$ ,主要的难点是如何对 $f_{v1}$ 求导:

$$ f_{v1}' (\chi)= \chi' \frac{\partial \bar{f}_{v1}}{ \partial \chi} = \frac{(\rho \tilde{\nu})'}{\mu} \frac{\partial \bar{f}_{v1}}{ \partial \chi} = \frac{\rho'  \bar{\tilde{\nu}} + \bar{\rho}  \tilde{\nu}' }{\mu} \frac{\partial \bar{f}_{v1}}{ \partial \chi} $$

代入得

$$\mu_\mathrm{eff}' =  \rho' \bar{f}_{v1} \bar{\tilde{\nu}} +( \rho'  \bar{\tilde{\nu}} + \bar{\rho}  \tilde{\nu}') \frac{\bar{\rho} \bar{\tilde{\nu}} }{\mu} \frac{\partial \bar{f}_{v1}}{ \partial \chi}  + \bar{\rho} \bar{f}_{v1} \tilde{\nu}' = ( \rho'  \bar{\tilde{\nu}} + \bar{\rho}  \tilde{\nu}')\left( \bar{f}_{v1} + \bar{\chi} \frac{\partial \bar{f}_{v1} }{\partial \chi}\right) $$

### 3.2 湍流源项

真正的大部头存在于湍流方程的源项,首先阐述无量纲壁面距离 $r$ : 

$$ r' = \frac{\nu'}{\bar{S} \kappa^2 d^2} - \frac{\bar{\nu} \tilde{S}'}{\bar{S}^2 \kappa^2 d^2} $$

考虑生成项 $G = \rho C_{b1} (1 - f_{t2}) \tilde{S} \tilde{\nu}$ 的求导:

$$G' = C_{b1} \left( \rho' (1-\bar{f}_{t2}) \bar{S} \bar{\nu} + \bar{\rho} (- f_{t2}') \bar{S} \bar{\nu} + \bar{\rho} (1-\bar{f}_{t2}) \tilde{S}' \bar{\nu} + \bar{\rho} (1-\bar{f}_{t2}) \bar{S} \nu' \right) = C_{b1} \left( \bar{S} \bar{\nu} (1-\bar{f}_{t2}) \rho' - \bar{\rho} \bar{S} \bar{\nu} f_{t2}' \chi' + \bar{\rho} \bar{\nu} (1-\bar{f}_{t2}) \tilde{S}' + \bar{\rho} \bar{S} (1-\bar{f}_{t2}) \nu' \right)$$

其中:

$$f_{t2}' \chi' = \bar{f}_{t2}' \bar{\chi} \left( \frac{\nu'}{\bar{\nu}} + \frac{\rho'}{\bar{\rho}} \right)$$

$$\tilde{S}' = \Omega' + \frac{\nu'}{\kappa^2 d^2} \bar{f}_{v2} + \frac{\bar{\nu}}{\kappa^2 d^2} f_{v2}' \chi'$$

其中 

$$\Omega' = \frac{\partial v'}{\partial x} - \frac{\partial u'}{\partial y},\chi' = \frac{\rho'  \bar{\tilde{\nu}} + \bar{\rho}  \tilde{\nu}' }{\mu}$$

各个导数

$$f_{t2}' := \frac{\partial \bar{f}_{t2}}{\partial \chi} = -2 \bar{\chi} C_{t4} \bar{f}_{t2}$$

$$f_{v2}' := \frac{\partial \bar{f}_{v2}}{\partial \chi} = \frac{3 \bar{\chi} \bar{f}_{v1} (1-\bar{f}_{v1}) -1}{(1+\chi \bar{f}_{v1})^2} $$

考虑壁面衰减项 $D= - \rho \left( C_{w1} f_w - \frac{C_{b1}}{\kappa^2} f_{t2} \right) \left( \frac{\tilde{\nu}}{d} \right)^2$ 的求导:

$$
\begin{aligned}
D' &= - \left[ \rho' \left( C_{w1} \bar{f}_w - \frac{C_{b1}}{\kappa^2} \bar{f}_{t2} \right) \left( \frac{\bar{\nu}}{d} \right)^2 + \bar{\rho} C_{w1} f_w' r' \left( \frac{\bar{\nu}}{d} \right)^2 \right. \\
&\quad \left. - \bar{\rho} \frac{C_{b1}}{\kappa^2} f_{t2}' \chi' \left( \frac{\bar{\nu}}{d} \right)^2 + \bar{\rho} \left( C_{w1} \bar{f}_w - \frac{C_{b1}}{\kappa^2} \bar{f}_{t2} \right) 2 \frac{\bar{\nu}}{d^2} \nu' \right].
\end{aligned}
$$

这里：

$$ f_w' := \frac{\partial \bar{f}_w}{\partial r} = \frac{\bar{f}_w}{g} \frac{C_{w6}^3}{g^6+C_{w3}^6} (1-C_{w2}+6C_{w2} \bar{r}^5) $$

其中 $g:=r +C_{w2} (r^6-r)$

考虑交叉扩散项 $X = \frac{C_{b2}}{\sigma} \rho (\nabla \tilde{\nu})^2$ 的求导:

$$
X' = \frac{C_{b2}}{\sigma} \left[ \rho' (\nabla \bar{\nu})^2 + 2 \bar{\rho} \nabla \bar{\nu} \cdot \nabla \nu' \right]
$$

考虑压缩性修正项 $C = - C_5 \frac{\rho \tilde{\nu}^2 S^2}{\gamma R T}$ 的求导:

$$
\begin{aligned}
C' &= - \frac{C_5}{\gamma R} \left[ \frac{\rho' \bar{\nu}^2 \bar{S}^2}{\bar{T}} + \frac{2 \bar{\rho} \bar{\nu} \bar{S}^2}{\bar{T}} \nu' + \frac{\bar{\rho} \bar{\nu}^2 (S^2)'}{\bar{T}} - \frac{\bar{\rho} \bar{\nu}^2 \bar{S}^2}{\bar{T}^2} T' \right] \\
&= -2 C_5 \frac{\bar{\rho} \bar{\nu}^2}{\gamma R \bar{T}} \left[ 2 \frac{\partial \bar{u}}{\partial x} \frac{\partial u'}{\partial x} + 2 \frac{\partial \bar{v}}{\partial y} \frac{\partial v'}{\partial y} + \left( \frac{\partial \bar{u}}{\partial y} + \frac{\partial \bar{v}}{\partial x} \right) \left( \frac{\partial u'}{\partial y} + \frac{\partial v'}{\partial x} \right) \right] \\
&\quad - 2 C_5 \frac{\bar{\rho} \bar{\nu} \bar{S}^2}{\gamma R \bar{T}} \nu' - C_5 \frac{\bar{\rho} \bar{\nu}^2 \bar{S}^2}{\gamma R \bar{T}} \rho' + C_5 \frac{\bar{\rho} \bar{\nu}^2 \bar{S}^2}{\gamma R \bar{T}^2} T'
\end{aligned}
$$

以上内容的合并极其繁杂,这里不再赘述.

## 4. 构建空间离散

一个单元的空间离散,在原文框架下讲一定能表示为周围13个单元的线性组合.原文对无粘对流项、粘性对流项和扩散项采取了不同的离散方法.

### 4.1 无粘对流项

无粘对流项可以写成 $\frac{\partial F q'}{\partial x}+\frac{\partial G q'}{\partial y}$ , 其中两个矩阵写为:

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

其中 $p = \rho R T$, $H = c_p T + \frac{1}{2} (u^2 + v^2)$ .由于这两个矩阵定义在绝对空间坐标上,因此在计算时需要转移到随体坐标:

$$\frac{\partial F q'}{\partial x}+\frac{\partial G q'}{\partial y} = J \left(\frac{\partial \tilde{F} \tilde{q'}}{\partial s}+\frac{\partial \tilde{G} \tilde{q}'}{\partial n} \right)$$

其中 $J$ 是Jacobi阵.在连续性、动量、能量方程的对流项将使用基于三阶迎风格式和四阶中心差分格式混合的格式.四阶中心差分格式的表述如下,假设当前的单元是`(i,j)`,其东侧面`E`为`(i+1/2,j)`,其面上扩散项的**影响矩阵**可以插值为

$$\boldsymbol{S}_{\mathrm{mid}}(\tilde{A}_{i+\frac{1}{2},j}) = \ominus \frac{1}{12} \tilde{A}_{i-1,j} \oplus \frac{7}{12} \tilde{A}_{i,j} \oplus \frac{7}{12} \tilde{A}_{i+1,j} \ominus \frac{1}{12} \tilde{A}_{i+2,j} $$

这里的 $\oplus$ 和 $\ominus$ 是一个抽象的算子,其被定义为 $\oplus P = P I_\kappa$ ,经过这些算子将组装一个若干维行向量.通俗的讲,就是把面上对流项插值为若干个单元的物理量向量 $\hat{q}$ 的线性组合

三阶迎风格式的表述如下,假设当前单元是`(i,j)`,其东侧面是`E`,其面上扩散项的影响矩阵可以插值为

$$\boldsymbol{S}_{\mathrm{up}}(\tilde{A}_{i+\frac{1}{2},j}) = \frac{1}{2} \left( (1+\mathrm{sgn}(\hat{u}_n)) \tilde{A}_{i+\frac{1}{2},j}^{-} + (1-\mathrm{sgn}(\hat{u}_n)) \tilde{A}_{i+\frac{1}{2},j}^{+} \right)$$

其中 $\hat{u}_n $ 为迎风速度,而面上正负扩散项可以表示为

$$\tilde{A}_{i+\frac{1}{2},j}^{+} = \ominus \frac{1}{3} \tilde{A}_{i+2,j} \oplus \frac{5}{6} \tilde{A}_{i+1,j} \oplus \tilde{A}_{i,j} $$

$$\tilde{A}_{i+\frac{1}{2},j}^{-} = \ominus \frac{1}{3} \tilde{A}_{i-1,j} \oplus \frac{5}{6} \tilde{A}_{i,j} \oplus \tilde{A}_{i+1,j} $$

最终构建的行向量是三阶迎风格式和四阶中心差分的加权平均,其权重系数为 $\alpha_H$

$$\boldsymbol{S}(\tilde{A}_{i+\frac{1}{2},j}) = \alpha_H \boldsymbol{S}_{\mathrm{up}} + (1-\alpha_H) \boldsymbol{S}_{\mathrm{mid}}$$

### 4.2 粘性对流项

使用一阶迎风格式.也就是说,面上的粘性对流项将完全由迎风侧的单元决定.这种做法虽然会给粘性对流项带来比较大的耗散,然而经验表明,在粘性上适当的迟滞,事实上有利于实际的模拟.

### 4.3 扩散项、源项

扩散项最难的部分是梯度的构建,这部分使用了Green-Gauss梯度重构.所谓Green-Gauss梯度构建,就是基于以下定理:

$$\int_\Omega \nabla \phi \mathrm{d} V = \int_{\partial \Omega} \phi \mathrm{d} S$$

基于有限体积的思想,这可以离散化为:

$$\overline{\nabla \phi} V = \sum_{\partial \Omega} \phi_f \boldsymbol{S} $$

其中 $\phi_f$ 是面中心上的物理量,如何构建这个面上物理量也是见仁见智的,你可以选择不同阶的,但是由于梯度是很难算准的,一般情况下把两个单元的平均值做二阶插值就完了.当然,刚才构建的是中心处梯度,构建面上梯度,也基本使用了二阶中心插值.

在实际的代码操作中,写梯度的影响矩阵是很麻烦的,因此强烈推荐使用一种链式的网格存储手段,这样寻址的方便性似乎会变高.
