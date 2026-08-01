# crouch — 方法与函数说明

## 目录

- [cell_class](#cell_class)
  - [`convect_mat`](#convect_matself)
  - [`form_influence`](#form_influenceselfindexa)
- [face_class](#face_class)
  - [`form_physics`](#form_physicsself)
  - [`jacobi`](#jacobiselfab)
- [convect](#convect)
  - [3rd upwind](#face_convect_mat_3rd_upwind)
  - [4th center](#face_convect_mat_4th_mid)
  - [hybrid](#convect_hybrid)
---

## `cell_class`

定义了单元编号,单元位置`x`,`y`,单元的参数向量$\bar{q} = \left (\bar{\rho},\bar{u},\bar{v},\bar{T},\bar{\tilde{\nu}}\right )^\top$,后面为了简略,由于在矩阵中的都是平均量,所以不加平均号了.

### `convect_mat(self)`

定义在`class cell_class`下的方法,构建矩阵$F,G$,这两个矩阵直接构建在单元中心上.

$$
F = \begin{pmatrix}
u & \rho & 0 & 0 & 0 \\
u^2 + RT & 2\rho u & 0 & \rho R & 0 \\
u v & \rho v & \rho u & 0 & 0 \\
u H & \rho(H + u^2) & \rho u v & \rho u c_p & 0 \\
0 & 0 & 0 & 0 & 0
\end{pmatrix}
,
G = \begin{pmatrix}
v & 0 & \rho & 0 & 0 \\
u v & \rho v & \rho u & 0 & 0 \\
v^2 + RT & 0 & 2\rho v & \rho R & 0 \\
v H & \rho u v & \rho(H + v^2) & \rho v c_p & 0 \\
0 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

其中 $p = \rho R T$，$H = c_p T + \frac{1}{2} (u^2 + v^2)$.

### `form_influence(self,index,A)`

将矩阵`A`**累加**进入该单元的影响行向量的`index`位中,在这个求解器中,我们要求本单元网格**有且仅有**和周围的12个网格存在关联.算上本单元自己的影响矩阵,一共有13个矩阵元素组成一个行向量`self.influence`

`self.influence`被初始化为13个全0矩阵$\boldsymbol{O}_\kappa$,$\kappa$的含义如下:

| 位置 | $\kappa$ | 位置 | $\kappa$ | 位置 | $\kappa$ | 位置 | $\kappa$ | 位置 | $\kappa$ | 位置 | $\kappa$ |
|------|----------|------|----------|------|----------|------|----------|------|----------|------|----------|
| `(i,j+2)` | 1 | `(i-1,j+1)` | 2 | `(i,j+1)` | 3 | `(i+1,j+1)` | 4 | `(i-2,j)` | 5 | `(i-1,j)` | 6 |
| `(i,j)` | 7 | `(i+1,j)` | 8 | `(i+2,j)` | 9 | `(i-1,j-1)` | 10 | `(i,j-1)` | 11 | `(i+1,j-1)` | 12 |
| `(i,j-2)` | 13 | | | | | | | | | | |

## `face_class`

面上拥有和`cell`一致的物理量种类,但是增加了关于面上法向量`(nx,ny)`的信息.定义一个面时,需要首先给出这个面的**横纵类型**,然后给出这个面的邻接单元编号,程序会自动取一种平均值计入`index`中,同时对各个物理量进行平均插值.`face_class`采取基于自身索引`index`来**直接**定位邻接网格.

### `form_physics(self)`

根据其两个邻接网格的物理量$\hat{q}$进行平均插值,以得到面上物理量$\hat{q}$的估计,这个$\hat{q}$是2阶精度的,仅可以用来匹配扩散项、源项而不用于对流项.

### `jacobi(self,A,B)`

对A,B两个量(可以是任何形式,只要是在代码层面定义了加减运算)进行jacobi变换:

$$(\tilde{A},\tilde{B})^\top = \boldsymbol{J} (A,B)^\top$$

在`face_class`里面定义了`face.jacobi`,可以直接用于该函数.函数返回一个元组,为变换的结果.本文后面凡是加波浪线的都是基于当地方向定义的物理量(及其矩阵).

## `convect`

根据原论文,在连续性、动量、能量方程的对流项将使用基于三阶迎风格式和四阶中心差分格式混合的格式

这里我们需要说明的是,面上的各个项需要转换为面上随体坐标,这部分将会被[`face_class.jacobi()`](#jacobiselfab)完成,完成后的对流项称为$\tilde{F}$和$\tilde{G}$

### `face_convect_mat_4th_mid`

四阶中心差分格式的表述如下,假设当前的单元是`(i,j)`,其东侧面`E`为`(i+1/2,j)`,其面上扩散项的**影响矩阵**可以插值为

$$\boldsymbol{S}_{\mathrm{mid}}(\tilde{A}_{i+\frac{1}{2},j}) = \ominus \frac{1}{12} \tilde{A}_{i-1,j} \oplus \frac{7}{12} \tilde{A}_{i,j} \oplus \frac{7}{12} \tilde{A}_{i+1,j} \ominus \frac{1}{12} \tilde{A}_{i+2,j} $$

这里的$\oplus$和$\ominus$是一个抽象的算子,其被定义为$\oplus P = P I_\kappa$,经过这些算子将组装一个13维行向量.通俗的讲,就是把面上对流项插值为若干个单元的物理量向量$\hat{q}$的线性组合.$\kappa$的对应表格已经在[`form_influence(self,index,A)`](#form_influenceselfindexa)给出.

### `face_convect_mat_3rd_upwind`

三阶迎风格式的表述如下,假设当前单元是`(i,j)`,其东侧面是`E`,其面上扩散项的影响矩阵可以插值为

$$\boldsymbol{S}_{\mathrm{up}}(\tilde{A}_{i+\frac{1}{2},j}) = \frac{1}{2} \left( (1+\mathrm{sgn}(\hat{u}_n)) \tilde{A}_{i+\frac{1}{2},j}^{-} + (1-\mathrm{sgn}(\hat{u}_n)) \tilde{A}_{i+\frac{1}{2},j}^{+} \right)$$

其中迎风速度$\hat{u}_n = $ `face_class.jacobi(u,v)`,而面上正负扩散项可以表示为

$$\tilde{A}_{i+\frac{1}{2},j}^{+} = \ominus \frac{1}{3} \tilde{A}_{i+2,j} \oplus \frac{5}{6} \tilde{A}_{i+1,j} \oplus \tilde{A}_{i,j} $$

$$\tilde{A}_{i+\frac{1}{2},j}^{-} = \ominus \frac{1}{3} \tilde{A}_{i-1,j} \oplus \frac{5}{6} \tilde{A}_{i,j} \oplus \tilde{A}_{i+1,j} $$

### `convect_hybrid`

最终构建的行向量是三阶迎风格式和四阶中心差分的加权平均,其权重系数为$\alpha_H$

$$\boldsymbol{S}(\tilde{A}_{i+\frac{1}{2},j}) = \alpha_H \boldsymbol{S}_{\mathrm{up}} + (1-\alpha_H) \boldsymbol{S}_{\mathrm{mid}}$$

## `readrans`

