# PurineCFD — 方法与函数说明

## 目录

- [cell_class](#cell_class)
  - [`convect_mat(self)`](#convect_matself)
- [face_class](#face_class)

---

## `cell_class`

定义了单元编号,单元位置`x`,`y`,单元的参数向量$\bar{q} = (\bar{\rho},\bar{u},\bar{v},\bar{T},\bar{\tilde{\nu}})^\top$,后面为了简略,由于在矩阵中的都是平均量,所以不加平均号了.

### `convect_mat(self)`

定义在`class cell_class`下的方法,构建矩阵$F,G$,这两个矩阵直接构建在单元中心上.

$$
F = \begin{bmatrix}
u & \rho & 0 & 0 & 0 \\
u^2 + RT & 2\rho u & 0 & \rho R & 0 \\
u v & \rho v & \rho u & 0 & 0 \\
u H & \rho(H + u^2) & \rho u v & \rho u c_p & 0 \\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
,
G = \begin{bmatrix}
v & 0 & \rho & 0 & 0 \\
u v & \rho v & \rho u & 0 & 0 \\
v^2 + RT & 0 & 2\rho v & \rho R & 0 \\
v H & \rho u v & \rho(H + v^2) & \rho v c_p & 0 \\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

其中 $p = \rho R T$，$H = c_p T + \tfrac12(u^2 + v^2)$.

## `face_class`

面上拥有和`cell`一致的物理量种类,但是增加了关于面上法向量`(nx,ny)`的信息.定义一个面时,需要首先给出这个面的**横纵类型**,然后给出这个面的邻接单元编号,程序会自动取一种平均值计入`index`中,同时对各个物理量进行平均插值.