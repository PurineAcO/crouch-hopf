import numpy as np
import scipy.sparse as sp
import classconfig as cc

# 槽位方向偏移映射字典
_OFFSET = {
    cc.dic['nn']: (0, 2),  cc.dic['n']: (0, 1),   cc.dic['c']: (0, 0),
    cc.dic['s']: (0, -1),  cc.dic['ss']: (0, -2),
    cc.dic['ee']: (2, 0),  cc.dic['e']: (1, 0),
    cc.dic['w']: (-1, 0),  cc.dic['ww']: (-2, 0),
    cc.dic['ne']: (1, 1),  cc.dic['nw']: (-1, 1),
    cc.dic['se']: (1, -1), cc.dic['sw']: (-1, -1),
}

# 虚单元映射字典
_WALL_MAP = np.diag([1.0, -1.0, -1.0, 1.0, -1.0])
_FAR_MAP = np.eye(5)

def _ghost_target(ns:int, nn_:int):
    """虚单元前处理,返回虚单元槽位`ns`,`nn_`和映射矩阵`MAP`"""
    if nn_ <= 0:
        return ns, 1 - nn_, _WALL_MAP            # n=0 -> 1; n=-1 -> 2
    return ns, 2 * cc.N_MAX + 1 - nn_, _FAR_MAP  # N_MAX+1 -> N_MAX; N_MAX+2 -> N_MAX-1

def _primitive_map(cell:cc.cell_class):
    """左乘的时间变换逆矩阵`W`."""
    rho, u, v, T, nu = cell.rho, cell.u, cell.v, cell.T, cell.miubl
    Cv = cc.cp - cc.R
    q2 = u * u + v * v
    return np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [-u / rho, 1.0 / rho, 0.0, 0.0, 0.0],
        [-v / rho, 0.0, 1.0 / rho, 0.0, 0.0],
        [-T / rho + q2 / (2.0 * Cv * rho), -u / (Cv * rho), -v / (Cv * rho), 1.0 / (Cv * rho), 0.0],
        [-nu / rho, 0.0, 0.0, 0.0, 1.0 / rho],
    ])

# 稀疏矩阵三元组
_rows, _cols, _vals = [], [], []

def formmat(cell:cc.cell_class):
    """开始写稀疏矩阵"""
    s, n = cell.index
    g_self = ((n - 1) * cc.S_MAX + (s - 1)) * 5     # 本块行起始行
    W = None if (n == 1 or n == cc.N_MAX) else _primitive_map(cell)
    for k in range(13):
        M = cell.influence[k]
        if not M.any():
            continue
        if W is not None:
            M = W @ M
        ds, dn = _OFFSET[k]
        ns = ((s + ds - 1) % cc.S_MAX) + 1        # s 周期回绕
        nn_ = n + dn
        if 1 <= nn_ <= cc.N_MAX:                  # 物理邻居: 直接铺
            col = ((nn_ - 1) * cc.S_MAX + (ns - 1)) * 5
        else:                                     # 虚邻居: 系数折叠
            ps, pn, Map = _ghost_target(ns, nn_)
            M = M @ Map
            col = ((pn - 1) * cc.S_MAX + (ps - 1)) * 5
        for i in range(5):
            row = g_self + i
            for j in range(5):
                v = M[i, j]
                if v != 0.0:
                    _rows.append(row)
                    _cols.append(col + j)
                    _vals.append(v)

def build():
    """形成`S`,`T`矩阵"""
    n_phys = cc.N_MAX * cc.S_MAX
    S = sp.csr_matrix((_vals, (_rows, _cols)), shape=(n_phys * 5, n_phys * 5))
    S.sum_duplicates()
    d = np.zeros(n_phys * 5)
    for n in range(2, cc.N_MAX):
        d[(n - 1) * cc.S_MAX * 5: n * cc.S_MAX * 5] = 1.0
    T = sp.diags(d)
    cc.BigMatrix = S
    cc.TMatrix = T
    return S, T