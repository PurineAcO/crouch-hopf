import numpy as np
import classconfig as cc

def get_scale(ransdata:str):
    """自动获知网格规模"""
    with open(ransdata, encoding="utf-8") as f:
        header = f.readline().split()
    data = np.loadtxt(ransdata, skiprows=1)
    col = {name: idx for idx, name in enumerate(header)}
    return int(data[:, col["s"]].max()), int(data[:, col["n"]].max())


def read_cells(ransdata:str,S_MAX:int,N_MAX:int):
    """读取`ransdata.txt`, 物理单元填入`CellList[s+HALO][n+HALO]`, 随后填充全部虚单元"""

    with open(ransdata, encoding="utf-8") as f:
        header = f.readline().split()

    # 数据列: s n x y sad vol rho u v T miubl E_s E_n E_idx W_s W_n W_idx N_s N_n N_idx S_s S_n S_idx
    data = np.loadtxt(ransdata, skiprows=1)
    col = {name: idx for idx, name in enumerate(header)}
    s = data[:, col["s"]].astype(int)
    n = data[:, col["n"]].astype(int)
    x = data[:, col["x"]]
    y = data[:, col["y"]]
    rho = data[:, col["rho"]]
    u = data[:, col["u"]]
    v = data[:, col["v"]]
    T = data[:, col["T"]]
    miubl = data[:, col["miubl"]]

    # halo 化下标空间: s=-HALO⋯S_MAX+HALO, n=-HALO⋯N_MAX+HALO, 角落保持 None
    cc.CellList.clear()
    cc.CellList.extend([[None] * (N_MAX + 2 * cc.HALO + 1) for _ in range(S_MAX + 2 * cc.HALO + 1)])
    for k in range(len(data)):
        i, j = s[k], n[k]
        cc.CellList[i + cc.HALO][j + cc.HALO] = cc.cell_class((i, j), x[k], y[k], rho[k], u[k], v[k], T[k], miubl[k])
    fill_ghost(S_MAX, N_MAX,cc.HALO)


def fill_ghost(S_MAX:int,N_MAX:int,h:int = cc.HALO):
    """填充 halo 虚单元"""
    # 物面虚层: 镜像
    for k in range(1, h + 1):
        for s in range(1, S_MAX + 1):
            c = cc.CellList[s + h][k + h]
            cc.CellList[s + h][1 - k + h] = cc.cell_class((s, 1 - k), 0.0, 0.0,
                                                          c.rho, -c.u, -c.v, c.T, -c.miubl)
    # 远场虚层: 对称
    for k in range(1, h + 1):
        for s in range(1, S_MAX + 1):
            c = cc.CellList[s + h][N_MAX + 1 - k + h]
            cc.CellList[s + h][N_MAX + k + h] = cc.cell_class((s, N_MAX + k), 0.0, 0.0,
                                                              c.rho, c.u, c.v, c.T, c.miubl)
    # 周期虚列: 循环
    for n in range(1, N_MAX + 1):
        for k in range(1, h + 1):
            c_hi = cc.CellList[S_MAX + 1 - k + h][n + h]
            cc.CellList[h + 1 - k][n + h] = cc.cell_class((1 - k, n), 0.0, 0.0,c_hi.rho,
                                                            c_hi.u, c_hi.v, c_hi.T, c_hi.miubl)
            c_lo = cc.CellList[k + h][n + h]
            cc.CellList[S_MAX + k + h][n + h] = cc.cell_class((S_MAX + k, n), 0.0, 0.0,
                                                              c_lo.rho, c_lo.u, c_lo.v, c_lo.T, c_lo.miubl)


def detect_orient(edgedata:str) -> int:
    """检测`NS`面的环方向,返回`1/-1`表示逆时针/顺时针排列"""
    m1 = t1 = None
    with open(edgedata, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.split()
            if p[0] != "NS" or p[2] != "1":
                continue
            s = int(p[1])
            nx, ny = float(p[10]), float(p[11])
            mx, my = float(p[12]), float(p[13])
            if s == 1:
                m1, t1 = (mx, my), (nx, ny)
            elif s == 2:
                ring = (mx - m1[0], my - m1[1])          # s 增大方向
                t_ccw = (-t1[1], t1[0])                  # 逆时针排列时的切向
                return 1 if ring[0] * t_ccw[0] + ring[1] * t_ccw[1] > 0 else -1
    raise RuntimeError("边数据缺少 NS 面")


def form_edge(edgedata:str):

    orient = detect_orient(edgedata)

    for row in cc.CellList:
        for cell in row:
            if cell is not None:
                cell.face = [None, None, None, None]
    cc.FaceList_WE.clear()
    cc.FaceList_NS.clear()

    with open(edgedata, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.split()
            etype, s, n = p[0], int(p[1]), int(p[2])
            c1_s, c1_n = int(p[4]), int(p[5])
            c2_s, c2_n = int(p[7]), int(p[8])
            nx, ny = float(p[10]), float(p[11])
            mx, my = float(p[12]), float(p[13])

            if etype == "NS":
                nei = cc.CellList[s + cc.HALO][cc.HALO] if c1_n == 0 else cc.CellList[c1_s + cc.HALO][c1_n + cc.HALO] # 源文件要求0是边界符号
                me = cc.CellList[s + cc.HALO][n + cc.HALO] if c2_n == 0 else cc.CellList[c2_s + cc.HALO][c2_n + cc.HALO]
                jac = [[nx, ny], [orient * (-ny), orient * nx]]  # 切向沿s增大方向
                face = cc.face_class("NS", (mx, my), jac, me, nei)
                me.south = face
                nei.north = face
                cc.FaceList_NS.append(face)
            elif etype == "WE":
                nei = cc.CellList[c1_s + cc.HALO][c1_n + cc.HALO]
                me = cc.CellList[c2_s + cc.HALO][c2_n + cc.HALO]
                jac = [[nx, ny], [-ny, nx]]                  # 切向沿 n 增大方向
                face = cc.face_class("WE", (mx, my), jac, me, nei)
                me.west = face
                nei.east = face
                cc.FaceList_WE.append(face)

    s_max = len(cc.CellList) - 2*cc.HALO - 1
    n_max = len(cc.CellList[0]) - 2*cc.HALO - 1
    for s in range(1, s_max + 1):
        for n in range(1, n_max + 1):
            cc.CellList[s + cc.HALO][n + cc.HALO].cell_jacobi()
    for s in range(-cc.HALO, s_max + cc.HALO + 1):
        for n in range(-cc.HALO, n_max + cc.HALO + 1):
            cell = cc.CellList[s + cc.HALO][n + cc.HALO]
            if cell is None:
                continue
            if 1 <= s <= s_max and 1 <= n <= n_max:
                continue
            if n < 1:
                src = cc.CellList[s + cc.HALO][1 - n + cc.HALO]
                cell.jacobian = [list(src.jacobian[0]), [-src.jacobian[1][0], -src.jacobian[1][1]]]
            elif n > n_max:
                src = cc.CellList[s + cc.HALO][2*n_max + 1 - n + cc.HALO]
                cell.jacobian = [list(src.jacobian[0]), list(src.jacobian[1])]
            elif s < 1:
                src = cc.CellList[s_max + s + cc.HALO][n + cc.HALO]
                cell.jacobian = [list(src.jacobian[0]), list(src.jacobian[1])]
            else:
                src = cc.CellList[s - s_max + cc.HALO][n + cc.HALO]
                cell.jacobian = [list(src.jacobian[0]), list(src.jacobian[1])]
    for s in range(1, s_max + 1):
        g0 = cc.CellList[s + cc.HALO][cc.HALO]
        gm1 = cc.CellList[s + cc.HALO][cc.HALO - 1]
        g0.south = cc.face_class("NS", (0.0, 0.0), [[0.0, 0.0], [0.0, 0.0]], g0, gm1)
        g90 = cc.CellList[s + cc.HALO][n_max + 1 + cc.HALO]
        g91 = cc.CellList[s + cc.HALO][n_max + 2 + cc.HALO]
        g90.north = cc.face_class("NS", (0.0, 0.0), [[0.0, 0.0], [0.0, 0.0]], g91, g90)

def read_rans(ranspath:str,edgepath:str):
    """读取`ransdata.txt`和`edge.txt`"""
    S_MAX, N_MAX = get_scale(ranspath)
    read_cells(ranspath,S_MAX,N_MAX)
    form_edge(edgepath)