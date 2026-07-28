"""
readrans.py — RANS 结果可视化 + C-block 结构化重建
生成6张图:
  1. 网格点位置图（分段色带）
  2-6. rho, u, v, T, miubl 云图（tripcolor + 翼型白色遮罩）
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, KDTree
import time

# ============================================================
# 1. 读取数据
# ============================================================
t0 = time.time()
print('=== 读取数据 ===')
data = np.loadtxt('1-24-14700', delimiter=',', skiprows=1)
node = data[:, 0].astype(int)
xy   = data[:, 1:3]
rho  = data[:, 3]; u = data[:, 4]; v = data[:, 5]
T    = data[:, 6]; miubl = data[:, 7]
N    = len(node)

surf = np.loadtxt('1-20-0005', delimiter=',', skiprows=1)
surf_xy = surf[:, 1:3]  # 翼型表面坐标
print(f'体网格 {N} 点, 翼型表面 {len(surf_xy)} 点')

# 翼型表面按角度排序(从尾缘下表面→前缘→上表面→尾缘)
cx, cy = np.mean(surf_xy, axis=0)
ang = np.arctan2(surf_xy[:,1]-cy, surf_xy[:,0]-cx)
surf_ordered = surf_xy[np.argsort(ang)]

# ============================================================
# 2. 全网格 Delaunay 三角化
# ============================================================
print('Delaunay 三角化...')
tri = Delaunay(xy)
all_tri = tri.simplices
print(f'{len(all_tri)} 三角形')

# ============================================================
# 3. 绘图
# ============================================================
print('绘图...')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# --- 图1: 网格点 ---
ax = axes[0,0]
nb = 10
band = np.linspace(1, N, nb+1).astype(int)
cols = plt.cm.jet(np.linspace(0,1,nb))
for bi in range(nb):
    m = (node >= band[bi]) & (node < band[bi+1])
    ax.scatter(xy[m,0], xy[m,1], s=2, c=[cols[bi]], alpha=0.6)
ax.set_aspect('equal')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title(f'Grid points (n={N})')
ax.set_xlim(-2, 3); ax.set_ylim(-2.5, 2.5)

# --- 图2-6: 云图: tripcolor + 翼型白色遮罩 ---
zoom = (-1.5, 1.5)
vlist = [('rho',r'$\rho$'),('u',r'$u$'),('v',r'$v$'),
         ('T',r'$T$'),('miubl',r'$\nu_t$')]
var = {'rho':rho, 'u':u, 'v':v, 'T':T, 'miubl':miubl}

# 预过滤 zoom 区域内的三角形
zm = (xy[:,0]>=zoom[0]) & (xy[:,0]<=zoom[1]) & (xy[:,1]>=zoom[0]) & (xy[:,1]<=zoom[1])
zm_idx = set(np.where(zm)[0])
zm_tri = all_tri[np.all(np.isin(all_tri, list(zm_idx)), axis=1)]

# 用 alpha-shape 思想过滤跨翼型空洞的三角形
# circumradius 过滤: 保留 R < 0.02 (翼型厚度 ~0.06)
tp = xy[zm_tri]
a = np.sqrt(np.sum((tp[:,1]-tp[:,0])**2, axis=1))
b = np.sqrt(np.sum((tp[:,2]-tp[:,1])**2, axis=1))
c = np.sqrt(np.sum((tp[:,0]-tp[:,2])**2, axis=1))
v01 = tp[:,1]-tp[:,0]; v02 = tp[:,2]-tp[:,0]
area = np.maximum(0.5*np.abs(v01[:,0]*v02[:,1]-v01[:,1]*v02[:,0]), 1e-30)
R = a*b*c / (4*area)
keep = R < 0.02
good_tri = zm_tri[keep]
print(f'Zoom 三角: {len(zm_tri)} → 过滤后 {len(good_tri)} (R<0.02)')

# 翼型填充多边形（白色）
surf_x = surf_ordered[:, 0]
surf_y = surf_ordered[:, 1]

for idx, (vn, vt) in enumerate(vlist):
    ax = axes[(idx+1)//3, (idx+1)%3]
    if len(good_tri) > 0:
        tc = ax.tripcolor(xy[:,0], xy[:,1], good_tri, var[vn],
                          shading='gouraud', cmap='jet')
        # 翼型白色遮罩
        ax.fill(surf_x, surf_y, facecolor='white', edgecolor='none', zorder=3)
        ax.set_xlim(zoom); ax.set_ylim(zoom)
        ax.set_aspect('equal')
        ax.set_title(vt)
        plt.colorbar(tc, ax=ax)
    else:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center')

# 在每个云图上加翼型轮廓线
for idx in range(5):
    ax = axes[(idx+1)//3, (idx+1)%3]
    ax.plot(surf_x, surf_y, 'k-', linewidth=0.8, zorder=4)

plt.tight_layout()
plt.savefig('Fig_all.png', dpi=150, bbox_inches='tight')
print('已保存: Fig_all.png')

# ============================================================
# 4. C-block 结构化重建 (alpha-shape + 法向推进)
# ============================================================
print('\n=== C-block 结构化重建 ===')

# 4a. 用 circumradius < 0.02 过滤, 移除跨翼型三角形的连接
print('计算 circumradius (全网格)...')
tp = xy[all_tri]
a = np.sqrt(np.sum((tp[:,1]-tp[:,0])**2, axis=1))
b = np.sqrt(np.sum((tp[:,2]-tp[:,1])**2, axis=1))
c = np.sqrt(np.sum((tp[:,0]-tp[:,2])**2, axis=1))
v01 = tp[:,1]-tp[:,0]; v02 = tp[:,2]-tp[:,0]
area = np.maximum(0.5*np.abs(v01[:,0]*v02[:,1]-v01[:,1]*v02[:,0]), 1e-30)
R_all = a*b*c / (4*area)
good_tri_all = all_tri[R_all < 0.02]
print(f'Alpha-shape 三角: {len(good_tri_all)}/{len(all_tri)}')

# 4b. 用这些三角形构建邻居图
from collections import defaultdict
node_nb = defaultdict(set)
for tv in good_tri_all:
    for a,b in [(tv[0],tv[1]),(tv[1],tv[2]),(tv[2],tv[0])]:
        node_nb[a].add(b); node_nb[b].add(a)

# 4c. 在体网格中匹配表面节点
st = KDTree(surf_xy)
d,_ = st.query(xy)
surf_idx = np.where(d < 1e-8)[0]
surf_set = set(surf_idx)

# 沿角度排序表面节点 (j方向)
ang_all = np.arctan2(xy[surf_idx,1]-cy, xy[surf_idx,0]-cx)
j_order = surf_idx[np.argsort(ang_all)].tolist()
nj = len(j_order)

# 4d. 逐层推进
print(f'逐层推进: {nj} j方向...')
layers = [j_order]
used = set(j_order)

for i in range(1, 600):
    cur = layers[i-1]
    nxt = []
    
    for j, nd in enumerate(cur):
        cand = [nb for nb in node_nb[nd] if nb not in used]
        if cand:
            # 挑最近的(物理距离)
            best = min(cand, key=lambda nb: np.linalg.norm(xy[nb]-xy[nd]))
            nxt.append(best)
            used.add(best)
        else:
            nxt.append(None)
    
    n_valid = sum(1 for n in nxt if n is not None)
    if n_valid < nj * 0.2:
        print(f'  层 {i}: {n_valid}/{nj} 有效, 停止')
        break
    
    # 补 None: 用近邻的
    for j in range(nj):
        if nxt[j] is None:
            for d in range(1, nj):
                if nxt[(j+d)%nj] is not None:
                    nxt[j] = nxt[(j+d)%nj]; break
                if nxt[(j-d)%nj] is not None:
                    nxt[j] = nxt[(j-d)%nj]; break
    
    layers.append(nxt)
    if i % 50 == 0:
        print(f'  层 {i}: {n_valid} 有效, 已用 {len(used)}/{N}')

ni = len(layers)
print(f'结构化网格: {ni} x {nj}, 覆盖 {len(used)}/{N}')

# 4e. 结构数组
var_names = {'rho':rho,'u':u,'v':v,'T':T,'miubl':miubl}
struct = {}
for vn, va in var_names.items():
    arr = np.full((ni, nj), np.nan)
    for i in range(ni):
        for j in range(nj):
            nd = layers[i][j]
            if nd is not None:
                arr[i,j] = va[nd]
    struct[vn] = arr

# 物理坐标网格
Xg = np.full((ni, nj), np.nan)
Yg = np.full((ni, nj), np.nan)
for i in range(ni):
    for j in range(nj):
        nd = layers[i][j]
        if nd is not None:
            Xg[i,j] = xy[nd,0]
            Yg[i,j] = xy[nd,1]

# 4f. 结构化云图
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
for idx, vn in enumerate(['rho','u','miubl']):
    ax = axes2[idx]
    mask = ~np.isnan(struct[vn])
    if mask.any():
        ax.pcolormesh(Xg, Yg, struct[vn], cmap='jet')
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_title(f'Structured: {vn}')
    plt.colorbar(ax.collections[-1], ax=ax)
plt.tight_layout()
plt.savefig('Fig_structured.png', dpi=150)
print('已保存: Fig_structured.png')

# 4g. 保存结构化网格到文件
print(f'\n保存结构化网格 ({ni}x{nj}) 到 grid_structured.txt ...')
with open('grid_structured.txt', 'w') as f:
    f.write(f'# Structured grid: {ni} x {nj} (i方向 {ni} 层, j方向 {nj} 点/层)\n')
    f.write(f'# 列: i j x y rho u v T miubl\n')
    for i in range(ni):
        for j in range(nj):
            nd = layers[i][j]
            if nd is not None:
                f.write(f'{i} {j} {Xg[i,j]:.15e} {Yg[i,j]:.15e} '
                        f'{struct["rho"][i,j]:.15e} {struct["u"][i,j]:.15e} '
                        f'{struct["v"][i,j]:.15e} {struct["T"][i,j]:.15e} '
                        f'{struct["miubl"][i,j]:.15e}\n')
print(f'已保存: grid_structured.txt ({ni}x{nj})')

print(f'\n总耗时: {time.time()-t0:.1f}s')
