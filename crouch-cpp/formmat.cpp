// formmat.cpp —— 稀疏矩阵组装
#include "formmat.h"

#include <Eigen/Sparse>

#include <vector>

namespace cc {

namespace {

// 13 槽位 -> (ds, dn) 偏移映射 (与 Python formmat._OFFSET 一致)
const std::pair<int, int> OFFSET[13] = {
    {0, 2}, {-1, 1}, {0, 1}, {1, 1}, {-2, 0}, {-1, 0}, {0, 0},
    {1, 0}, {2, 0},  {-1, -1}, {0, -1}, {1, -1}, {0, -2}};

// 虚单元映射矩阵
const Mat5 WALL_MAP = [] {
    Mat5 m = Mat5::Zero();
    m.diagonal() << 1.0, -1.0, -1.0, 1.0, -1.0;
    return m;
}();
const Mat5 FAR_MAP = Mat5::Identity();

// 全局三元组累积
std::vector<Eigen::Triplet<double>> trips;

// 虚单元前处理: 返回虚单元槽位 (ps, pn) 与映射矩阵 map
void ghost_target(int ns, int nn_, int& ps, int& pn, const Mat5*& map) {
    if (nn_ <= 0) {           // n=0 -> 1; n=-1 -> 2
        ps = ns;
        pn = 1 - nn_;
        map = &WALL_MAP;
    } else {                  // N_MAX+1 -> N_MAX; N_MAX+2 -> N_MAX-1
        ps = ns;
        pn = 2 * N_MAX + 1 - nn_;
        map = &FAR_MAP;
    }
}

// 时间变换逆矩阵
Mat5 primitive_map(const cell_class* cell) {
    double rho = cell->rho, u = cell->u, v = cell->v;
    double T = cell->T, nu = cell->miubl;
    double Cv = cp - R;             // 定容比热
    double q2 = u * u + v * v;
    Mat5 W;
    W << 1.0, 0.0, 0.0, 0.0, 0.0,
         -u / rho, 1.0 / rho, 0.0, 0.0, 0.0,
         -v / rho, 0.0, 1.0 / rho, 0.0, 0.0,
         -T / rho + q2 / (2.0 * Cv * rho), -u / (Cv * rho), -v / (Cv * rho), 1.0 / (Cv * rho), 0.0,
         -nu / rho, 0.0, 0.0, 0.0, 1.0 / rho;
    return W;
}

}  // namespace

void formmat(cell_class* cell) {
    int s = cell->index.first, n = cell->index.second;
    int g_self = ((n - 1) * S_MAX + (s - 1)) * 5;   // 本块行起始行
    // 边界层 (n==1 壁面, n==N_MAX 远场) 不左乘 W (同 Python 77abb24)
    bool apply_W = (n != 1 && n != N_MAX);
    Mat5 W;
    if (apply_W) W = primitive_map(cell);
    for (int k = 0; k < 13; k++) {
        Mat5 M = cell->influence[k];
        if (!M.any()) continue;
        if (apply_W) M = W * M;   // 左乘 W, 在虚邻居折叠之前
        auto [ds, dn] = OFFSET[k];
        int ns = ((s + ds - 1) % S_MAX + S_MAX) % S_MAX + 1;  // s 周期回绕
        int nn_ = n + dn;
        int col;
        if (1 <= nn_ && nn_ <= N_MAX) {           // 物理邻居: 直接铺
            col = ((nn_ - 1) * S_MAX + (ns - 1)) * 5;
        } else {                                  // 虚邻居: 系数折叠
            int ps, pn;
            const Mat5* map;
            ghost_target(ns, nn_, ps, pn, map);
            M = M * (*map);
            col = ((pn - 1) * S_MAX + (ps - 1)) * 5;
        }
        for (int i = 0; i < 5; i++) {
            int row = g_self + i;
            for (int j = 0; j < 5; j++) {
                double v = M(i, j);
                if (v != 0.0) trips.emplace_back(row, col + j, v);
            }
        }
    }
}

std::pair<SparseMatrix, SparseMatrix> build() {
    int n_phys = N_MAX * S_MAX;
    SparseMatrix S(n_phys * 5, n_phys * 5);
    S.setFromTriplets(trips.begin(), trips.end());
    S.makeCompressed();
    trips.clear();

    // T: 对角矩阵, 只在 n=2..N_MAX-1(内部层)为1
    SparseMatrix T(n_phys * 5, n_phys * 5);
    std::vector<Eigen::Triplet<double>> ttrips;
    for (int n = 2; n < N_MAX; n++)
        for (int i = (n - 1) * S_MAX * 5; i < n * S_MAX * 5; i++) ttrips.emplace_back(i, i, 1.0);
    T.setFromTriplets(ttrips.begin(), ttrips.end());
    T.makeCompressed();
    return {S, T};
}

}  // namespace cc
