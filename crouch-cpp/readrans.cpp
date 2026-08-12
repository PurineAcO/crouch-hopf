// readrans.cpp —— 读取网格/流场数据并构建数据结构
#include "readrans.h"

#include "classconfig.h"

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace cc {

namespace {

// 读取文件首行表头, 返回 列名->索引 映射
std::map<std::string, int> read_header(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot open file: " + path);
    std::string line;
    std::getline(f, line);
    std::istringstream iss(line);
    std::map<std::string, int> col;
    std::string name;
    int idx = 0;
    while (iss >> name) col[name] = idx++;
    return col;
}

}  // namespace

std::pair<int, int> get_scale(const std::string& ransdata) {
    std::ifstream f(ransdata);
    if (!f.is_open()) throw std::runtime_error("cannot open file: " + ransdata);
    std::string line;
    std::getline(f, line);  // 跳过表头
    int s_max = 0, n_max = 0;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        int s, n;
        iss >> s >> n;
        if (s > s_max) s_max = s;
        if (n > n_max) n_max = n;
    }
    return {s_max, n_max};
}

void read_cells(const std::string& ransdata, int S_MAX_, int N_MAX_, int h) {
    auto col = read_header(ransdata);
    std::ifstream f(ransdata);
    if (!f.is_open()) throw std::runtime_error("cannot open file: " + ransdata);
    std::string line;
    std::getline(f, line);  // 跳过表头

    HALO_cellinit(S_MAX_, N_MAX_);

    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::vector<double> vals;
        double v;
        while (iss >> v) vals.push_back(v);
        int i = static_cast<int>(vals[col.at("s")]);
        int j = static_cast<int>(vals[col.at("n")]);
        CellList[i + h][j + h] = new_cell({i, j}, vals[col.at("x")], vals[col.at("y")],
                                          vals[col.at("rho")], vals[col.at("u")],
                                          vals[col.at("v")], vals[col.at("T")],
                                          vals[col.at("miubl")], vals[col.at("vol")],
                                          vals[col.at("sad")]);
    }
    fill_ghost(S_MAX_, N_MAX_, h);
}

void fill_ghost(int S_MAX_, int N_MAX_, int h) {
    // 物面虚层: 镜像 (n=1-k)
    for (int k = 1; k <= h; k++) {
        for (int s = 1; s <= S_MAX_; s++) {
            cell_class* c = CellList[s + h][k + h];
            CellList[s + h][1 - k + h] =
                new_cell({s, k}, 0.0, 0.0, c->rho, -c->u, -c->v, c->T, -c->miubl, c->vol, c->sad);
        }
    }
    // 远场虚层: 对称 (n=N_MAX+k)
    for (int k = 1; k <= h; k++) {
        for (int s = 1; s <= S_MAX_; s++) {
            cell_class* c = CellList[s + h][N_MAX_ + 1 - k + h];
            CellList[s + h][N_MAX_ + k + h] =
                new_cell({s, N_MAX_ + 1 - k}, 0.0, 0.0, c->rho, c->u, c->v, c->T, c->miubl, c->vol, c->sad);
        }
    }
    // 周期虚列: 循环 (s=1-k 与 s=S_MAX+k)
    for (int n = 1; n <= N_MAX_; n++) {
        for (int k = 1; k <= h; k++) {
            cell_class* c_hi = CellList[S_MAX_ + 1 - k + h][n + h];
            CellList[h + 1 - k][n + h] =
                new_cell({S_MAX_ + 1 - k, n}, 0.0, 0.0, c_hi->rho, c_hi->u, c_hi->v,
                         c_hi->T, c_hi->miubl, c_hi->vol, c_hi->sad);
            cell_class* c_lo = CellList[k + h][n + h];
            CellList[S_MAX_ + k + h][n + h] =
                new_cell({k, n}, 0.0, 0.0, c_lo->rho, c_lo->u, c_lo->v,
                         c_lo->T, c_lo->miubl, c_lo->vol, c_lo->sad);
        }
    }
}

int detect_orient(const std::string& edgedata) {
    std::ifstream f(edgedata);
    if (!f.is_open()) throw std::runtime_error("cannot open file: " + edgedata);
    std::string line;
    std::getline(f, line);  // 跳过表头
    double m1x = 0, m1y = 0, t1x = 0, t1y = 0;
    bool have_m1 = false;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::string etype;
        int s, n;
        iss >> etype >> s >> n;
        if (etype != "NS" || n != 1) continue;
        // 跳过 idx c1_s c1_n c1_idx c2_s c2_n c2_idx, 其后是 nx ny mx my
        double nx, ny, mx, my;
        std::string junk;
        for (int i = 0; i < 7; i++) iss >> junk;
        iss >> nx >> ny >> mx >> my;
        if (s == 1) {
            m1x = mx; m1y = my; t1x = nx; t1y = ny;
            have_m1 = true;
        } else if (s == 2 && have_m1) {
            double ring_x = mx - m1x, ring_y = my - m1y;   // s 增大方向
            double t_ccw_x = -t1y, t_ccw_y = t1x;          // 逆时针排列时的切向
            return (ring_x * t_ccw_x + ring_y * t_ccw_y > 0) ? 1 : -1;
        }
    }
    throw std::runtime_error("边数据缺少 NS 面");
}

void form_edge(const std::string& edgedata, int h) {
    int orient = detect_orient(edgedata);

    FaceList_WE.clear();
    FaceList_NS.clear();

    std::ifstream f(edgedata);
    if (!f.is_open()) throw std::runtime_error("cannot open file: " + edgedata);
    std::string line;
    std::getline(f, line);  // 跳过表头

    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::string etype;
        int s, n, idx, c1_s, c1_n, c1_idx, c2_s, c2_n, c2_idx;
        double nx, ny, mx, my;
        iss >> etype >> s >> n >> idx >> c1_s >> c1_n >> c1_idx >> c2_s >> c2_n >> c2_idx >> nx >> ny >> mx >> my;

        if (etype == "NS") {
            // 处理NS面的时候,需要考虑远场壁面的虚单元
            cell_class* nei = (c1_n == 0) ? CellList[s + h][h] : CellList[c1_s + h][c1_n + h];
            cell_class* me = (c2_n == 0) ? CellList[s + h][n + h] : CellList[c2_s + h][c2_n + h];
            Eigen::Matrix2d jac;
            jac << nx, ny, orient * (-ny), orient * nx;   // 切向沿s增大方向
            face_class* face = new_face("NS", {mx, my}, jac, me, nei);
            me->south = face;
            nei->north = face;
            FaceList_NS.push_back(face);
        } else if (etype == "WE") {
            cell_class* nei = CellList[c1_s + h][c1_n + h];
            cell_class* me = CellList[c2_s + h][c2_n + h];
            Eigen::Matrix2d jac;
            jac << nx, ny, -ny, nx;                       // 切向沿 n 增大方向
            face_class* face = new_face("WE", {mx, my}, jac, me, nei);
            me->west = face;
            nei->east = face;
            FaceList_WE.push_back(face);
        }
    }

    int s_max = static_cast<int>(CellList.size()) - 2 * h - 1;
    int n_max = static_cast<int>(CellList[0].size()) - 2 * h - 1;
    // 物理单元 jacobi
    for (int s = 1; s <= s_max; s++)
        for (int n = 1; n <= n_max; n++) CellList[s + h][n + h]->cell_jacobi();
    // 物面虚层 jacobian: 第二行取反
    for (int n = 1 - h; n < 1; n++)
        for (int s = 1; s <= s_max; s++) {
            cell_class* src = CellList[s + h][1 - n + h];
            cell_class* dst = CellList[s + h][n + h];
            dst->jacobian(0, 0) = src->jacobian(0, 0);
            dst->jacobian(0, 1) = src->jacobian(0, 1);
            dst->jacobian(1, 0) = -src->jacobian(1, 0);
            dst->jacobian(1, 1) = -src->jacobian(1, 1);
        }
    // 远场虚层 jacobian: 复制
    for (int n = n_max + 1; n <= n_max + h; n++)
        for (int s = 1; s <= s_max; s++) {
            cell_class* src = CellList[s + h][2 * n_max + 1 - n + h];
            CellList[s + h][n + h]->jacobian = src->jacobian;
        }
    // 周期虚列 jacobian: 复制
    for (int s = 1 - h; s < 1; s++)
        for (int n = 1; n <= n_max; n++) {
            cell_class* src = CellList[s_max + s + h][n + h];
            CellList[s + h][n + h]->jacobian = src->jacobian;
        }
    for (int s = s_max + 1; s <= s_max + h; s++)
        for (int n = 1; n <= n_max; n++) {
            cell_class* src = CellList[s - s_max + h][n + h];
            CellList[s + h][n + h]->jacobian = src->jacobian;
        }
    // 物面/远场虚单元的 south/north 伪面(供链式访问, 贡献为0)
    Eigen::Matrix2d zero_jac = Eigen::Matrix2d::Zero();
    for (int s = 1; s <= s_max; s++) {
        cell_class* g0 = CellList[s + h][h];
        cell_class* gm1 = CellList[s + h][h - 1];
        g0->south = new_face("NS", {0.0, 0.0}, zero_jac, g0, gm1);
        cell_class* gN1 = CellList[s + h][n_max + 1 + h];
        cell_class* gN2 = CellList[s + h][n_max + 2 + h];
        gN1->north = new_face("NS", {0.0, 0.0}, zero_jac, gN2, gN1);
    }
    // 虚层 east/west 槽位: 补零 jacobian 伪面
    for (int s = 1; s <= s_max; s++) {
        cell_class* g0 = CellList[s + h][h];
        cell_class* gN1 = CellList[s + h][n_max + 1 + h];
        g0->east = new_face("WE", {0.0, 0.0}, zero_jac, g0, g0);
        g0->west = new_face("WE", {0.0, 0.0}, zero_jac, g0, g0);
        gN1->east = new_face("WE", {0.0, 0.0}, zero_jac, gN1, gN1);
        gN1->west = new_face("WE", {0.0, 0.0}, zero_jac, gN1, gN1);
    }
}

void read_rans(const std::string& ranspath, const std::string& edgepath) {
    auto [s_max, n_max] = get_scale(ranspath);
    S_MAX = s_max;
    N_MAX = n_max;
    read_cells(ranspath, S_MAX, N_MAX, HALO);
    form_edge(edgepath, HALO);
}

}  // namespace cc
