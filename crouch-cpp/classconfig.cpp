// classconfig.cpp —— cell_class / face_class 实现
#include "classconfig.h"

#include <stdexcept>
#include <utility>

namespace cc {

// ———————————————————— cell_class ————————————————————
cell_class::cell_class(std::pair<int, int> index_, double x_, double y_, double rho_,
                       double u_, double v_, double T_, double miubl_, double vol_, double sad_)
    : index(index_), x(x_), y(y_), rho(rho_), u(u_), v(v_), T(T_),
      miubl(miubl_), vol(vol_), sad(sad_) {
    H = cp * T + 0.5 * (u * u + v * v);       // 焓
    influence.fill(Mat5::Zero());             // 13个全零5×5矩阵
    cell_convect_mat();                       // 构建对流项矩阵
}

void cell_class::form_influence(int idx, const Mat5& A) { influence[idx] += A; }

void cell_class::cell_convect_mat() {
    F << u, rho, 0, 0, 0,
         u * u + R * T, 2 * rho * u, 0, rho * R, 0,
         u * v, rho * v, rho * u, 0, 0,
         u * H, rho * (H + u * u), rho * u * v, rho * u * cp, 0,
         0, 0, 0, 0, 0;
    G << v, 0, rho, 0, 0,
         u * v, rho * v, rho * u, 0, 0,
         v * v + R * T, 0, 2 * rho * v, rho * R, 0,
         v * H, rho * u * v, rho * (H + v * v), rho * v * cp, 0,
         0, 0, 0, 0, 0;
}

void cell_class::cell_jacobi() {
    const auto m_w = west->mid, m_e = east->mid;
    const auto m_s = south->mid, m_n = north->mid;
    jacobian(0, 0) = m_e.first - m_w.first;    // s_vec.x
    jacobian(0, 1) = m_e.second - m_w.second;  // s_vec.y
    jacobian(1, 0) = m_n.first - m_s.first;    // n_vec.x
    jacobian(1, 1) = m_n.second - m_s.second;  // n_vec.y
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> cell_class::viscous_convect_vec() const {
    Eigen::VectorXd v1(5), v2(5);
    v1 << u * miubl, rho * miubl, 0, 0, rho * u;
    v2 << v * miubl, 0, rho * miubl, 0, rho * v;
    return {v1, v2};
}

// ———————————————————— face_class ————————————————————
face_class::face_class(std::string direction_, std::pair<double, double> mid_,
                       const Eigen::Matrix2d& jacobi_, cell_class* me_, cell_class* nei_)
    : direction(std::move(direction_)), me(me_), nei(nei_), mid(mid_), jacobian(jacobi_) {
    form_physics();         // 形成面上物理量,二阶中心差分
    recognize_direction();  // 将面与单元的槽位对应起来
}

void face_class::recognize_direction() {
    if (direction == "NS") {
        south = nei;
        north = me;
        east = west = nullptr;
    } else if (direction == "WE") {
        west = nei;
        east = me;
        north = south = nullptr;
    } else {
        throw std::runtime_error("face_class: direction must be 'NS' or 'WE'");
    }
}

void face_class::form_physics() {
    rho = (me->rho + nei->rho) / 2;
    u = (me->u + nei->u) / 2;
    v = (me->v + nei->v) / 2;
    T = (me->T + nei->T) / 2;
    H = (me->H + nei->H) / 2;
    miubl = (me->miubl + nei->miubl) / 2;
}

void face_class::grad_2nd_mid() {
    ugrad = (me->ugrad + nei->ugrad) / 2;
    vgrad = (me->vgrad + nei->vgrad) / 2;
    Tgrad = (me->Tgrad + nei->Tgrad) / 2;
    miublgrad = (me->miublgrad + nei->miublgrad) / 2;
}

// ———————————————————— 全局存储 ————————————————————
std::vector<std::vector<cell_class*>> CellList;
std::vector<face_class*> FaceList_WE;
std::vector<face_class*> FaceList_NS;
std::vector<std::unique_ptr<cell_class>> cell_owner;
std::vector<std::unique_ptr<face_class>> face_owner;

cell_class* new_cell(std::pair<int, int> index, double x, double y, double rho,
                     double u, double v, double T, double miubl, double vol, double sad) {
    auto p = std::make_unique<cell_class>(index, x, y, rho, u, v, T, miubl, vol, sad);
    cell_class* raw = p.get();
    cell_owner.push_back(std::move(p));
    return raw;
}

face_class* new_face(std::string direction, std::pair<double, double> mid,
                     const Eigen::Matrix2d& jacobi, cell_class* me, cell_class* nei) {
    auto p = std::make_unique<face_class>(std::move(direction), mid, jacobi, me, nei);
    face_class* raw = p.get();
    face_owner.push_back(std::move(p));
    return raw;
}

void HALO_cellinit(int S_MAX_, int N_MAX_) {
    CellList.clear();
    CellList.resize(S_MAX_ + 2 * HALO + 1);
    for (auto& row : CellList) row.assign(N_MAX_ + 2 * HALO + 1, nullptr);
}

cell_class* goto_HALOcell(int s, int n) { return CellList[s + HALO][n + HALO]; }

}  // namespace cc
