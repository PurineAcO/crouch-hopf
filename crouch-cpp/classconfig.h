// classconfig.h —— cell/face 数据结构与全局存储(转写自 crouch/classconfig.py)
#pragma once

#include "config.h"

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cc {

struct face_class;
struct cell_class;

// —— 单元类 ——
struct cell_class {
    std::pair<int, int> index;                 // 单元编号(s,n), 从1开始
    double x = 0.0, y = 0.0;                   // 单元中心坐标
    double rho = 0.0, u = 0.0, v = 0.0;        // 密度、x/y速度
    double T = 0.0, H = 0.0;                   // 静温、焓
    double miubl = 0.0;                        // 修正湍流粘度ν̃
    double vol = 0.0, sad = 0.0;               // 体积、壁面距离
    Eigen::Matrix2d jacobian = Eigen::Matrix2d::Zero();  // 单元 jacobi 矩阵 [s_vec; n_vec]

    // 梯度
    Eigen::Vector2d ugrad = Eigen::Vector2d::Zero();
    Eigen::Vector2d vgrad = Eigen::Vector2d::Zero();
    Eigen::Vector2d Tgrad = Eigen::Vector2d::Zero();
    Eigen::Vector2d miublgrad = Eigen::Vector2d::Zero();

    // 湍流字典
    double mu_eff = 0.0, lambda_eff = 0.0, chi = 0.0, fv1 = 0.0;
    double fv2 = 0.0, fw = 0.0, ft2 = 0.0, Omega = 0.0;
    double S = 0.0, Sbl = 0.0, r = 0.0, mu = 0.0, g = 0.0;

    // 邻接面, 槽位顺序 [W, E, S, N]
    face_class* west = nullptr;
    face_class* east = nullptr;
    face_class* south = nullptr;
    face_class* north = nullptr;

    // 本单元流动量可写为13个矩阵的线性组合
    std::array<Mat5, 13> influence{};
    Mat5 F = Mat5::Zero();                     // x对流项矩阵
    Mat5 G = Mat5::Zero();                     // y对流项矩阵

    cell_class() = default;
    cell_class(std::pair<int, int> index_, double x_, double y_, double rho_,
               double u_, double v_, double T_, double miubl_, double vol_, double sad_);

    void form_influence(int idx, const Mat5& A);
    void cell_convect_mat();
    void cell_jacobi();

    // jacobi 变换: (A,B) -> (A*J00+B*J01, A*J10+B*J11), 支持矩阵或向量
    template <typename T>
    std::pair<T, T> jacobi(const T& A, const T& B) const {
        return {A * jacobian(0, 0) + B * jacobian(0, 1),
                A * jacobian(1, 0) + B * jacobian(1, 1)};
    }

    // 返回两个5维向量(用于粘性对流项一阶迎风)
    std::pair<Eigen::VectorXd, Eigen::VectorXd> viscous_convect_vec() const;
};

// —— 面类 ——
struct face_class {
    std::string direction;                     // 有且仅有 "WE" 与 "NS"
    cell_class* me = nullptr;                  // 高侧网格
    cell_class* nei = nullptr;                 // 低侧网格
    std::pair<double, double> mid{0.0, 0.0};   // 面中点
    Eigen::Matrix2d jacobian = Eigen::Matrix2d::Zero();  // [Xn,Yn; Xs,Ys]

    // 梯度
    Eigen::Vector2d ugrad = Eigen::Vector2d::Zero();
    Eigen::Vector2d vgrad = Eigen::Vector2d::Zero();
    Eigen::Vector2d Tgrad = Eigen::Vector2d::Zero();
    Eigen::Vector2d miublgrad = Eigen::Vector2d::Zero();

    // 面上的湍流字典(仅用于扩散项)
    double mu_eff = 0.0, lambda_eff = 0.0, chi = 0.0, fv1 = 0.0, mu = 0.0;
    double tauxx = 0.0, tauxy = 0.0, tauyy = 0.0;

    // 面上的物理量(二阶中心插值)
    double rho = 0.0, u = 0.0, v = 0.0, T = 0.0, H = 0.0, miubl = 0.0;

    // 槽位(指向 cell)
    cell_class* west = nullptr;
    cell_class* east = nullptr;
    cell_class* south = nullptr;
    cell_class* north = nullptr;

    face_class() = default;
    face_class(std::string direction_, std::pair<double, double> mid_,
               const Eigen::Matrix2d& jacobi_, cell_class* me_, cell_class* nei_);

    void recognize_direction();
    void form_physics();

    template <typename T>
    std::pair<T, T> jacobi(const T& A, const T& B) const {
        return {A * jacobian(0, 0) + B * jacobian(0, 1),
                A * jacobian(1, 0) + B * jacobian(1, 1)};
    }

    void grad_2nd_mid();

    double vn() const { return jacobi(u, v).first; }   // 法向速度
    double nx() const { return jacobian(0, 0); }
    double ny() const { return jacobian(0, 1); }
};

// —— 全局存储 ——
extern std::vector<std::vector<cell_class*>> CellList;   // [s][n], halo化下标
extern std::vector<face_class*> FaceList_WE;
extern std::vector<face_class*> FaceList_NS;

// 对象所有权容器(进程结束时统一释放)
extern std::vector<std::unique_ptr<cell_class>> cell_owner;
extern std::vector<std::unique_ptr<face_class>> face_owner;

// 创建 cell / face(归 owner 所有, 返回裸指针)
cell_class* new_cell(std::pair<int, int> index, double x, double y, double rho,
                     double u, double v, double T, double miubl, double vol, double sad);
face_class* new_face(std::string direction, std::pair<double, double> mid,
                     const Eigen::Matrix2d& jacobi, cell_class* me, cell_class* nei);

// halo 化下标空间: s=-HALO⋯S_MAX+HALO, n=-HALO⋯N_MAX+HALO, 角落保持 nullptr
void HALO_cellinit(int S_MAX_, int N_MAX_);
cell_class* goto_HALOcell(int s, int n);

}  // namespace cc
