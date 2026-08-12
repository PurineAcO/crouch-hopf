// convect.h —— 对流项离散(三阶迎风 + 四阶中心 + 混合)
#pragma once

#include "classconfig.h"

#include <array>

namespace cc {

// 将 cell_nei 的 F/G 按 cell_me 进行 jacobi 变换
std::pair<Mat5, Mat5> convect_sum_jacobi(cell_class* cell_me, cell_class* cell_nei);

// 粘性对流项 jacobi 变换(返回两个5维向量)
std::pair<Eigen::VectorXd, Eigen::VectorXd> viscous_convect_sum_jacobi(cell_class* cell_me,
                                                                       cell_class* cell_nei);

// 四阶中心差分格式
std::array<Mat5, 13> face_convect_mat_4th_mid(cell_class* cell);

// 三阶迎风格式
std::array<Mat5, 13> face_convect_mat_3rd_upwind(cell_class* cell);

// 粘性项一阶迎风格式(返回13个5维向量)
std::array<Eigen::VectorXd, 13> viscous_convect_1st_upwind(cell_class* cell);

// 混合对流项: 三阶迎风αH + 四阶中心(1-αH), 写入 cell.influence
void convect_hybrid(cell_class* cell);

}  // namespace cc
