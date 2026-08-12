// grad.h —— Green-Gauss 梯度重构
#pragma once

#include "classconfig.h"

#include <map>
#include <string>

namespace cc {

// 计算某个 face 的梯度影响矩阵(Green-Gauss), 会与8个网格挂钩, 返回字典
std::map<std::string, Eigen::Vector2d> green_gauss_face_vari(face_class* face);

// 计算某个 cell 的梯度影响矩阵(Green-Gauss), 会与5个网格挂钩, 返回字典
std::map<std::string, Eigen::Vector2d> green_gauss_cell_vari(cell_class* cell);

// 基于 Green-Gauss 的单元梯度构建, 按北、南、东、西的顺序
void green_gauss_from_JST(cell_class* cell, face_class* facenorth, face_class* facesouth,
                          face_class* faceeast, face_class* facewest);

}  // namespace cc
