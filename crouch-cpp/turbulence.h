// turbulence.h —— Spalart-Allmaras 湍流模型: 参数/扩散项/源项
#pragma once

#include "classconfig.h"

#include <array>

namespace cc {

// 计算 SA 模型引起的有效粘度系数 μeff 等参数
void SA_calc_constants(cell_class* cell);

// 对面上的湍流字典进行二阶中心插值并构建切应力
void diffusion_2nd_mid_SA(face_class* face);

// 计算单元扩散项矩阵, 累加到 cell.influence
void cell_diffusion(cell_class* cell);

// 计算面上的湍流扩散项, 返回 [D6,D7,D2,D3,D10,D11,D8,D5]
std::array<Mat5, 8> face_diffusion(face_class* face);

// 单元上的源项矩阵, 直接构建在 cell.influence
void cell_source(cell_class* cell);

}  // namespace cc
