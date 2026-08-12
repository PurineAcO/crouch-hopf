// formmat.h —— 把单元 influence 组装成稀疏矩阵 S/T
#pragma once

#include "classconfig.h"

#include <Eigen/SparseCore>

#include <utility>

namespace cc {

using SparseMatrix = Eigen::SparseMatrix<double>;

// 开始写稀疏矩阵(将 cell 的13个影响矩阵铺入全局三元组)
void formmat(cell_class* cell);

// 形成 S, T 矩阵
std::pair<SparseMatrix, SparseMatrix> build();

}  // namespace cc
