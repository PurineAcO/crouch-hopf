// eigmain.h —— ARPACK(Spectra) 广义特征值求解
#pragma once

#include "formmat.h"

namespace cc {

// 求解广义特征值问题 S x = λ T x (shift-invert, 实移位 sigma), 保存 result.csv
int solve_eig(const SparseMatrix& S, const SparseMatrix& T, int k = 20, double sigma = 0.0,
              int maxiter = 3000, double tol = 1e-8, bool save = true);

}  // namespace cc
