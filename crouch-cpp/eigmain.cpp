// eigmain.cpp —— 用 Spectra(ARPACK 的 C++ 移植)求解广义特征值
#include "eigmain.h"

#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/Util/CompInfo.h>

#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cc {

namespace {

// 自定义算子: y = (S - sigma*T)^{-1} (T x)  (广义 shift-invert)
struct GenShiftOp {
    using Scalar = double;

    int n;
    const SparseMatrix* S;
    const SparseMatrix* T;
    mutable Eigen::SparseLU<SparseMatrix> solver;

    GenShiftOp(const SparseMatrix& S_, const SparseMatrix& T_, double sigma)
        : n(static_cast<int>(S_.rows())), S(&S_), T(&T_) {
        solver.compute(S_ - sigma * T_);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("SparseLU factorization of (S - sigma*T) failed");
    }

    int rows() const { return n; }
    void set_shift(double) {}

    void perform_op(const double* x_in, double* y_out) const {
        Eigen::Map<const Eigen::VectorXd> x(x_in, n);
        Eigen::Map<Eigen::VectorXd> y(y_out, n);
        y.noalias() = solver.solve(*T * x);
    }
};

}  // namespace

int solve_eig(const SparseMatrix& S, const SparseMatrix& T, int k, double sigma,
              int maxiter, double tol, bool save) {
    int n = static_cast<int>(S.rows());
    int ncv = std::min(3 * k, n);

    GenShiftOp op(S, T, sigma);
    Spectra::GenEigsRealShiftSolver<GenShiftOp> eigs(op, k, ncv, sigma);
    eigs.init();

    int nconv = static_cast<int>(
        eigs.compute(Spectra::SortRule::LargestMagn, maxiter, tol, Spectra::SortRule::LargestReal));

    if (eigs.info() != Spectra::CompInfo::Successful) {
        std::cerr << "[eig] WARNING: solver did not fully converge (" << nconv << "/" << k
                  << " converged)\n";
    }

    auto vals = eigs.eigenvalues();
    auto vecs = eigs.eigenvectors();

    // 按 Re(lambda) 降序排列(与 Python argsort(-real) 一致)
    std::vector<int> order(k);
    for (int i = 0; i < k; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return vals[a].real() > vals[b].real();
    });

    std::cout << "[eig] top " << k << " growth rates Re(lambda):\n";
    std::cout << "    Re(lambda)      Im(lambda)      growth       freq       resid\n";

    std::vector<double> res(k, 0.0);
    for (int i = 0; i < k; i++) {
        // per-mode normalized residual
        Eigen::VectorXcd lhs = S * vecs.col(i);
        Eigen::VectorXcd rhs = vals[i] * (T * vecs.col(i));
        res[i] = (lhs - rhs).norm() / ((lhs).norm() + (rhs).norm() + 1e-300);
    }

    std::vector<std::tuple<int, double, double, double, double, double>> rows;
    for (int idx = 0; idx < k; idx++) {
        int i = order[idx];
        double re = vals[i].real(), im = vals[i].imag();
        double freq = im / (2 * M_PI);
        std::cout << re << " " << im << " " << re << " " << freq << " " << res[i] << "\n";
        rows.emplace_back(idx + 1, re, im, re, freq, res[i]);
    }

    if (save) {
        std::ofstream f("result.csv");
        f << "mode,Re(lambda),Im(lambda),growth,freq,resid\n";
        for (auto& r : rows) {
            f << std::get<0>(r) << "," << std::get<1>(r) << "," << std::get<2>(r) << ","
              << std::get<3>(r) << "," << std::get<4>(r) << "," << std::get<5>(r) << "\n";
        }
        std::cout << "[eig] saved result.csv\n";
    }
    return nconv;
}

}  // namespace cc
