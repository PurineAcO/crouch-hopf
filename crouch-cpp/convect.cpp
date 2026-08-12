// convect.cpp —— 对流项离散实现
#include "convect.h"

namespace cc {

std::pair<Mat5, Mat5> convect_sum_jacobi(cell_class* cell_me, cell_class* cell_nei) {
    return cell_me->jacobi(cell_nei->F, cell_nei->G);
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> viscous_convect_sum_jacobi(cell_class* cell_me,
                                                                       cell_class* cell_nei) {
    auto [v0, v1] = cell_nei->viscous_convect_vec();
    return cell_me->jacobi(v0, v1);
}

std::array<Mat5, 13> face_convect_mat_4th_mid(cell_class* cell) {
    std::array<Mat5, 13> influence;
    influence.fill(Mat5::Zero());
    influence[dic::nn] = 1.0 / 12 * convect_sum_jacobi(cell, cell->north->north->north->north).second;
    influence[dic::n] = -2.0 / 3 * convect_sum_jacobi(cell, cell->north->north).second;
    influence[dic::s] = 2.0 / 3 * convect_sum_jacobi(cell, cell->south->south).second;
    influence[dic::ss] = -1.0 / 12 * convect_sum_jacobi(cell, cell->south->south->south->south).second;
    influence[dic::ee] = 1.0 / 12 * convect_sum_jacobi(cell, cell->east->east->east->east).first;
    influence[dic::e] = -2.0 / 3 * convect_sum_jacobi(cell, cell->east->east).first;
    influence[dic::w] = 2.0 / 3 * convect_sum_jacobi(cell, cell->west->west).first;
    influence[dic::ww] = -1.0 / 12 * convect_sum_jacobi(cell, cell->west->west->west->west).first;
    return influence;
}

std::array<Mat5, 13> face_convect_mat_3rd_upwind(cell_class* cell) {
    std::array<Mat5, 13> influence;
    influence.fill(Mat5::Zero());

    if (cell->north->vn() <= 0) {
        influence[dic::nn] += -1.0 / 6 * convect_sum_jacobi(cell, cell->north->north->north->north).second;
        influence[dic::n] += 5.0 / 6 * convect_sum_jacobi(cell, cell->north->north).second;
        influence[dic::c] += 1.0 / 3 * convect_sum_jacobi(cell, cell).second;
    } else {
        influence[dic::s] += -1.0 / 6 * convect_sum_jacobi(cell, cell->south->south).second;
        influence[dic::c] += 5.0 / 6 * convect_sum_jacobi(cell, cell).second;
        influence[dic::n] += 1.0 / 3 * convect_sum_jacobi(cell, cell->north->north).second;
    }

    if (cell->south->vn() >= 0) {
        influence[dic::ss] += 1.0 / 6 * convect_sum_jacobi(cell, cell->south->south->south->south).second;
        influence[dic::s] += -5.0 / 6 * convect_sum_jacobi(cell, cell->south->south).second;
        influence[dic::c] += -1.0 / 3 * convect_sum_jacobi(cell, cell).second;
    } else {
        influence[dic::n] += 1.0 / 6 * convect_sum_jacobi(cell, cell->north->north).second;
        influence[dic::c] += -5.0 / 6 * convect_sum_jacobi(cell, cell).second;
        influence[dic::s] += -1.0 / 3 * convect_sum_jacobi(cell, cell->south->south).second;
    }

    if (cell->east->vn() <= 0) {
        influence[dic::ee] += -1.0 / 6 * convect_sum_jacobi(cell, cell->east->east->east->east).first;
        influence[dic::e] += 5.0 / 6 * convect_sum_jacobi(cell, cell->east->east).first;
        influence[dic::c] += 1.0 / 3 * convect_sum_jacobi(cell, cell).first;
    } else {
        influence[dic::w] += -1.0 / 6 * convect_sum_jacobi(cell, cell->west->west).first;
        influence[dic::c] += 5.0 / 6 * convect_sum_jacobi(cell, cell).first;
        influence[dic::e] += 1.0 / 3 * convect_sum_jacobi(cell, cell->east->east).first;
    }

    if (cell->west->vn() >= 0) {
        influence[dic::ww] += 1.0 / 6 * convect_sum_jacobi(cell, cell->west->west->west->west).first;
        influence[dic::w] += -5.0 / 6 * convect_sum_jacobi(cell, cell->west->west).first;
        influence[dic::c] += -1.0 / 3 * convect_sum_jacobi(cell, cell).first;
    } else {
        influence[dic::e] += 1.0 / 6 * convect_sum_jacobi(cell, cell->east->east).first;
        influence[dic::c] += -5.0 / 6 * convect_sum_jacobi(cell, cell).first;
        influence[dic::w] += -1.0 / 3 * convect_sum_jacobi(cell, cell->west->west).first;
    }

    return influence;
}

std::array<Eigen::VectorXd, 13> viscous_convect_1st_upwind(cell_class* cell) {
    std::array<Eigen::VectorXd, 13> influence;
    for (auto& v : influence) v = Eigen::VectorXd::Zero(5);

    if (cell->north->vn() <= 0) {
        influence[dic::n] = viscous_convect_sum_jacobi(cell, cell->north->north).second;
    } else {
        influence[dic::c] = viscous_convect_sum_jacobi(cell, cell).second;
    }

    if (cell->south->vn() >= 0) {
        influence[dic::s] = viscous_convect_sum_jacobi(cell, cell->south->south).second;
    } else {
        influence[dic::c] = viscous_convect_sum_jacobi(cell, cell).second;
    }

    if (cell->east->vn() <= 0) {
        influence[dic::e] = viscous_convect_sum_jacobi(cell, cell->east->east).first;
    } else {
        influence[dic::c] = viscous_convect_sum_jacobi(cell, cell).first;
    }

    if (cell->west->vn() >= 0) {
        influence[dic::w] = viscous_convect_sum_jacobi(cell, cell->west->west).first;
    } else {
        influence[dic::c] = viscous_convect_sum_jacobi(cell, cell).first;
    }

    return influence;
}

void convect_hybrid(cell_class* cell) {
    auto upwind = face_convect_mat_3rd_upwind(cell);
    auto mid = face_convect_mat_4th_mid(cell);
    auto viscous = viscous_convect_1st_upwind(cell);
    for (int j = 0; j < 13; j++) {
        cell->form_influence(j, (alpha_H * upwind[j] + (1 - alpha_H) * mid[j]) / cell->vol);
        // np.vstack([np.zeros((4,5)), viscous[j]]) : 前4行全零, 第5行=viscous[j]
        Mat5 vstack = Mat5::Zero();
        vstack.row(4) = viscous[j].transpose();
        cell->form_influence(j, vstack / cell->vol);
    }
}

}  // namespace cc
