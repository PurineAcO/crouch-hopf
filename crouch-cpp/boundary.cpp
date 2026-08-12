// boundary.cpp —— 翼面/远场边界条件
#include "boundary.h"

#include <cmath>
#include <stdexcept>

namespace cc {

void wing_boundary(cell_class* cell) {
    if (cell->index.second != 1) throw std::runtime_error("Not wing-up cell");
    double dx1 = cell->north->north->x - cell->x;
    double dy1 = cell->north->north->y - cell->y;
    double dx2 = cell->north->north->north->north->x - cell->north->north->x;
    double dy2 = cell->north->north->north->north->y - cell->north->north->y;
    double det = dx1 * dy2 - dx2 * dy1;
    double C0 = ((dy1 - dy2) * cell->south->nx() + (dx2 - dx1) * cell->south->ny()) / det;
    double C1 = (dy2 * cell->south->nx() - dx2 * cell->south->ny()) / det;
    double C2 = -1 * (dy1 * cell->south->nx() - dx1 * cell->south->ny()) / det;
    Mat5 A = Mat5::Zero(), B = Mat5::Zero(), D = Mat5::Zero();
    A << C0, 0, 0, 0, 0,
         0, 1, 0, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 0, C0, 0,
         0, 0, 0, 0, 1;
    B << C1, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 0, C1, 0,
         0, 0, 0, 0, 0;
    D << C2, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 0, C2, 0,
         0, 0, 0, 0, 0;
    cell->form_influence(dic::c, A);
    cell->form_influence(dic::n, B);
    cell->form_influence(dic::nn, D);
}

void far_boundary(cell_class* cell) {
    if (cell->index.second != N_MAX) throw std::runtime_error("Not Far-in cell");
    double CT = std::sqrt(gamma * R) / ((gamma - 1) * cell->T);
    double KR = -1 * (R * (gamma - 1) * cell->T) / std::pow(cell->rho, gamma);
    double KT = R / std::pow(cell->rho, gamma - 1);
    double dx1 = cell->south->south->x - cell->x;
    double dy1 = cell->south->south->y - cell->y;
    double dx2 = cell->south->south->south->south->x - cell->south->south->x;
    double dy2 = cell->south->south->south->south->y - cell->south->south->y;
    double det = dx1 * dy2 - dx2 * dy1;
    double C0 = ((dy1 - dy2) * cell->north->nx() + (dx2 - dx1) * cell->north->ny()) / det;
    double C1 = (dy2 * cell->north->nx() - dx2 * cell->north->ny()) / det;
    double C2 = -1 * (dy1 * cell->north->nx() - dx1 * cell->north->ny()) / det;
    double nrm = std::sqrt(cell->north->nx() * cell->north->nx() + cell->north->ny() * cell->north->ny());
    double kx = cell->north->nx() / nrm;
    double ky = cell->north->ny() / nrm;

    Mat5 A = Mat5::Zero(), B = Mat5::Zero(), D = Mat5::Zero();
    if (cell->north->vn() <= 0) {  // 入流边界
        A << 0, kx, ky, CT, 0,
             0, C0 * kx, C0 * ky, -C0 * CT, 0,
             0, ky, -kx, 0, 0,
             KR, 0, 0, KT, 0,
             0, 0, 0, 0, 1;
        B << 0, 0, 0, 0, 0,
             0, C1 * kx, C1 * ky, -C1 * CT, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0;
        D << 0, 0, 0, 0, 0,
             0, C2 * kx, C2 * ky, -C2 * CT, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0;
    } else {  // 出流边界
        A << 0, C0 * kx, C0 * ky, C0 * CT, 0,
             0, kx, ky, -CT, 0,
             0, C0 * ky, -C0 * kx, 0, 0,
             C0 * KR, 0, 0, C0 * KT, 0,
             0, 0, 0, 0, C0;
        B << 0, C1 * kx, C1 * ky, C1 * CT, 0,
             0, 0, 0, 0, 0,
             0, C1 * ky, -C1 * kx, 0, 0,
             C1 * KR, 0, 0, C1 * KT, 0,
             0, 0, 0, 0, C1;
        D << 0, C2 * kx, C2 * ky, C2 * CT, 0,
             0, 0, 0, 0, 0,
             0, C2 * ky, -C2 * kx, 0, 0,
             C2 * KR, 0, 0, C2 * KT, 0,
             0, 0, 0, 0, C2;
    }
    cell->form_influence(dic::c, A);
    cell->form_influence(dic::s, B);
    cell->form_influence(dic::ss, D);
}

}  // namespace cc
