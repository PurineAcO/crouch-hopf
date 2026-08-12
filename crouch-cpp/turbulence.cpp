// turbulence.cpp —— SA 湍流模型实现
#include "turbulence.h"

#include "grad.h"

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

namespace cc {

namespace {

// 槽位名称 -> influence 编号
int dic_name(const std::string& s) {
    if (s == "c") return dic::c;
    if (s == "n") return dic::n;
    if (s == "s") return dic::s;
    if (s == "e") return dic::e;
    if (s == "w") return dic::w;
    if (s == "ne") return dic::ne;
    if (s == "nw") return dic::nw;
    if (s == "se") return dic::se;
    if (s == "sw") return dic::sw;
    if (s == "nn") return dic::nn;
    if (s == "ss") return dic::ss;
    if (s == "ee") return dic::ee;
    if (s == "ww") return dic::ww;
    throw std::runtime_error("unknown dic name: " + s);
}

// 含 B/C/G 中心项形式的扩散矩阵 (D6/D7)
// g0/g1: Dx 第5行用 miublgrad[g0], Dy 第5行用 miublgrad[g1]
Mat5 matD_center(const face_class* face, const Eigen::Vector2d& dire, int g0, int g1, bool take_first) {
    const double& mue = face->mu_eff;
    const double& lae = face->lambda_eff;
    const double d0 = dire[0], d1 = dire[1];
    const double B1 = face->miubl * 2 * (face->ugrad[0] - 1.0 / 3 * (face->ugrad[0] + face->vgrad[1])) *
                      (face->fv1 * (4 - 3 * face->fv1));
    const double B2 = face->rho * 2 * (face->ugrad[0] - 1.0 / 3 * (face->ugrad[0] + face->vgrad[1])) *
                      (face->fv1 * (4 - 3 * face->fv1));
    const double C1 = face->miubl * 2 * (face->ugrad[1] + face->vgrad[0]) * (face->fv1 * (4 - 3 * face->fv1));
    const double C2 = face->rho * 2 * (face->ugrad[1] + face->vgrad[0]) * (face->fv1 * (4 - 3 * face->fv1));
    const double E1 = face->miubl * 2 * (face->vgrad[1] - 1.0 / 3 * (face->ugrad[0] + face->vgrad[1])) *
                      (face->fv1 * (4 - 3 * face->fv1));
    const double E2 = face->rho * 2 * (face->vgrad[1] - 1.0 / 3 * (face->ugrad[0] + face->vgrad[1])) *
                      (face->fv1 * (4 - 3 * face->fv1));
    const double G1 = face->miubl * (face->Tgrad[0] * (face->fv1 * (4 - 3 * face->fv1)) / Prt);
    const double G2 = face->rho * (face->Tgrad[0] * (face->fv1 * (4 - 3 * face->fv1)) / Prt);

    Mat5 Dx, Dy;
    Dx << 0, 0, 0, 0, 0,
          B1 / 2, 4.0 / 3 * mue * d0, -2.0 / 3 * mue * d1, 0, B2 / 2,
          C1 / 2, mue * d1, mue * d0, 0, C2 / 2,
          face->u * B1 / 2 + face->v * C1 / 2 + G1 / 2,
              face->tauxx / 2 + 4.0 / 3 * face->u * mue * d0 + face->v * mue * d1,
              face->tauxy / 2 - 2.0 / 3 * face->u * mue * d1 + face->v * mue * d0,
              lae * d0,
              face->u * B2 / 2 + face->v * C2 / 2 + G2 / 2,
          0.5 * face->miublgrad[g0] * face->miubl * inv_sigma, 0, 0, 0,
              0.5 * face->miublgrad[g0] * face->rho * inv_sigma + mue * dire[g0] * inv_sigma;
    Dy << 0, 0, 0, 0, 0,
          C1 / 2, mue * d1, mue * d0, 0, C2 / 2,
          E1 / 2, -2.0 / 3 * mue * d0, 4.0 / 3 * mue * d1, 0, E2 / 2,
          face->u * C1 / 2 + face->v * E1 / 2 + G1 / 2,
              face->tauxy / 2 + face->u * mue * d1 - 2.0 / 3 * face->v * mue * d0,
              face->tauyy / 2 + face->u * mue * d1 + 4.0 / 3 * face->v * mue * d0,
              lae * d1,
              face->u * C2 / 2 + face->v * E2 / 2 + G2 / 2,
          0.5 * face->miublgrad[g1] * face->miubl * inv_sigma, 0, 0, 0,
              0.5 * face->miublgrad[g1] * face->rho * inv_sigma + mue * dire[g1] * inv_sigma;
    auto pr = face->jacobi(Dx, Dy);
    return take_first ? pr.first : pr.second;
}

// 纯粘性形式的扩散矩阵 (D2..D5, D8, D10, D11)
Mat5 matD_viscous(const face_class* face, const Eigen::Vector2d& dire, bool take_first) {
    const double& mue = face->mu_eff;
    const double& lae = face->lambda_eff;
    const double d0 = dire[0], d1 = dire[1];
    Mat5 Dx, Dy;
    Dx << 0, 0, 0, 0, 0,
          0, 4.0 / 3 * mue * d0, -2.0 / 3 * mue * d1, 0, 0,
          0, mue * d1, mue * d0, 0, 0,
          0, 4.0 / 3 * face->u * mue * d0 + face->v * mue * d1,
              -2.0 / 3 * face->u * mue * d1 + face->v * mue * d0, lae * d0, 0,
          0, 0, 0, 0, mue * d0 * inv_sigma;
    Dy << 0, 0, 0, 0, 0,
          0, mue * d1, mue * d0, 0, 0,
          0, -2.0 / 3 * mue * d0, 4.0 / 3 * mue * d1, 0, 0,
          0, face->u * mue * d1 - 2.0 / 3 * face->v * mue * d0,
              face->u * mue * d1 + 4.0 / 3 * face->v * mue * d0, lae * d1, 0,
          0, 0, 0, 0, mue * d1 * inv_sigma;
    auto pr = face->jacobi(Dx, Dy);
    return take_first ? pr.first : pr.second;
}

}  // namespace

void SA_calc_constants(cell_class* cell) {
    // 有效粘度 μeff (Sutherland 公式 + 阻尼函数)
    cell->mu = mu0 * std::pow(cell->T / T0, 1.5) * (T0 + Ts) / (cell->T + Ts);
    cell->chi = cell->rho * cell->miubl / cell->mu;
    cell->fv1 = (cell->chi * cell->chi * cell->chi) / (cell->chi * cell->chi * cell->chi + Cv1 * Cv1 * Cv1);
    cell->mu_eff = cell->mu + cell->rho * cell->fv1 * cell->miubl;
    // 有效导热系数 λeff
    cell->lambda_eff = cell->mu / Pr + (cell->rho * cell->miubl * cell->fv1) / Prt;
    // 源项部分参数
    cell->ft2 = Ct3 * std::exp(-Ct4 * (cell->chi * cell->chi));
    cell->fv2 = 1 - (cell->chi) / (1 + cell->chi * cell->fv1);
    cell->Omega = std::abs(cell->ugrad[1] - cell->vgrad[0]);
    cell->Sbl = cell->Omega + cell->fv2 * cell->miubl / ((kappa * cell->sad) * (kappa * cell->sad));
    cell->r = std::min(cell->miubl / (cell->Sbl * kappa * kappa * cell->sad * cell->sad), rmax);
    cell->g = cell->r + Cw2 * (std::pow(cell->r, 6) - cell->r);
    cell->fw = cell->g * std::pow((1 + Cw3 * Cw3 * Cw3 * Cw3 * Cw3 * Cw3) /
                                      (std::pow(cell->g, 6) + Cw3 * Cw3 * Cw3 * Cw3 * Cw3 * Cw3),
                                  1.0 / 6);
    cell->S = std::sqrt(2 * cell->ugrad[0] * cell->ugrad[0] + 2 * cell->vgrad[1] * cell->vgrad[1] +
                        (cell->ugrad[1] + cell->vgrad[0]) * (cell->ugrad[1] + cell->vgrad[0]));
}

void diffusion_2nd_mid_SA(face_class* face) {
    face->mu_eff = (face->me->mu_eff + face->nei->mu_eff) / 2;
    face->lambda_eff = (face->me->lambda_eff + face->nei->lambda_eff) / 2;
    face->chi = (face->me->chi + face->nei->chi) / 2;
    face->fv1 = (face->me->fv1 + face->nei->fv1) / 2;
    face->mu = (face->me->mu + face->nei->mu) / 2;
    face->tauxx = face->mu_eff * (face->ugrad[0] - 1.0 / 3 * (face->ugrad[0] + face->vgrad[1]));
    face->tauxy = face->mu_eff * (face->ugrad[1] + face->vgrad[0]);
    face->tauyy = face->mu_eff * (face->vgrad[1] - 1.0 / 3 * (face->ugrad[0] + face->vgrad[1]));
}

void cell_diffusion(cell_class* cell) {
    std::array<Mat5, 13> influence;
    influence.fill(Mat5::Zero());

    // 东面
    {
        const char* dirs[] = {"c", "e", "n", "ne", "s", "se", "ee", "w"};
        auto res = face_diffusion(cell->east);
        for (int i = 0; i < 8; i++) influence[dic_name(dirs[i])] += res[i];
    }
    // 西面
    {
        const char* dirs[] = {"w", "c", "nw", "n", "sw", "s", "e", "ww"};
        auto res = face_diffusion(cell->west);
        for (int i = 0; i < 8; i++) influence[dic_name(dirs[i])] += res[i];
    }
    // 南面
    {
        const char* dirs[] = {"s", "c", "se", "sw", "ss", "w", "n", "e"};
        auto res = face_diffusion(cell->south);
        for (int i = 0; i < 8; i++) influence[dic_name(dirs[i])] += res[i];
    }
    // 北面
    {
        const char* dirs[] = {"c", "n", "e", "w", "s", "nw", "nn", "ne"};
        auto res = face_diffusion(cell->north);
        for (int i = 0; i < 8; i++) influence[dic_name(dirs[i])] += res[i];
    }

    for (int i = 0; i < 13; i++) cell->form_influence(i, -influence[i] / cell->vol);
}

std::array<Mat5, 8> face_diffusion(face_class* face) {
    bool take_first = (face->direction == "WE");   // jacobiflag=0 -> first
    auto dic_ = green_gauss_face_vari(face);

    Mat5 D6, D7, D2, D3, D10, D11, D8, D5;
    if (face->direction == "WE") {
        D6 = matD_center(face, dic_.at("w"), 0, 0, take_first);
        D7 = matD_center(face, dic_.at("e"), 0, 1, take_first);
        D2 = matD_viscous(face, dic_.at("nw"), take_first);
        D3 = matD_viscous(face, dic_.at("ne"), take_first);
        D10 = matD_viscous(face, dic_.at("sw"), take_first);
        D11 = matD_viscous(face, dic_.at("se"), take_first);
        D8 = matD_viscous(face, dic_.at("ee"), take_first);
        D5 = matD_viscous(face, dic_.at("ww"), take_first);
    } else {  // NS
        D6 = matD_center(face, dic_.at("s"), 0, 0, take_first);
        D7 = matD_center(face, dic_.at("n"), 0, 1, take_first);
        D2 = matD_viscous(face, dic_.at("se"), take_first);
        D3 = matD_viscous(face, dic_.at("sw"), take_first);
        D10 = matD_viscous(face, dic_.at("ss"), take_first);
        D11 = matD_viscous(face, dic_.at("nw"), take_first);
        D8 = matD_viscous(face, dic_.at("nn"), take_first);
        D5 = matD_viscous(face, dic_.at("ne"), take_first);
    }
    return {D6, D7, D2, D3, D10, D11, D8, D5};
}

void cell_source(cell_class* cell) {
    double dfv2 = (3 * cell->chi * cell->fv1 * (1 - cell->fv1) - 1) /
                  ((1 + cell->chi * cell->fv1) * (1 + cell->chi * cell->fv1));
    double dft2 = -2 * cell->chi * Ct4 * cell->ft2;
    double dfw = cell->fw / cell->g * (Cw3 * Cw3 * Cw3 * Cw3 * Cw3 * Cw3) /
                     (std::pow(cell->g, 6) + Cw3 * Cw3 * Cw3 * Cw3 * Cw3 * Cw3) *
                     (1 - Cw2 + 6 * Cw2 * std::pow(cell->r, 5));

    double R1 = Cb1 * (1 - cell->ft2 - cell->chi * dft2);
    double R2 = Cb1 * cell->r * (cell->chi * dfv2 * (1 - cell->ft2) + cell->ft2 + cell->chi * dft2);
    double R3 = -Cw1 * kappa * kappa * cell->r *
                (cell->fw - dfw * dfv2 * cell->r * cell->r * cell->chi);
    double R4 = Cb2 * inv_sigma * (cell->miublgrad[0] * cell->miublgrad[0] +
                                   cell->miublgrad[1] * cell->miublgrad[1]);
    double R5 = -C5 * cell->miubl * cell->miubl * cell->S * cell->S / (cc::gamma * cc::R * cell->T);
    double Rsrc = (R1 + R2 + R3) * cell->Sbl * cell->miubl + R4 + R5;

    double M1 = Cb1 * (1 - cell->ft2 - cell->chi * dft2);
    double M2 = Cb1 * cell->r *
                ((cell->fv2 + cell->chi * dfv2) * (1 - cell->ft2) + 2 * cell->ft2 + cell->chi * dft2);
    double M3 = -Cw1 * kappa * kappa * cell->r *
                (2 * cell->fw + cell->r * dfw - cell->r * cell->r * dfw * (cell->fv2 + cell->chi * dfv2));
    double M4 = -2 * C5 * cell->rho * cell->miubl * cell->S * cell->S / (cc::gamma * cc::R * cell->T);
    double M0 = (M1 + M2 + M3) * cell->Sbl * cell->rho + M4;

    double T_ = C5 * cell->rho * cell->miubl * cell->miubl * cell->S * cell->S / (cc::gamma * cc::R * cell->T * cell->T);

    double O1 = cell->rho * cell->miubl / cell->Omega *
                (Cb1 * (1 - cell->ft2) + Cw1 * kappa * kappa * cell->r * cell->r * dfw) *
                (cell->ugrad[1] - cell->vgrad[0]);
    double O2 = 2 * Cb2 * inv_sigma * cell->rho;
    double O3 = -2 * C5 * cell->rho * cell->miubl * cell->miubl / (cc::gamma * cc::R * cell->T);
    double O4 = O3 * (cell->ugrad[1] + cell->vgrad[0]);

    auto dic_ = green_gauss_cell_vari(cell);
    const char* dirs[] = {"c", "n", "s", "e", "w"};
    for (const char* dire : dirs) {
        const auto& d = dic_.at(dire);
        double U = O1 * d[1] + O4 * d[1] + O3 * 2 * cell->ugrad[0] * d[0];
        double V = -O1 * d[0] + O4 * d[0] + O3 * 2 * cell->vgrad[1] * d[1];
        double M = O2 * (cell->miublgrad[0] * d[0] + cell->miublgrad[1] * d[1]);

        // np 广播: 5x5 矩阵每行加上 -(向量)^T
        Mat5 add = Mat5::Zero();
        if (std::string(dire) == "c") {
            Eigen::RowVectorXd v(5);
            v << Rsrc, U, V, T_, M + M0;
            add.rowwise() -= v;
            cell->form_influence(dic::c, add);
        } else {
            Eigen::RowVectorXd v(5);
            v << 0, U, V, 0, M;
            add.rowwise() -= v;
            cell->form_influence(dic_name(dire), add);
        }
    }
}

}  // namespace cc
