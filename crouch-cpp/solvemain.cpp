// solvemain.cpp —— 主程序: 读数据→边界→梯度→对流→扩散/源项→组装→特征值
// 用法: solvemain [ransdata] [edge]  (默认 ransdata.txt, edge.txt)
#include "classconfig.h"
#include "readrans.h"
#include "boundary.h"
#include "grad.h"
#include "convect.h"
#include "turbulence.h"
#include "formmat.h"
#include "eigmain.h"

#include <chrono>
#include <iostream>
#include <string>

using namespace cc;

int main(int argc, char* argv[]) {
    std::string ranspath = argc > 1 ? argv[1] : "ransdata.txt";
    std::string edgepath = argc > 2 ? argv[2] : "edge.txt";
    auto starttime = std::chrono::steady_clock::now();

    // 读网格/流场
    read_rans(ranspath, edgepath);
    std::cout << "RANS data read done. S_MAX=" << S_MAX << " N_MAX=" << N_MAX << "\n";

    // 边界条件(翼面 n=1, 远场 n=N_MAX)
    for (int s = 1; s <= S_MAX; s++) {
        far_boundary(goto_HALOcell(s, N_MAX));
        wing_boundary(goto_HALOcell(s, 1));
    }

    // Green-Gauss 梯度(单元+面)
    for (int n = 1; n <= N_MAX; n++)
        for (int s = 1; s <= S_MAX; s++) {
            cell_class* c = goto_HALOcell(s, n);
            green_gauss_from_JST(c, c->north, c->south, c->east, c->west);
        }
    for (face_class* f : FaceList_NS) f->grad_2nd_mid();
    for (face_class* f : FaceList_WE) f->grad_2nd_mid();

    // 对流项(跳过物面/远场)
    for (int n = 2; n < N_MAX; n++)
        for (int s = 1; s <= S_MAX; s++) convect_hybrid(goto_HALOcell(s, n));

    // SA 湍流参数(物理+虚单元), 面上二阶插值
    for (int n = 1; n <= N_MAX; n++)
        for (int s = 1; s <= S_MAX; s++) SA_calc_constants(goto_HALOcell(s, n));
    for (int s = 1; s <= S_MAX; s++) {
        for (int n : {0, -1}) SA_calc_constants(goto_HALOcell(s, n));
        for (int n : {N_MAX + 1, N_MAX + 2}) SA_calc_constants(goto_HALOcell(s, n));
    }
    for (face_class* f : FaceList_NS) diffusion_2nd_mid_SA(f);
    for (face_class* f : FaceList_WE) diffusion_2nd_mid_SA(f);

    // 扩散项 + 源项(跳过物面/远场)
    for (int n = 2; n < N_MAX; n++)
        for (int s = 1; s <= S_MAX; s++) {
            cell_class* c = goto_HALOcell(s, n);
            cell_diffusion(c);
            cell_source(c);
        }
    std::cout << "All the Flux is OK.\n";

    // 组装稀疏矩阵 S/T
    for (int n = 1; n <= N_MAX; n++)
        for (int s = 1; s <= S_MAX; s++) formmat(goto_HALOcell(s, n));
    auto [S, T] = build();
    std::cout << "S: " << S.rows() << "x" << S.cols() << " nnz = " << S.nonZeros() << "\n";
    std::cout << "T: " << T.rows() << "x" << T.cols() << " nnz = " << T.nonZeros() << "\n";

    // 特征值分析
    solve_eig(S, T);

    auto endtime = std::chrono::steady_clock::now();
    std::cout << "Total time: "
              << std::chrono::duration<double>(endtime - starttime).count() << " s\n";
    return 0;
}
