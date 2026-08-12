// config.h —— 全局常量与类型
// 常量来源: 项目根 config.json(与原 crouch/classconfig.py 一致)
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace cc {

// —— S-A 湍流模型常数 ——
inline constexpr double inv_sigma = 1.5;    // 1/σ, 湍流扩散系数
inline constexpr double Cv1 = 7.1;          // 阻尼函数常数
inline constexpr double Ct3 = 1.2;          // 转捩修正
inline constexpr double Ct4 = 0.5;          // 转捩修正
inline constexpr double fv3 = 1.0;          // 涡量模修正
inline constexpr double kappa = 0.41;       // von Kármán 常数
inline constexpr double Cb1 = 0.1355;       // 生成项
inline constexpr double Cb2 = 0.622;        // 扩散项
inline constexpr double Cw1 = 3.2391;       // 破坏项
inline constexpr double Cw2 = 0.3;          // 壁面阻尼
inline constexpr double Cw3 = 2.0;          // 壁面阻尼
inline constexpr double rmax = 10.0;        // 无量纲距离 r 上限
inline constexpr double C5 = 3.5;           // 可压缩修正

// —— 物性常数(Sutherland 等) ——
inline constexpr double R = 287.06;         // 气体常数
inline constexpr double cp = 1004.71;       // 定压比热
inline constexpr double gamma = 1.4;        // 比热比
inline constexpr double mu0 = 1.716e-5;     // Sutherland 参考粘度
inline constexpr double T0 = 273.15;        // Sutherland 参考温度
inline constexpr double Ts = 110.4;         // Sutherland 温度
inline constexpr double Pr = 0.72;          // 层流 Prandtl 数
inline constexpr double Prt = 0.9;          // 湍流 Prandtl 数

// —— 求解器参数 ——
inline int S_MAX = 0;                       // 每层单元数(读数据确定)
inline int N_MAX = 0;                       // 层数(读数据确定)
inline constexpr double alpha_H = 0.2;      // 对流混合格式系数
inline constexpr int HALO = 2;              // 虚单元层数

// 5×5 稠密矩阵(变量序: ρ,u,v,T,ν̃)
using Mat5 = Eigen::Matrix<double, 5, 5>;
using Vec5 = Eigen::Matrix<double, 5, 1>;

// influence 的 13 个槽位编号(与原 Python classconfig.dic 一致)
namespace dic {
inline constexpr int nn = 0, nw = 1, n = 2, ne = 3, ww = 4, w = 5, c = 6,
                     e = 7, ee = 8, sw = 9, s = 10, se = 11, ss = 12;
}  // namespace dic

}  // namespace cc
