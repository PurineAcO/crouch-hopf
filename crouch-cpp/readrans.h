// readrans.h —— 读取 ransdata.txt / edge.txt, 构建 CellList 与 FaceList
#pragma once

#include "config.h"

#include <string>
#include <utility>

namespace cc {

// 自动获知网格规模, 返回 (S_MAX, N_MAX)
std::pair<int, int> get_scale(const std::string& ransdata);

// 读取 ransdata.txt, 物理单元填入 CellList[s+HALO][n+HALO], 随后填充全部虚单元
void read_cells(const std::string& ransdata, int S_MAX_, int N_MAX_, int h = HALO);

// 填充 halo 虚单元
void fill_ghost(int S_MAX_, int N_MAX_, int h = HALO);

// 检测 NS 面的环方向, 返回 1/-1 表示逆时针/顺时针排列
int detect_orient(const std::string& edgedata);

// 读取 edge.txt, 构建全部面与单元 Jacobi
void form_edge(const std::string& edgedata, int h = HALO);

// 总入口: 读取 ransdata.txt 和 edge.txt
void read_rans(const std::string& ranspath, const std::string& edgepath);

}  // namespace cc
