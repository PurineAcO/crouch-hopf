// boundary.h —— 翼面/远场边界条件
#pragma once

#include "classconfig.h"

namespace cc {

// 处理壁面处的边界条件, 要求提供的网格必须是壁面处(n=1)的
void wing_boundary(cell_class* cell);

// 处理远场边界条件(n=N_MAX), 依据入流/出流特征边界
void far_boundary(cell_class* cell);

}  // namespace cc
