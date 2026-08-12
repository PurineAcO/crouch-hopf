#!/bin/bash
# run.sh —— 编译并运行 C++ 全局稳定性求解器
# 用法: ./run.sh [ransdata] [edge]   默认读取 ransdata.txt / edge.txt
set -e
cd "$(dirname "$0")"

RANS="${1:-ransdata.txt}"
EDGE="${2:-edge.txt}"

mkdir -p build
clang++ -O2 -std=c++17 -Ithird_party/eigen -Ithird_party/spectra/include \
    solvemain.cpp classconfig.cpp readrans.cpp boundary.cpp grad.cpp convect.cpp \
    turbulence.cpp formmat.cpp eigmain.cpp -o build/solvemain

echo "== 运行求解器: ransdata=$RANS edge=$EDGE =="
./build/solvemain "$RANS" "$EDGE"
