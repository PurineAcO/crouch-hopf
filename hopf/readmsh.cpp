#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <vector>
#include <iostream>
#include "translate.h"

struct Face {               // 每面 4 个 int32 (n0 n1 c0 c1)
    int n0, n1, c0, c1;
};
struct FaceZone {
    long first, last;       // 面编号范围
    size_t paren;           // 数据块 '(' 位置
};

int main(int argc, char* argv[]) {
    const char* path = (argc > 1) ? argv[1] : "testdata/ICM.msh"; // 如果不使用命令行请更改这一段.

    FILE* fp = fopen(path, "rb");
    if (!fp) { std::cerr << "cannot open file: " << path << std::endl; return 1; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t)fsize);
    if (fread(buf.data(), 1, (size_t)fsize, fp) != (size_t)fsize) {
        std::cerr << "failed to read file" << std::endl;
        fclose(fp);return 1;    // 如果没有读取到正确数量的char就宣告失败.
    }
    fclose(fp);

    int nd = 3;    // 表示网格文件的维度.默认是3.如果在"(2 "后出现的是2,则说明是2d网格.此时更换这个值
    parse_decl_dec(buf.data(), buf.size(), "(2 ", nd);
    int num_nodes = 0;
    parse_decl_count(buf.data(), buf.size(), "(10 (0 1 ", num_nodes); // TODO:这一段也不一定是"(10 0 1 "
    if (num_nodes <= 0) {
        std::cerr << "cannot parse node declaration (not a Fluent msh file?)" << std::endl;return 1;
    }
    std::cout << "total nodes: " << num_nodes << std::endl;

    // 节点
    size_t n_paren = 0;
    if (!locate_paren(buf.data(), buf.size(), "(3010", n_paren)) {
        std::cerr << "cannot locate node coordinate block" << std::endl;
        return 1;
    }
    std::vector<double> xs, ys, zs;
    size_t n_end = 0;
    if (!read_node_coords(buf, n_paren, num_nodes, nd, xs, ys, zs, n_end)) {
        std::cerr << "cannot read node coordinates" << std::endl;
        return 1;
    }

    std::vector<FaceZone> fzones;size_t search = 0;
    while (true) {
        size_t m = find_bytes(buf.data() + search, buf.size() - search, "(3013");
        if (m == (size_t)-1) break; // 找不到更多 (3013
        m += search; 
        long fz_first,fz_last;
        size_t paren = form_facezone(buf, m, fz_first, fz_last);
        if (fz_first >= 0 && fz_last >= 0 && paren < buf.size())
            fzones.push_back({ fz_first, fz_last, paren }); // 把偏移量写进去,此时是记不住面的名称的,只能记住编号.
        search = m + 5;
    }

    if (fzones.empty()) {std::cerr << "no face data blocks found" << std::endl;return 1;}

    long num_faces = fzones.back().last;
    std::cout << "total faces: " << num_faces << " (" << fzones.size() << " zones)" << std::endl;
    
    for (size_t k = 0; k < fzones.size(); ++k) {
        std::cout << "  face zone #" << k + 1 << ": faces " << fzones[k].first
                  << "~" << fzones[k].last
                  << " (" << (fzones[k].last - fzones[k].first + 1) << ")" << std::endl;
    }

    std::vector<std::vector<Face>> faces(fzones.size());
    for (size_t k = 0; k < fzones.size(); ++k) {
        long nf = fzones[k].last - fzones[k].first + 1;
        faces[k].reserve((size_t)nf);
        for (long i = 0; i < nf; ++i) {
            const uint8_t* p = buf.data() + fzones[k].paren + 1 + (size_t)i * 16;
            Face f;
            f.n0 = read_i32(p + 0);
            f.n1 = read_i32(p + 4);
            f.c0 = read_i32(p + 8);
            f.c1 = read_i32(p + 12);
            faces[k].push_back(f);
        }
    }

    // 翻译
    if (argc > 2) {
        FILE* out = fopen(argv[2], "w");
        if (!out) {
            std::cerr << "cannot create output file: " << argv[2] << std::endl;
            return 1;
        }

        // 1) 复制头部文本
        fwrite(buf.data(), 1, n_paren, out);

        // 2) 节点坐标
        fprintf(out, "(\n");
        for (int i = 0; i < num_nodes; ++i) {
            if (nd >= 3)
                fprintf(out, "  %d %.10g %.10g %.10g\n", i + 1, xs[i], ys[i], zs[i]);
            else
                fprintf(out, "  %d %.10g %.10g\n", i + 1, xs[i], ys[i]);
        }
        fprintf(out, ")\n");

        // 3) 坐标块后的文本
        fwrite(buf.data() + n_end, 1, fzones[0].paren - n_end, out);

        // 4) 各面块
        for (size_t k = 0; k < fzones.size(); ++k) {
            fprintf(out, "(\n");
            for (const Face& f : faces[k])
                fprintf(out, "  %d %d %d %d\n", f.n0, f.n1, f.c0, f.c1);
            fprintf(out, ")\n");

            size_t f_end = fzones[k].paren + 1 + (size_t)(fzones[k].last - fzones[k].first + 1) * 16;
            size_t next = (k + 1 < fzones.size()) ? fzones[k + 1].paren : buf.size();
            fwrite(buf.data() + f_end, 1, next - f_end, out);
        }

        fclose(out);
        std::cout << "translated msh written to: " << argv[2] << std::endl;
    }

    return 0;
}
