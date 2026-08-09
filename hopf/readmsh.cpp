#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>

// 小端读取工具
static uint32_t rd_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static int32_t rd_i32(const uint8_t* p) { return (int32_t)rd_u32(p); }
static double rd_f64_le(const uint8_t* p) {
    uint64_t u = (uint64_t)rd_u32(p) | ((uint64_t)rd_u32(p + 4) << 32);
    double d;
    std::memcpy(&d, &u, sizeof(d));
    return d;
}

// 内存搜索
static size_t find_bytes(const uint8_t* buf, size_t len, const char* pat) {
    size_t plen = std::strlen(pat);
    if (len < plen) return (size_t)-1;
    for (size_t i = 0; i + plen <= len; ++i)
        if (std::memcmp(buf + i, pat, plen) == 0) return i;
    return (size_t)-1;
}

// 解析连续十六进制数字
static long parse_hex_at(const uint8_t* buf, size_t len, size_t pos) {
    long val = 0;
    bool any = false;
    while (pos < len) {
        char c = (char)buf[pos];
        int d;
        if (c >= '0' && c <= '9')      d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else break;
        val = val * 16 + d;
        any = true;
        ++pos;
    }
    return any ? val : -1;
}

// 解析声明行中的十六进制数量
static bool parse_decl_count(const uint8_t* buf, size_t len,
                             const char* decl_pat, int& count) {
    size_t pos = find_bytes(buf, len, decl_pat);
    if (pos == (size_t)-1) return false;
    long v = parse_hex_at(buf, len, pos + std::strlen(decl_pat));
    if (v < 0) return false;
    count = (int)v;
    return true;
}

static long parse_dec_at(const uint8_t* buf, size_t len, size_t pos) {
    long val = 0;
    bool any = false;
    while (pos < len) {
        char c = (char)buf[pos];
        if (c < '0' || c > '9') break;
        val = val * 10 + (c - '0');
        any = true;
        ++pos;
    }
    return any ? val : -1;
}

// 解析声明行中的十进制值
static bool parse_decl_dec(const uint8_t* buf, size_t len,
                           const char* decl_pat, int& value) {
    size_t pos = find_bytes(buf, len, decl_pat);
    if (pos == (size_t)-1) return false;
    long v = parse_dec_at(buf, len, pos + std::strlen(decl_pat));
    if (v < 0) return false;
    value = (int)v;
    return true;
}

// 定位二进制数据块 '(' 的位置
static bool locate_paren(const uint8_t* buf, size_t len,
                         const char* decl_marker, size_t& paren) {
    size_t pos = find_bytes(buf, len, decl_marker);
    if (pos == (size_t)-1) return false;
    while (pos < len && buf[pos] != '\n') ++pos;
    while (pos < len && buf[pos] != '(') ++pos;
    if (pos >= len) return false;
    paren = pos;
    return true;
}

struct Face {              // 每面 4 个 int32 (n0 n1 c0 c1)
    int n0, n1, c0, c1;
};
struct FaceZone {
    long first, last;   // 面编号范围
    size_t paren;       // 数据块 '(' 位置
};

int main(int argc, char* argv[]) {
    const char* path = (argc > 1) ? argv[1] : "testdata/ICM.msh";

    // 读取整个文件
    FILE* fp = fopen(path, "rb");
    if (!fp) { std::cerr << "cannot open file: " << path << std::endl; return 1; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t)fsize);
    if (fread(buf.data(), 1, (size_t)fsize, fp) != (size_t)fsize) {
        std::cerr << "failed to read file" << std::endl;
        fclose(fp);
        return 1;
    }
    fclose(fp);

    // 头部声明
    int nd = 3;
    parse_decl_dec(buf.data(), buf.size(), "(2 ", nd);
    int num_nodes = 0;
    parse_decl_count(buf.data(), buf.size(), "(10 (0 1 ", num_nodes);
    if (num_nodes <= 0) {
        std::cerr << "cannot parse node declaration (not a Fluent msh file?)" << std::endl;
        return 1;
    }
    std::cout << "total nodes: " << num_nodes << std::endl;

    // 节点
    size_t n_paren = 0;
    if (!locate_paren(buf.data(), buf.size(), "(3010", n_paren)) {
        std::cerr << "cannot locate node coordinate block" << std::endl;
        return 1;
    }
    size_t n_data = (size_t)num_nodes * (size_t)nd * 8;

    std::vector<double> xs(num_nodes), ys(num_nodes), zs(num_nodes, 0.0);
    for (int i = 0; i < num_nodes; ++i) {
        const uint8_t* p = buf.data() + n_paren + 1 + (size_t)i * (size_t)nd * 8;
        xs[i] = rd_f64_le(p + 0);
        ys[i] = rd_f64_le(p + 8);
        if (nd >= 3) zs[i] = rd_f64_le(p + 16);
    }

    // 面块
    std::vector<FaceZone> fzones;
    size_t search = 0;
    while (true) {
        size_t m = find_bytes(buf.data() + search, buf.size() - search, "(3013");
        if (m == (size_t)-1) break;
        m += search;
        size_t p = m + 5;   // 跳过 "(3013"
        while (p < buf.size() && buf[p] != '(') ++p;
        ++p;
        while (p < buf.size() && (buf[p] == ' ' || buf[p] == '\t')) ++p;
        while (p < buf.size() && buf[p] != ' ') ++p;
        while (p < buf.size() && buf[p] == ' ') ++p;
        long first = parse_hex_at(buf.data(), buf.size(), p);
        while (p < buf.size() && buf[p] != ' ') ++p;
        while (p < buf.size() && buf[p] == ' ') ++p;
        long last = parse_hex_at(buf.data(), buf.size(), p);
        size_t paren = m;
        while (paren < buf.size() && buf[paren] != '\n') ++paren;
        while (paren < buf.size() && buf[paren] != '(') ++paren;
        if (first >= 0 && last >= 0 && paren < buf.size())
            fzones.push_back({ first, last, paren });
        search = m + 5;
    }

    if (fzones.empty()) {
        std::cerr << "no face data blocks found" << std::endl;
        return 1;
    }
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
            f.n0 = rd_i32(p + 0);
            f.n1 = rd_i32(p + 4);
            f.c0 = rd_i32(p + 8);
            f.c1 = rd_i32(p + 12);
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
        size_t n_end = n_paren + 1 + n_data;
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
