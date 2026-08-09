#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

uint32_t read_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int32_t read_i32(const uint8_t* p){
    return (int32_t)read_u32(p);
}

double read_f64(const uint8_t* p){
    uint64_t u = (uint64_t) read_u32(p) | ((uint64_t) read_u32(p+4) << 32);
    double d; std::memcpy(&d,&u,sizeof(d));return d;
}

size_t find_bytes(const uint8_t* buf, size_t len, const char* pat) {
    size_t plen = std::strlen(pat);
    if (len < plen) return (size_t)-1;   // 如果待查找区域比pat小,一定查找不到
    for (size_t i = 0; i + plen <= len; ++i)
        if (std::memcmp(buf + i, pat, plen) == 0) return i; 
    return (size_t)-1; 
}

long parse_hex_at(const uint8_t* buf, size_t len, size_t pos) {
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

bool parse_decl_count(const uint8_t* buf, size_t len, const char* decl_pat, int& count) {
    size_t pos = find_bytes(buf, len, decl_pat);
    if (pos == (size_t)-1) return false;
    long v = parse_hex_at(buf, len, pos + std::strlen(decl_pat));
    if (v < 0) return false;
    count = (int)v;
    return true;
}

long parse_dec_at(const uint8_t* buf, size_t len, size_t pos) {
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

bool parse_decl_dec(const uint8_t* buf, size_t len, const char* decl_pat, int& value) {
    size_t pos = find_bytes(buf, len, decl_pat);
    if (pos == (size_t)-1) return false;
    long v = parse_dec_at(buf, len, pos + std::strlen(decl_pat));
    if (v < 0) return false;
    value = (int)v;
    return true;
}

bool locate_paren(const uint8_t* buf, size_t len, const char* decl_marker, size_t& paren) {
    size_t pos = find_bytes(buf, len, decl_marker);
    if (pos == (size_t)-1) return false;
    while (pos < len && buf[pos] != '\n') ++pos;
    while (pos < len && buf[pos] != '(') ++pos;
    if (pos >= len) return false;
    paren = pos;
    return true;
}

bool read_node_coords(const std::vector<uint8_t>& buf, size_t n_paren,int num_nodes, int nd,
                      std::vector<double>& xs,std::vector<double>& ys,std::vector<double>& zs,size_t& out_end) {

    if ((nd != 2 && nd != 3) || num_nodes < 0) return false;

    const size_t stride = (size_t)nd * 8;                                   // 每节点占用的字节数
    const size_t n_data = (size_t)num_nodes * stride;                       // 数据区总字节数
    const size_t start = n_paren + 1;                                       // 数据区起始偏移
    if (start > buf.size() || n_data > buf.size() - start) return false;    // 越界检查

    xs.resize((size_t)num_nodes);
    ys.resize((size_t)num_nodes);
    zs.resize((size_t)num_nodes);

    for (int i = 0; i < num_nodes; ++i) {
        const uint8_t* p = buf.data() + start + (size_t)i * stride;
        xs[i] = read_f64(p + 0);
        ys[i] = read_f64(p + 8);
        if (nd == 3) zs[i] = read_f64(p + 16);
    }
    out_end = start + n_data;
    return true;
}

size_t form_facezone(const std::vector<uint8_t>& buf, size_t startpoint,long &first, long& last){
    size_t p = startpoint + 5;                                          // 跳过 "(3013"
    while (p < buf.size() && buf[p] != '(') ++p;                        // 找到了左括号,在左括号停下
    ++p;                                                                // 停在了左括号右侧第一个字符
    while (p < buf.size() && (buf[p] == ' ' || buf[p] == '\t')) ++p;    // 忽略空格,前进到第一个有效字符,此时是面号
    while (p < buf.size() && buf[p] != ' ') ++p;                        // 翻越面号区间,停在空格
    while (p < buf.size() && buf[p] == ' ') ++p;                        // 翻越第二个空格,停在了开始面编号
    first = parse_hex_at(buf.data(), buf.size(), p);      // 读取开始面这个16进制
    while (p < buf.size() && buf[p] != ' ') ++p;
    while (p < buf.size() && buf[p] == ' ') ++p;
    last = parse_hex_at(buf.data(), buf.size(), p);       // 读取结束面这个16进制
    size_t paren = startpoint;
    while (paren < buf.size() && buf[paren] != '\n') ++paren;
    while (paren < buf.size() && buf[paren] != '(') ++paren;           // 跳过后面所有没用的东西,直接到数据区
    return paren;
}
