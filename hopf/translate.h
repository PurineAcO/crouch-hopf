#include <cstddef>
#include <cstdint>
#include <vector>

// 读取无符号32位整数.
uint32_t read_u32(const uint8_t* p);

// 读取有符号32位整数.
int32_t read_i32(const uint8_t* p);

// 读取双精度浮点数.
double read_f64(const uint8_t* p);

// 查找字符串pat,返回第一次出现的起始下标,否则为-1.
size_t find_bytes(const uint8_t* buf, size_t len, const char* pat);

// 解析连续十六进制数字.
long parse_hex_at(const uint8_t* buf, size_t len, size_t pos);

// 解析声明行中的十六进制数量,在decl_pat后解析一个十六进制value,返回是否解析成功.
bool parse_decl_count(const uint8_t* buf, size_t len, const char* decl_pat, int& count);

// 解析连续十进制数字.
long parse_dec_at(const uint8_t* buf, size_t len, size_t pos);

// 解析声明行中的十进制值,在decl_pat后解析一个十进制值到value,返回是否解析成功.
bool parse_decl_dec(const uint8_t* buf, size_t len, const char* decl_pat, int& value);

// 定位二进制数据块 '(' 的位置:先找到decl_marker,随后找到其下一行的左括号,以此为数据开端.
bool locate_paren(const uint8_t* buf, size_t len, const char* decl_marker, size_t& paren);

// 读出节点坐标块中的所有节点坐标(2D/3D),返回是否解析成功.
bool read_node_coords(const std::vector<uint8_t>& buf, size_t n_paren,int num_nodes, int nd,
                      std::vector<double>& xs,std::vector<double>& ys,std::vector<double>& zs,size_t& out_end);

// 读出一个FaceZone的起始终止位点
size_t form_facezone(const std::vector<uint8_t>& buf, size_t startpoint,long &first, long& last);