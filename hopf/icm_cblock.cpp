// icm_cblock.cpp
// 从 ICM_translated.txt 识别 C-block 结构化网格 (半O + 矩形), 导出 icm_cblock.txt
// 用法: icm_cblock [输入.msh] [输出.txt]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

typedef std::pair<int, int> Edge;      // 归一化边 (小, 大)
typedef std::pair<double, double> Pt;

struct CellRec {                       // 已归类单元的块信息
    int blk = 0, i = 0, j = 0;
    std::vector<int> quad;             // 4 节点 (无序)
};

// ---------- 读取网格 ----------
struct Mesh {
    std::map<int, Pt> nodes;           // 节点号 -> 坐标
    std::map<int, std::set<int>> cells;// 单元 -> 节点集
    std::map<Edge, std::vector<int>> ec;  // 边 -> 关联单元(不含0)
    std::vector<Edge> eorder;          // 边按文件首次出现顺序 (与 python dict 一致)
    long long faces = 0;               // 面总数
};

static Mesh load(const char* fn) {
    Mesh m;
    FILE* fp = fopen(fn, "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "(3010 ", 6) == 0) {       // 节点块
            fgets(line, sizeof(line), fp);           // 跳过 '('
            while (fgets(line, sizeof(line), fp)) {
                if (strchr(line, ')')) break;
                int id; double x, y;
                if (sscanf(line, "%d %lf %lf", &id, &x, &y) == 3)
                    m.nodes[id] = {x, y};
            }
        } else if (strncmp(line, "(3013 (", 7) == 0) {  // 面块
            fgets(line, sizeof(line), fp);           // 跳过 '('
            while (fgets(line, sizeof(line), fp)) {
                if (strchr(line, ')')) break;
                int n1, n2, c1, c2;
                if (sscanf(line, "%d %d %d %d", &n1, &n2, &c1, &c2) != 4) continue;
                ++m.faces;
                if (c1) { m.cells[c1].insert(n1); m.cells[c1].insert(n2); }
                if (c2) { m.cells[c2].insert(n1); m.cells[c2].insert(n2); }
                Edge e = {std::min(n1, n2), std::max(n1, n2)};
                if (!m.ec.count(e)) m.eorder.push_back(e);
                if (c1) m.ec[e].push_back(c1);
                if (c2) m.ec[e].push_back(c2);
            }
        }
    }
    fclose(fp);
    return m;
}

// ---------- 小工具 ----------
// 取路径所在目录 (无 '/' 时返回 ".")
static std::string dir_of(const char* path) {
    std::string s = path;
    size_t pos = s.find_last_of('/');
    if (pos == std::string::npos) return ".";
    return pos == 0 ? "/" : s.substr(0, pos);
}

static Pt emid(const Mesh& m, const Edge& e) {
    return {(m.nodes.at(e.first).first + m.nodes.at(e.second).first) / 2,
            (m.nodes.at(e.first).second + m.nodes.at(e.second).second) / 2};
}

// 边界边排成闭合环, 返回节点序列
static std::vector<int> boundary_loop(const std::vector<Edge>& edges) {
    std::map<int, std::vector<int>> deg;
    for (const Edge& e : edges) {
        deg[e.first].push_back(e.second);
        deg[e.second].push_back(e.first);
    }
    std::vector<int> loop;
    int start = deg.begin()->first, prev = -1, cur = start;
    loop.push_back(start);
    while ((int)loop.size() < (int)deg.size() + 2) {
        int nxt = -1;
        for (int n : deg[cur]) if (n != prev) { nxt = n; break; }
        if (nxt == start) break;
        loop.push_back(nxt);
        prev = cur; cur = nxt;
    }
    return loop;
}

static std::vector<Edge> edges_of(const Mesh& m, int c) {
    std::vector<Edge> es;
    std::vector<int> ns(m.cells.at(c).begin(), m.cells.at(c).end());
    for (int a : ns) for (int b : ns)
        if (a < b && m.ec.count({a, b})) es.push_back({a, b});
    return es;
}

// es 中与 e 无公共端点的边 (对边)
static bool opp(const std::vector<Edge>& es, const Edge& e, Edge& out) {
    for (const Edge& x : es) {
        if (x == e) continue;
        if (x.first != e.first && x.second != e.first &&
            x.first != e.second && x.second != e.second) { out = x; return true; }
    }
    return false;
}

// 边 e 上单元 c 的另一侧单元 (e 恰有两个单元时), 否则 0
static int other(const Mesh& m, const Edge& e, int c) {
    auto it = m.ec.find(e);
    if (it == m.ec.end() || it->second.size() != 2) return 0;
    return it->second[0] == c ? it->second[1] : it->second[0];
}

// ---------- 主流程 ----------
int main(int argc, char** argv) {
    const char* in = argc > 1 ? argv[1] : "testdata/ICM_translated.txt";
    std::string out = argc > 2 ? argv[2] : dir_of(in) + "/icm_cblock.txt";  // 默认输出与输入同目录
    const Mesh& m = load(in);

    // 边界边: 距原点 >1000 为 FAR, 否则为 WING
    std::vector<Edge> far_edges, wing_edges;
    for (const Edge& e : m.eorder) {
        const auto& cl = m.ec.at(e);
        if (cl.size() != 1) continue;
        Pt mid = emid(m, e);
        if (std::hypot(mid.first, mid.second) > 1000) far_edges.push_back(e);
        else wing_edges.push_back(e);
    }
    std::set<Edge> far_set(far_edges.begin(), far_edges.end());
    printf("nodes %d, faces %lld\n", (int)m.nodes.size(), (long long)m.faces);
    printf("cells %d, far edges %d, wing edges %d\n",
           (int)m.cells.size(), (int)far_edges.size(), (int)wing_edges.size());

    // ---- 半O: WING 环 + 逐环向外追踪 ----
    std::vector<int> wing_loop = boundary_loop(wing_edges);
    int n_slot = (int)wing_loop.size();
    std::map<Edge, int> wpos;
    for (int k = 0; k < n_slot; ++k)
        wpos[{std::min(wing_loop[k], wing_loop[(k + 1) % n_slot]),
              std::max(wing_loop[k], wing_loop[(k + 1) % n_slot])}] = k;

    std::vector<int> ring0(n_slot, 0);              // 首环单元 (0=无)
    std::vector<Edge> in0(n_slot);                  // 各槽位的入边
    std::vector<bool> in0ok(n_slot, false);
    for (const Edge& e : wing_edges) {
        Edge se = {std::min(e.first, e.second), std::max(e.first, e.second)};
        int k = wpos.at(se);
        ring0[k] = m.ec.at(se)[0];
        in0[k] = se; in0ok[k] = true;
    }

    std::vector<std::vector<int>> rings = {ring0};
    std::vector<std::vector<Edge>> rings_in = {in0};
    std::vector<std::vector<bool>> rings_ok = {in0ok};
    while (true) {
        const std::vector<int>& cur = rings.back();
        const std::vector<Edge>& curin = rings_in.back();
        std::vector<int> nxt(n_slot, 0);
        std::vector<Edge> nxtin(n_slot);
        std::vector<bool> nxtok(n_slot, false);
        bool any = false;
        for (int p = 0; p < n_slot; ++p) {
            if (!cur[p]) continue;
            Edge oe;
            if (!opp(edges_of(m, cur[p]), curin[p], oe) || far_set.count(oe)) continue;
            int c2 = other(m, oe, cur[p]);
            if (!c2) continue;
            nxt[p] = c2; nxtin[p] = oe; nxtok[p] = true; any = true;
        }
        if (!any) break;
        rings.push_back(nxt); rings_in.push_back(nxtin); rings_ok.push_back(nxtok);
    }
    int n_i1 = (int)rings.size();                   // 半O环数

    // 楔槽(矩形缺口) / 半O槽位
    std::vector<int> wedge_pos;
    std::set<int> halfo_pos;
    for (int p = 0; p < n_slot; ++p) {
        if (!rings[0][p]) continue;
        if (!rings.back()[p]) wedge_pos.push_back(p);
        else halfo_pos.insert(p);
    }
    if (wedge_pos.empty()) { fprintf(stderr, "no wedge slots found\n"); return 1; }
    int s = (wedge_pos.back() + 1) % n_slot;
    while (!halfo_pos.count(s)) s = (s + 1) % n_slot;
    std::vector<int> chain;                         // 半O槽位链 (环形)
    for (int p = s; (int)chain.size() < (int)halfo_pos.size(); p = (p + 1) % n_slot)
        if (halfo_pos.count(p)) chain.push_back(p);
    int n_j1 = (int)chain.size();                   // 半O支数
    int n_i2 = 0;                                   // 矩形层数
    for (int r = 0; r < n_i1; ++r) if (rings[r][wedge_pos[0]]) ++n_i2;
    printf("half-O: %d rings x %d branches (wedge %d), rect layers %d\n",
           n_i1, n_j1, (int)wedge_pos.size(), n_i2);

    // 各环的节点序列 (槽位 -> 节点)
    auto extract = [&](const std::vector<Edge>& edges, const std::vector<bool>& ok) {
        std::vector<int> nr(n_slot, 0);
        const Edge& e0 = edges[chain[0]];
        const Edge& e1 = edges[chain[1]];
        int shared = (e0.first == e1.first || e0.first == e1.second) ? e0.first : e0.second;
        nr[chain[0]] = (e0.first == shared) ? e0.second : e0.first;
        nr[chain[1]] = shared;
        for (int k = 1; k < (int)chain.size(); ++k) {
            int p = chain[k];
            if (!ok[p]) continue;
            const Edge& e = edges[p];
            if (e.first == nr[p]) nr[(p + 1) % n_slot] = e.second;
            else if (e.second == nr[p]) nr[(p + 1) % n_slot] = e.first;
        }
        return nr;
    };
    std::vector<std::vector<int>> node_rings;
    for (int r = 0; r < n_i1; ++r)
        node_rings.push_back(extract(rings_in[r], rings_ok[r]));
    std::vector<Edge> outer(n_slot);                // 最外层单元的外边
    std::vector<bool> outerok(n_slot, false);
    for (int p : chain) {
        int c = rings.back()[p];
        if (!c) continue;
        Edge oe;
        if (opp(edges_of(m, c), rings_in.back()[p], oe)) { outer[p] = oe; outerok[p] = true; }
    }
    node_rings.push_back(extract(outer, outerok));  // 半O外边界节点环

    // ---- 矩形: 背边识别 + 列追踪 ----
    std::vector<int> far_loop = boundary_loop(far_edges);
    int nf = (int)far_loop.size();
    double maxx = -1e300;
    std::vector<double> fx(nf);
    for (int k = 0; k < nf; ++k) {
        Edge e = {std::min(far_loop[k], far_loop[(k + 1) % nf]),
                  std::max(far_loop[k], far_loop[(k + 1) % nf])};
        fx[k] = emid(m, e).first;
        maxx = std::max(maxx, fx[k]);
    }
    std::vector<Edge> main_back;                    // 背边 = FAR 环上 x 最大段
    for (int k = 0; k < nf; ++k)
        if (fx[k] >= maxx - 5)
            main_back.push_back({std::min(far_loop[k], far_loop[(k + 1) % nf]),
                                 std::max(far_loop[k], far_loop[(k + 1) % nf])});
    std::sort(main_back.begin(), main_back.end(),
              [&](const Edge& a, const Edge& b) { return emid(m, a).second < emid(m, b).second; });
    int n_j2 = (int)main_back.size();               // 矩形支数
    printf("rect: %d branches x %d layers\n", n_j2, n_i2);

    std::vector<std::vector<int>> cols;             // 每列单元
    std::vector<std::vector<Edge>> colfront;        // 每列层间边
    std::vector<Edge> colback;                      // 每列背边
    for (const Edge& be : main_back) {
        int c0 = m.ec.at(be)[0];
        bool dup = false;
        for (const auto& cc : cols) if (cc[0] == c0) { dup = true; break; }
        if (dup) continue;
        std::vector<int> col = {c0};
        std::vector<Edge> fr;
        Edge fe = be;
        for (int step = 0; step < 200; ++step) {
            Edge oe;
            if (!opp(edges_of(m, col.back()), fe, oe)) break;
            int c2 = other(m, oe, col.back());
            if (!c2) break;
            fr.push_back(oe);
            col.push_back(c2);
            fe = oe;
        }
        int m1 = std::min(n_i2, (int)col.size());
        int m2 = std::min(n_i2 - 1, (int)fr.size());
        cols.push_back(std::vector<int>(col.begin(), col.begin() + m1));
        colfront.push_back(std::vector<Edge>(fr.begin(), fr.begin() + m2));
        colback.push_back(be);
    }

    // 每列 n_i2+1 条边: 背边 + 层间边 + 前沿边
    std::vector<std::vector<Edge>> all_edges;
    for (int k = 0; k < (int)cols.size(); ++k) {
        std::vector<Edge> ae = {colback[k]};
        ae.insert(ae.end(), colfront[k].begin(), colfront[k].end());
        Edge e79;
        if (opp(edges_of(m, cols[k].back()), colfront[k].back(), e79))
            ae.push_back({std::min(e79.first, e79.second), std::max(e79.first, e79.second)});
        all_edges.push_back(ae);
    }

    // 矩形节点网格 Nr[(层,列)]
    std::map<std::pair<int, int>, int> Nr;
    for (int i = 0; i <= n_i2; ++i) {
        const Edge& e1 = all_edges[0][i];
        Nr[{i + 1, 1}] = e1.first;
        Nr[{i + 1, 2}] = e1.second;
        for (int j = 0; j < n_j2 - 1; ++j) {
            const Edge& e = all_edges[j + 1][i];
            int base = Nr[{i + 1, j + 2}];
            if (e.first == base) Nr[{i + 1, j + 3}] = e.second;
            else if (e.second == base) Nr[{i + 1, j + 3}] = e.first;
        }
    }

    // ---- 组装: 单元 -> 块 ----
    std::map<std::set<int>, int> key2cell;
    for (const auto& kv : m.cells) key2cell[kv.second] = kv.first;
    std::map<int, CellRec> rec;
    auto add = [&](const std::set<int>& q, int b, int i, int j) {
        auto it = key2cell.find(q);
        if (it == key2cell.end()) return;
        CellRec& r = rec[it->second];
        r.blk = b; r.i = i; r.j = j;
        r.quad.assign(q.begin(), q.end());
    };

    std::vector<int> node_seq = chain;              // 半O支序列 (n_j1+1)
    node_seq.push_back(wedge_pos[0]);
    for (int i = 0; i < n_i1; ++i)
        for (int j = 0; j < n_j1; ++j) {
            int p0 = node_seq[j], p1 = node_seq[j + 1];
            add({node_rings[i][p0], node_rings[i][p1],
                 node_rings[i + 1][p1], node_rings[i + 1][p0]}, 1, i + 1, j + 1);
        }
    for (int i = 0; i < n_i2; ++i)
        for (int j = 0; j < n_j2; ++j)
            add({Nr[{i + 1, j + 1}], Nr[{i + 2, j + 1}],
                 Nr[{i + 2, j + 2}], Nr[{i + 1, j + 2}]}, 2, i + 1, j + 1);

    int nb1 = 0, nb2 = 0;
    for (const auto& kv : rec) (kv.second.blk == 1 ? nb1 : nb2)++;
    printf("half-O %d + rect %d = %d\n", nb1, nb2, (int)rec.size());

    // ---- 几何量 ----
    auto geom = [&](const std::vector<int>& q, double& cx, double& cy,
                    double& vol, std::vector<int>& o) {
        cx = 0; cy = 0;
        for (int n : q) { cx += m.nodes.at(n).first; cy += m.nodes.at(n).second; }
        cx /= 4; cy /= 4;
        o = q;
        std::sort(o.begin(), o.end(), [&](int a, int b) {
            return std::atan2(m.nodes.at(a).second - cy, m.nodes.at(a).first - cx) <
                   std::atan2(m.nodes.at(b).second - cy, m.nodes.at(b).first - cx);
        });
        double a = 0;
        for (int k = 0; k < 4; ++k) {
            int u = o[k], v = o[(k + 1) % 4];
            a += m.nodes.at(u).first * m.nodes.at(v).second -
                 m.nodes.at(v).first * m.nodes.at(u).second;
        }
        vol = std::fabs(a) / 2;
    };

    // ---- 邻接 + 边界类型 ----
    std::map<int, std::set<int>> neigh;
    for (const auto& kv : m.ec) {
        if (kv.second.size() != 2) continue;
        neigh[kv.second[0]].insert(kv.second[1]);
        neigh[kv.second[1]].insert(kv.second[0]);
    }
    std::map<Edge, int> btype;
    for (const Edge& e : far_edges) btype[e] = 1;
    for (const Edge& e : wing_edges) btype[e] = 2;

    // ---- 导出 ----
    FILE* fp = fopen(out.c_str(), "w");
    if (!fp) { fprintf(stderr, "cannot open output %s\n", out.c_str()); return 1; }
    int N = (int)m.cells.size();
    fprintf(fp, "# C-block: half O: %d x %d, rect: %d x %d\n", n_i1, n_j1, n_i2, n_j2);
    fprintf(fp, "%d %d\n", N, (int)m.nodes.size());
    for (const auto& kv : m.cells) {
        int c = kv.first;
        double cx, cy, vol;
        std::vector<int> o;
        int b = 2, i = 0, j = 0;
        if (rec.count(c)) {
            const CellRec& r = rec[c];
            b = r.blk; i = r.i; j = r.j;
            geom(r.quad, cx, cy, vol, o);
        } else {
            geom(std::vector<int>(kv.second.begin(), kv.second.end()), cx, cy, vol, o);
        }
        std::vector<int> nb(neigh[c].begin(), neigh[c].end());
        std::sort(nb.begin(), nb.end());
        nb.resize(4, 0);
        int bts[4];
        for (int t = 0; t < 4; ++t) {
            Edge e = {std::min(o[t], o[(t + 1) % 4]), std::max(o[t], o[(t + 1) % 4])};
            auto it = m.ec.find(e);
            int bt = 0;
            if (it != m.ec.end() && it->second.size() == 2) {
                int b0 = rec.count(it->second[0]) ? rec[it->second[0]].blk : 0;
                int b1 = rec.count(it->second[1]) ? rec[it->second[1]].blk : 0;
                if (b0 != b1) { auto q = btype.find(e); bt = q == btype.end() ? 0 : q->second; }
            } else {
                auto q = btype.find(e);
                bt = q == btype.end() ? 0 : q->second;
            }
            bts[t] = bt;
        }
        fprintf(fp, "%d %d %d %.12g %.12g %.12g %d %d %d %d %d %d %d %d %d %d %d %d\n",
                b, i, j, cx, cy, vol,
                o[0], o[1], o[2], o[3],
                nb[0], nb[1], nb[2], nb[3],
                bts[0], bts[1], bts[2], bts[3]);
    }
    for (const auto& kv : m.nodes)
        fprintf(fp, "%d %.12g %.12g\n", kv.first, kv.second.first, kv.second.second);
    fclose(fp);
    return 0;
}
