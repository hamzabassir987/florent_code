#include "map_gen.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <sstream>

namespace florent_rl {
namespace {

enum : uint8_t { EMPTY = 0, WALL = 1, ORE_TI = 2, ORE_VO = 4, VOID_ = 5 };

struct Map {
    int W = 0, H = 0, sym = 0;
    std::vector<uint8_t> env;
    int core[2][2] = {{0, 0}, {0, 0}};
    int pit_axis = -1, pit_at = 0, pit_w = 0;
    uint8_t& at(int x, int y) { return env[size_t(y) * W + x]; }
    uint8_t get(int x, int y) const { return env[size_t(y) * W + x]; }
    bool inb(int x, int y) const { return x >= 0 && y >= 0 && x < W && y < H; }
    void mirror(int x, int y, int& mx, int& my) const {
        mx = (sym == 2) ? x : W - 1 - x;
        my = (sym == 1) ? y : H - 1 - y;
    }
    void set2(int x, int y, uint8_t v) { int mx, my; mirror(x, y, mx, my); at(x, y) = v; at(mx, my) = v; }
    bool in_core_zone(int x, int y, int margin) const {
        for (int t = 0; t < 2; t++)
            if (x >= core[t][0] - margin && x <= core[t][0] + 1 + margin && y >= core[t][1] - margin && y <= core[t][1] + 1 + margin) return true;
        return false;
    }
};

int ri(std::mt19937_64& rng, int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }
float rf(std::mt19937_64& rng) { return std::uniform_real_distribution<float>(0.f, 1.f)(rng); }

// protobuf writer
void var(std::string& o, uint64_t v) { while (v >= 0x80) { o.push_back(char(v | 0x80)); v >>= 7; } o.push_back(char(v)); }
void key(std::string& o, int f, int wt) { var(o, uint64_t(f) << 3 | uint64_t(wt)); }
void vfield(std::string& o, int f, uint64_t v) { if (v) { key(o, f, 0); var(o, v); } }     // proto3: zero omitted
void bfield(std::string& o, int f, const std::string& b) { key(o, f, 2); var(o, b.size()); o += b; }

std::string encode(const Map& m) {
    std::string o;
    vfield(o, 1, m.W);
    vfield(o, 2, m.H);
    for (int y = 0; y < m.H; y++) {
        std::string packed, row;
        for (int x = 0; x < m.W; x++) var(packed, m.get(x, y));
        bfield(row, 1, packed);
        bfield(o, 3, row);
    }
    for (int t = 0; t < 2; t++) {
        std::string c, p;
        vfield(p, 1, m.core[t][0]); vfield(p, 2, m.core[t][1]);
        vfield(c, 1, t + 1); vfield(c, 2, t); bfield(c, 3, p);
        bfield(o, 4, c);
    }
    vfield(o, 5, m.sym);
    if (m.pit_axis >= 0) {
        std::string p;
        vfield(p, 1, m.pit_axis); vfield(p, 2, m.pit_at); vfield(p, 3, m.pit_w);
        bfield(o, 8, p);
    }
    return o;
}

// tiles reachable on foot from core t (walls and void block; ore and cores are walkable)
int flood(const Map& m, int t, std::vector<uint8_t>& seen) {
    seen.assign(m.env.size(), 0);
    std::deque<std::pair<int, int>> q;
    q.push_back({m.core[t][0], m.core[t][1]});
    seen[size_t(m.core[t][1]) * m.W + m.core[t][0]] = 1;
    int n = 0;
    while (!q.empty()) {
        auto [x, y] = q.front(); q.pop_front(); n++;
        static const int D[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (auto& d : D) {
            int nx = x + d[0], ny = y + d[1];
            if (!m.inb(nx, ny)) continue;
            size_t i = size_t(ny) * m.W + nx;
            if (seen[i] || m.env[i] == WALL || m.env[i] == VOID_) continue;
            seen[i] = 1; q.push_back({nx, ny});
        }
    }
    return n;
}

bool try_generate(const MapGenParams& P, std::mt19937_64& rng, Map& m) {
    m = Map();
    const bool pit = rf(rng) < P.pit_share;
    m.sym = pit ? ri(rng, 1, 2) : ri(rng, 0, 2);
    // core_gap N >= 0: a random gap in 4..N per map (N capped at 24, the largest a
    // 30-tile map fits: 30 - 2 - (gap + 1) >= 2, parity); -1: random 7..21
    const int gmax = std::min(P.core_gap, 24);
    int gap = P.core_gap >= 0 ? ri(rng, std::clamp(P.core_gap_min, 0, gmax), gmax) : ri(rng, 7, 21);
    // size: as the real maps, grown if the requested gap needs it
    m.W = ri(rng, 16, 30); m.H = ri(rng, 17, 29);
    auto& span = (m.sym == 2) ? m.H : m.W;                      // the axis the cores are mirrored across
    if (m.sym != 0) span = std::max(span, std::min(30, gap + 5));
    else { m.W = std::max(m.W, std::min(30, gap + 5)); }
    m.env.assign(size_t(m.W) * m.H, EMPTY);
    // pit: a void band through the middle, on the mirror axis
    if (pit) {
        const int len = (m.sym == 1) ? m.W : m.H;
        std::vector<int> ws;
        for (int w = 1; w <= 4; w++) if ((len - w) % 2 == 0) ws.push_back(w);
        m.pit_w = ws[ri(rng, 0, int(ws.size()) - 1)];
        m.pit_at = (len - m.pit_w) / 2;
        m.pit_axis = (m.sym == 1) ? 0 : 1;
        for (int i = m.pit_at; i < m.pit_at + m.pit_w; i++)
            for (int j = 0; j < (m.sym == 1 ? m.H : m.W); j++)
                if (m.sym == 1) m.at(i, j) = VOID_; else m.at(j, i) = VOID_;
        gap = std::max(gap, m.pit_w + 2);
    }
    // cores: A, then B = A's mirror; Chebyshev gap between the footprints = max(|dx|, |dy|) - 1
    const int sep = gap + 1;                                    // |bx - ax| (or |by - ay|) on the mirrored axis
    auto place_axis = [&](int len, int& a) -> bool {            // a + (len - 2 - a) mirrored: len - 2 - 2a = sep
        const int twice = len - 2 - sep;
        if (twice < 2) return false;                            // a >= 1
        a = twice / 2;                                          // parity: gap may come out 1 smaller
        return true;
    };
    int ax = 0, ay = 0;
    if (m.sym == 1) { if (!place_axis(m.W, ax)) return false; ay = ri(rng, 1, m.H - 3); }
    else if (m.sym == 2) { if (!place_axis(m.H, ay)) return false; ax = ri(rng, 1, m.W - 3); }
    else {
        // 180 rotation: the gap on one axis, the other axis closer
        if (rf(rng) < 0.5f) {
            if (!place_axis(m.W, ax)) return false;
            const int lo = std::max(1, (m.H - 2 - sep) / 2), hi = (m.H - 2) / 2;
            ay = ri(rng, std::min(lo, hi), hi);
        } else {
            if (!place_axis(m.H, ay)) return false;
            const int lo = std::max(1, (m.W - 2 - sep) / 2), hi = (m.W - 2) / 2;
            ax = ri(rng, std::min(lo, hi), hi);
        }
    }
    m.core[0][0] = ax; m.core[0][1] = ay;
    m.core[1][0] = (m.sym == 2) ? ax : m.W - 2 - ax;         // B's footprint = the mirror of A's
    m.core[1][1] = (m.sym == 1) ? ay : m.H - 2 - ay;
    if (m.core[0][0] == m.core[1][0] && m.core[0][1] == m.core[1][1]) return false;
    for (int t = 0; t < 2; t++)
        for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++)
            if (!m.inb(m.core[t][0] + dx, m.core[t][1] + dy) || m.get(m.core[t][0] + dx, m.core[t][1] + dy) == VOID_) return false;
    // walls: small mirrored clusters, never next to a core or on the pit
    const int wall_target = int((P.wall_lo + rf(rng) * (P.wall_hi - P.wall_lo)) * m.W * m.H);
    int walls = 0, guard = 0;
    while (walls < wall_target && guard++ < 4000) {
        int x = ri(rng, 0, m.W - 1), y = ri(rng, 0, m.H - 1);
        const int len = ri(rng, 1, 5);
        for (int k = 0; k < len; k++) {
            if (m.inb(x, y) && m.get(x, y) == EMPTY && !m.in_core_zone(x, y, 2)) {
                int mx, my; m.mirror(x, y, mx, my);
                if (m.get(mx, my) == EMPTY && !m.in_core_zone(mx, my, 2)) { m.set2(x, y, WALL); walls += (mx == x && my == y) ? 1 : 2; }
            }
            if (rf(rng) < 0.5f) x += ri(rng, 0, 1) * 2 - 1; else y += ri(rng, 0, 1) * 2 - 1;
        }
    }
    // ore: mirrored clusters, off the core footprints
    auto place_ore = [&](uint8_t kind, int target) {
        int placed = 0, g = 0;
        while (placed < target && g++ < 4000) {
            int x = ri(rng, 0, m.W - 1), y = ri(rng, 0, m.H - 1);
            const int len = ri(rng, 1, 4);
            for (int k = 0; k < len && placed < target; k++) {
                if (m.inb(x, y) && m.get(x, y) == EMPTY && !m.in_core_zone(x, y, 0)) {
                    int mx, my; m.mirror(x, y, mx, my);
                    if (!(mx == x && my == y) && m.get(mx, my) == EMPTY && !m.in_core_zone(mx, my, 0)) { m.set2(x, y, kind); placed += 2; }
                }
                if (rf(rng) < 0.5f) x += ri(rng, 0, 1) * 2 - 1; else y += ri(rng, 0, 1) * 2 - 1;
            }
        }
    };
    place_ore(ORE_TI, 2 * ri(rng, 4, 18));
    place_ore(ORE_VO, 2 * ri(rng, 0, 3));
    // connectivity
    int open = 0;
    for (auto v : m.env) open += v != WALL && v != VOID_;
    std::vector<uint8_t> seen;
    const int reach = flood(m, 0, seen);
    const bool b_reached = seen[size_t(m.core[1][1]) * m.W + m.core[1][0]] != 0;
    if (!pit) return b_reached && reach >= int(0.9 * open);
    return reach >= int(0.9 * open / 2);
}

}  // namespace

std::string generate_map(const MapGenParams& P, std::mt19937_64& rng, MapGenInfo* info) {
    Map m;
    for (int attempt = 0; attempt < 200; attempt++)
        if (try_generate(P, rng, m)) break;
    if (info) {
        int ti = 0, vo = 0, wa = 0;
        for (auto v : m.env) { ti += v == ORE_TI; vo += v == ORE_VO; wa += v == WALL; }
        const int gap = std::max(std::abs(m.core[0][0] - m.core[1][0]), std::abs(m.core[0][1] - m.core[1][1])) - 1;
        std::ostringstream s;
        s << m.W << "x" << m.H << " sym " << m.sym << " gap " << gap << " walls " << wa << " ti " << ti << " vo " << vo;
        if (m.pit_axis >= 0) s << " pit axis " << m.pit_axis << " at " << m.pit_at << " w " << m.pit_w;
        info->W = m.W; info->H = m.H; info->sym = m.sym; info->gap = gap; info->ti = ti; info->vo = vo; info->walls = wa;
        info->pit = m.pit_axis >= 0; info->line = s.str();
    }
    return encode(m);
}

}  // namespace florent_rl
