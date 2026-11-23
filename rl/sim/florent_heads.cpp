#include "florent_heads.hpp"

#include <cmath>
#include <limits>

namespace florent_rl {
using namespace fe;

const int BUILD_ORDER[9] = {CONVEYOR, SPLITTER, HARVESTER, GUNNER, SENTINEL, LAUNCHER, SPIKE, BARRIER, MINE};
const int CARD4[4] = {N, E, S, fe::W};
const int CONVERT_TENTHS[5] = {0, 10, 25, 50, 100};  // percent of current Voltium

static bool directional(int t) { return t == CONVEYOR || t == SPLITTER || t == GUNNER || t == SENTINEL; }

static const std::vector<int> HEADS[NUM_POLICIES] = {
    {HC_SPAWN, HC_CONVERT},
    {HB_MODE, HB_MOVE, HB_BTYPE, HB_BTILE, HB_BDIR, HB_ATTACK, HB_HEAL, HB_DESTROY},
    {HG_MODE, HG_FIRE, HG_ROTATE},
    {HS_MODE, HS_FIRE},
    {HL_MODE, HL_BOT, HL_DEST},
};

const std::vector<int>& policy_heads(int p) { return HEADS[p]; }
int policy_logit_count(int p) { int n = 0; for (int h : HEADS[p]) n += HEAD_SIZE[h]; return n; }
int head_offset(int p, int head) {
    int off = 0;
    for (int h : HEADS[p]) { if (h == head) return off; off += HEAD_SIZE[h]; }
    return -1;
}
int policy_of_type(int t) {
    switch (t) {
    case CORE: return P_CORE;
    case BUILDER: return P_BUILDER;
    case GUNNER: return P_GUNNER;
    case SENTINEL: return P_SENTINEL;
    case LAUNCHER: return P_LAUNCHER;
    default: return -1;
    }
}

// ---- legal option tables -------------------------------------------------
struct CoreOpts { uint8_t spawn[13] = {1}; uint8_t conv[5] = {1}; int amount[5] = {0}; int sx[13], sy[13]; };
static void core_opts(const Game& g, int a, CoreOpts& o) {
    const Entity& e = g.ents[a];
    int k = 1;
    for (int dy = -1; dy <= 2; dy++)
        for (int dx = -1; dx <= 2; dx++) {
            if (dx >= 0 && dx <= 1 && dy >= 0 && dy <= 1) continue;
            o.sx[k] = e.x + dx; o.sy[k] = e.y + dy;
            o.spawn[k] = g.inb(o.sx[k], o.sy[k]) && g.can_spawn(a, o.sx[k], o.sy[k]) == OK;
            k++;
        }
    int vo = g.teams[e.team].vo;
    for (int i = 1; i < 5; i++) {
        o.amount[i] = (vo * CONVERT_TENTHS[i] + 99) / 100;
        o.conv[i] = vo > 0 && o.amount[i] > 0 && g.can_convert(a, o.amount[i]) == OK;
    }
}

struct BuilderOpts {
    uint8_t move[4] = {0}, attack[4] = {0}, heal[4] = {0}, destroy[4] = {0};
    uint8_t build[9][4][8] = {};      // [type][tile][dir index] (non-directional: dir index 0)
    int tx[4], ty[4];
};
static void builder_opts(const Game& g, int a, BuilderOpts& o) {
    const Entity& e = g.ents[a];
    for (int i = 0; i < 4; i++) {
        o.tx[i] = e.x + DX[CARD4[i]]; o.ty[i] = e.y + DY[CARD4[i]];
        o.move[i] = g.can_move(a, CARD4[i]) == OK;
        bool in = g.inb(o.tx[i], o.ty[i]);
        o.attack[i] = in && g.can_attack(a, o.tx[i], o.ty[i]) == OK;
        o.heal[i] = in && g.can_heal(a, o.tx[i], o.ty[i]) == OK;
        o.destroy[i] = in && g.can_destroy(a, o.tx[i], o.ty[i]) == OK;
        for (int t = 0; t < 9; t++) {
            int type = BUILD_ORDER[t];
            if (!in) continue;
            if (directional(type)) {
                for (int d = 0; d < 8; d++) o.build[t][i][d] = g.can_build(a, type, o.tx[i], o.ty[i], d + 1) == OK;
            } else {
                o.build[t][i][0] = g.can_build(a, type, o.tx[i], o.ty[i], C) == OK;
            }
        }
    }
}

struct LineOpts { uint8_t fire[5] = {0}, rot[8] = {0}; int fx[5], fy[5]; };
static void line_opts(const Game& g, int a, LineOpts& o) {
    const Entity& e = g.ents[a];
    for (int k = 0; k < 5; k++) {
        o.fx[k] = e.x + DX[e.dir] * (k + 1); o.fy[k] = e.y + DY[e.dir] * (k + 1);
        o.fire[k] = g.inb(o.fx[k], o.fy[k]) && g.can_fire(a, o.fx[k], o.fy[k]) == OK;
    }
    if (e.type == GUNNER)
        for (int d = 0; d < 8; d++) o.rot[d] = g.can_rotate(a, d + 1) == OK;
}

struct LauncherOpts { uint8_t ok[8][120] = {}; uint8_t bot[8] = {0}; int bx[8], by[8], dx[120], dy[120]; };
static void launcher_opts(const Game& g, int a, LauncherOpts& o) {
    const Entity& e = g.ents[a];
    int j = 0;
    for (int dy = -5; dy <= 5; dy++)
        for (int dx = -5; dx <= 5; dx++) {
            if (!dx && !dy) continue;
            o.dx[j] = e.x + dx; o.dy[j] = e.y + dy; j++;
        }
    int i = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            o.bx[i] = e.x + dx; o.by[i] = e.y + dy;
            bool has_bot = g.inb(o.bx[i], o.by[i]) && g.bot[g.idx(o.bx[i], o.by[i])] >= 0;
            if (has_bot)
                for (int k = 0; k < 120; k++)
                    if (g.inb(o.dx[k], o.dy[k]) && g.can_launch(a, o.bx[i], o.by[i], o.dx[k], o.dy[k]) == OK) {
                        o.ok[i][k] = 1; o.bot[i] = 1;
                    }
            i++;
        }
}

static bool any(const uint8_t* m, int n) { for (int i = 0; i < n; i++) if (m[i]) return true; return false; }

bool has_choice(const Game& g, int a) {
    if (a < 0 || a >= int(g.ents.size()) || !g.ents[a].alive) return false;
    switch (g.ents[a].type) {
    case CORE: { CoreOpts o; core_opts(g, a, o); return any(o.spawn + 1, 12) || any(o.conv + 1, 4); }
    case BUILDER: {
        BuilderOpts o; builder_opts(g, a, o);
        return any(o.move, 4) || any(o.attack, 4) || any(o.heal, 4) || any(o.destroy, 4) ||
               any(&o.build[0][0][0], 9 * 4 * 8);
    }
    case GUNNER: case SENTINEL: { LineOpts o; line_opts(g, a, o); return any(o.fire, 5) || any(o.rot, 8); }
    case LAUNCHER: { LauncherOpts o; launcher_opts(g, a, o); return any(o.bot, 8); }
    default: return false;
    }
}

bool spike_action(const Game& g, int a, Candidate& out) {
    const Entity& e = g.ents[a];
    if (e.type != SPIKE) return false;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            int x = e.x + dx, y = e.y + dy;
            if (!g.inb(x, y)) continue;
            int u = g.bot[g.idx(x, y)], b = g.tile_building(a, x, y);
            bool enemy = (u >= 0 && g.ents[u].team != e.team) || (b >= 0 && g.ents[b].team != e.team);
            if (enemy && g.can_fire(a, x, y) == OK) { out = {ActionKind::Fire, x, y}; return true; }
        }
    return false;
}

// ---- masked softmax --------------------------------------------------------
float masked_logp(const float* l, const std::vector<uint8_t>& m, int c) {
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < m.size(); i++) if (m[i]) mx = std::max(mx, l[i]);
    double z = 0;
    for (size_t i = 0; i < m.size(); i++) if (m[i]) z += std::exp(double(l[i]) - mx);
    return float(double(l[c]) - mx - std::log(z));
}

int argmax_choice(const float* l, const std::vector<uint8_t>& m, float& logp) {
    int best = -1;
    for (size_t i = 0; i < m.size(); i++)
        if (m[i] && (best < 0 || l[i] > l[best])) best = int(i);   // ties: lowest index
    logp = best >= 0 ? masked_logp(l, m, best) : 0.f;
    return best;
}

// ---- decode ----------------------------------------------------------------
Decision decide(const Game& g, int a, const float* logits, const Chooser& choose) {
    Decision d;
    const Entity& e = g.ents[a];
    int p = policy_of_type(e.type);
    auto run = [&](int head, std::vector<uint8_t> mask) {
        HeadRecord r; r.head = head; r.mask = std::move(mask);
        r.choice = choose(head, logits + head_offset(p, head), r.mask, r.logp);
        d.records.push_back(r);
        return r.choice;
    };
    auto vec = [](const uint8_t* m, int n) { return std::vector<uint8_t>(m, m + n); };

    if (p == P_CORE) {
        CoreOpts o; core_opts(g, a, o);
        int s = run(HC_SPAWN, vec(o.spawn, 13));
        if (s > 0) d.actions.push_back({ActionKind::Spawn, o.sx[s], o.sy[s]});
        int c = run(HC_CONVERT, vec(o.conv, 5));
        if (c > 0) d.actions.push_back({ActionKind::Convert, o.amount[c]});
    } else if (p == P_BUILDER) {
        BuilderOpts o; builder_opts(g, a, o);
        uint8_t bt[9] = {0};
        for (int t = 0; t < 9; t++) bt[t] = any(&o.build[t][0][0], 4 * 8);
        uint8_t mode[6] = {1, any(o.move, 4), any(bt, 9), any(o.attack, 4), any(o.heal, 4), any(o.destroy, 4)};
        int m = run(HB_MODE, vec(mode, 6));
        if (m == 1) {
            int k = run(HB_MOVE, vec(o.move, 4));
            d.actions.push_back({ActionKind::Move, CARD4[k]});
        } else if (m == 2) {
            int t = run(HB_BTYPE, vec(bt, 9));
            uint8_t tiles[4];
            for (int i = 0; i < 4; i++) tiles[i] = any(o.build[t][i], 8);
            int i = run(HB_BTILE, vec(tiles, 4));
            int dir = C;
            if (directional(BUILD_ORDER[t])) dir = run(HB_BDIR, vec(o.build[t][i], 8)) + 1;
            d.actions.push_back({ActionKind::Build, BUILD_ORDER[t], o.tx[i], o.ty[i], dir});
        } else if (m == 3) {
            int i = run(HB_ATTACK, vec(o.attack, 4));
            d.actions.push_back({ActionKind::Attack, o.tx[i], o.ty[i]});
        } else if (m == 4) {
            int i = run(HB_HEAL, vec(o.heal, 4));
            d.actions.push_back({ActionKind::Heal, o.tx[i], o.ty[i]});
        } else if (m == 5) {
            int i = run(HB_DESTROY, vec(o.destroy, 4));
            d.actions.push_back({ActionKind::Destroy, o.tx[i], o.ty[i]});
        }
    } else if (p == P_GUNNER) {
        LineOpts o; line_opts(g, a, o);
        uint8_t mode[3] = {1, any(o.fire, 5), any(o.rot, 8)};
        int m = run(HG_MODE, vec(mode, 3));
        if (m == 1) { int k = run(HG_FIRE, vec(o.fire, 5)); d.actions.push_back({ActionKind::Fire, o.fx[k], o.fy[k]}); }
        else if (m == 2) { int k = run(HG_ROTATE, vec(o.rot, 8)); d.actions.push_back({ActionKind::Rotate, k + 1}); }
    } else if (p == P_SENTINEL) {
        LineOpts o; line_opts(g, a, o);
        uint8_t mode[2] = {1, any(o.fire, 5)};
        if (run(HS_MODE, vec(mode, 2)) == 1) {
            int k = run(HS_FIRE, vec(o.fire, 5));
            d.actions.push_back({ActionKind::Fire, o.fx[k], o.fy[k]});
        }
    } else if (p == P_LAUNCHER) {
        LauncherOpts o; launcher_opts(g, a, o);
        uint8_t mode[2] = {1, any(o.bot, 8)};
        if (run(HL_MODE, vec(mode, 2)) == 1) {
            int i = run(HL_BOT, vec(o.bot, 8));
            int k = run(HL_DEST, vec(o.ok[i], 120));
            d.actions.push_back({ActionKind::Launch, o.bx[i], o.by[i], o.dx[k], o.dy[k]});
        }
    }
    return d;
}

}  // namespace florent_rl
