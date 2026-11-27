#include "florent_heads.hpp"
#include "obs.hpp"

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
static void core_opts(const Game& g, int a, CoreOpts& o, const Targets* tg = nullptr) {
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
    // Voltium has no other use than ammo (no price, no tie-break): always convert
    // all of it (a single option, not recorded); "none" only when that is illegal
    (void)tg;
    if (o.conv[4]) { o.conv[0] = 0; o.conv[1] = o.conv[2] = o.conv[3] = 0; }
    else o.conv[1] = o.conv[2] = o.conv[3] = 0;
}

struct BuilderOpts {
    uint8_t move[4] = {0}, attack[4] = {0}, heal[4] = {0}, destroy[4] = {0};
    uint8_t build[9][4][8] = {};      // [type][tile][dir index] (non-directional: dir index 0)
    int tx[4], ty[4];
};
// What a shot / attack on (x, y) would damage, as seen by unit a (enemy mines
// are hidden from it). The engine lets every damaging action hit our own side,
// so the masks below allow only enemy targets:
//  * builder attack damages the building on the tile (never a builder);
//  * a gunner / sentinel shot hits the builder bot on the tile, else the building;
//  * spikes and mines only ever hit enemies (engine rule), launchers do no damage.
static bool enemy_building_on(const Game& g, int a, int x, int y) {
    if (!g.inb(x, y)) return false;
    const int b = g.tile_building(a, x, y);
    return b >= 0 && g.ents[b].team != g.ents[a].team;
}
static bool enemy_hit(const Game& g, int a, int x, int y) {
    if (!g.inb(x, y)) return false;
    const int u = g.bot[g.idx(x, y)], b = g.tile_building(a, x, y);
    const int hit = u >= 0 ? u : b;
    return hit >= 0 && g.ents[hit].team != g.ents[a].team;
}
static bool in_list(const std::vector<int>& v, int t) { for (int x : v) if (x == t) return true; return false; }
// sentinel / gunner built at (x, y) facing Dir d: does it aim at a target? (Targets)
static bool aims(const Game& g, int type, int x, int y, int d, const Targets& tg) {
    const bool diag = d % 2 == 0;                     // Dir: N=1 NE=2 E=3 ...
    const int len = type == SENTINEL ? (diag ? 4 : 5) : (diag ? 2 : 3);
    for (int k = 1; k <= len; k++) {
        const int xx = x + DX[d] * k, yy = y + DY[d] * k;
        if (!g.inb(xx, yy)) return false;
        const int t = g.idx(xx, yy);
        if (type == SENTINEL) { if (in_list(tg.core, t)) return true; continue; }   // a sentinel shot is not blocked
        if (in_list(tg.bld, t)) return true;
        if (g.env[t] == WALL) return false;
    }
    return false;
}
static void builder_opts(const Game& g, int a, BuilderOpts& o, const int8_t* plan4 = nullptr, int own_launchers = -1, const Targets* tg = nullptr) {
    const Entity& e = g.ents[a];
    for (int i = 0; i < 4; i++) {
        o.tx[i] = e.x + DX[CARD4[i]]; o.ty[i] = e.y + DY[CARD4[i]];
        bool in = g.inb(o.tx[i], o.ty[i]);
        // hidden enemy mine: no building visible to us, yet an attack there is legal
        // (the official can_fire only needs a building on the tile; checked in fcode).
        // Probed before acting (can_attack needs a builder that can still act and 2 Ti).
        const bool can_hit = in && g.can_attack(a, o.tx[i], o.ty[i]) == OK;
        const bool hidden_mine = can_hit && g.tile_building(a, o.tx[i], o.ty[i]) < 0;
        o.move[i] = g.can_move(a, CARD4[i]) == OK && !hidden_mine;     // never step on it
        // attack a hidden mine only where it blocks the conveyor plan (clearing every
        // mine cost the untrained network ~1500 Ti in a game); avoiding it is free
        const bool mine_blocks = hidden_mine && plan4 && plan4[i] != 0;
        o.attack[i] = can_hit && (mine_blocks || enemy_building_on(g, a, o.tx[i], o.ty[i]));
        o.heal[i] = in && g.can_heal(a, o.tx[i], o.ty[i]) == OK;
        o.destroy[i] = in && g.can_destroy(a, o.tx[i], o.ty[i]) == OK;
        for (int t = 0; t < 9; t++) {
            int type = BUILD_ORDER[t];
            if (!in) continue;
            if (type == LAUNCHER && own_launchers >= MAX_LAUNCHERS) continue;
            if (type == GUNNER) continue;                       // gunners removed
            if (type == SENTINEL && (!tg || tg->own_turrets >= MAX_SENTINELS)) continue;   // scripted, capped
            if (directional(type)) {
                const bool planned = plan4 && (type == CONVEYOR || type == SPLITTER);
                for (int d = 0; d < 8; d++)
                    o.build[t][i][d] = (!planned || plan4[i] == d + 1) &&
                                       (!tg || (type != GUNNER && type != SENTINEL) || aims(g, type, o.tx[i], o.ty[i], d + 1, *tg)) &&
                                       g.can_build(a, type, o.tx[i], o.ty[i], d + 1) == OK;
            } else {
                o.build[t][i][0] = g.can_build(a, type, o.tx[i], o.ty[i], C) == OK;
            }
        }
    }
}

struct LineOpts { uint8_t fire[5] = {0}, rot[8] = {0}; int fx[5], fy[5]; };
// Turrets: fire only where the shot hits a visible enemy (no wasted ammo, no
// friendly fire); a gunner may rotate only to a direction whose line, within
// reach (3 straight, 2 diagonal), first meets an enemy (walls and anything
// else block it). Identical in bots/econ/heads.py.
static void line_opts(const Game& g, int a, LineOpts& o) {
    const Entity& e = g.ents[a];
    for (int k = 0; k < 5; k++) {
        o.fx[k] = e.x + DX[e.dir] * (k + 1); o.fy[k] = e.y + DY[e.dir] * (k + 1);
        o.fire[k] = g.inb(o.fx[k], o.fy[k]) && enemy_hit(g, a, o.fx[k], o.fy[k]) && g.can_fire(a, o.fx[k], o.fy[k]) == OK;
    }
    if (e.type == GUNNER)
        for (int d = 0; d < 8; d++) {
            if (g.can_rotate(a, d + 1) != OK) continue;
            const int len = (d % 2 == 0) ? 3 : 2;          // Dir d + 1: N, E, S, W straight; NE, SE, SW, NW diagonal
            for (int k = 1; k <= len; k++) {
                const int x = e.x + DX[d + 1] * k, y = e.y + DY[d + 1] * k;
                if (!g.inb(x, y) || g.env[g.idx(x, y)] == WALL) break;
                if (enemy_hit(g, a, x, y)) { o.rot[d] = 1; break; }
                if (g.bot[g.idx(x, y)] >= 0 || g.tile_building(a, x, y) >= 0) break;   // own unit / building blocks
            }
        }
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
    // pit launcher: on the edge of a pit 1-2 wide it only throws our own builder
    // (the first one next to it) across the pit, to any legal tile there: the
    // network decides launch or wait and where to land. As launcher_opts in heads.py.
    if (g.pit_axis >= 0 && g.pit_w >= 1 && g.pit_w <= 2) {
        const int c = g.pit_axis == 0 ? e.x : e.y;
        const bool west = c == g.pit_at - 1, east = c == g.pit_at + g.pit_w;
        if (west || east) {
            bool taken = false;
            for (int b = 0; b < 8; b++) {
                const int u = g.inb(o.bx[b], o.by[b]) ? g.bot[g.idx(o.bx[b], o.by[b])] : -1;
                const bool mine = !taken && u >= 0 && g.ents[u].team == e.team;
                bool any_across = false;
                for (int k = 0; k < 120; k++) {
                    const int tc = g.pit_axis == 0 ? o.dx[k] : o.dy[k];
                    const bool across = west ? tc >= g.pit_at + g.pit_w : tc < g.pit_at;
                    o.ok[b][k] = mine && across && o.ok[b][k];
                    any_across = any_across || o.ok[b][k];
                }
                o.bot[b] = any_across;
                if (any_across) taken = true;
            }
        }
    }
}

static bool any(const uint8_t* m, int n) { for (int i = 0; i < n; i++) if (m[i]) return true; return false; }

template <class T>
static void plan4_impl(const T* sp, int8_t plan4[4]) {
    const int cs = fe::PATCH * fe::PATCH, c = fe::HALF;
    const int cell[4] = {(c - 1) * fe::PATCH + c, c * fe::PATCH + c + 1, (c + 1) * fe::PATCH + c, c * fe::PATCH + c - 1};  // N E S W
    for (int i = 0; i < 4; i++) plan4[i] = int8_t(sp[size_t(fe::C_PATH) * cs + cell[i]]);
}
void plan4_from_obs(const int32_t* sp, int8_t plan4[4]) { plan4_impl(sp, plan4); }
void plan4_from_obs(const float* sp, int8_t plan4[4]) { plan4_impl(sp, plan4); }
void plan4_from_obs(const uint8_t* sp, int8_t plan4[4]) { plan4_impl(sp, plan4); }

bool has_choice(const Game& g, int a, const int8_t* plan4, int own_launchers, const Targets* tg) {
    if (a < 0 || a >= int(g.ents.size()) || !g.ents[a].alive) return false;
    switch (g.ents[a].type) {
    case CORE: { CoreOpts o; core_opts(g, a, o, tg); return any(o.spawn + 1, 12) || any(o.conv + 1, 4); }
    case BUILDER: {
        BuilderOpts o; builder_opts(g, a, o, plan4, own_launchers, tg);
        return any(o.move, 4) || any(o.attack, 4) || any(o.heal, 4) || any(o.destroy, 4) ||
               any(&o.build[0][0][0], 9 * 4 * 8);
    }
    case GUNNER: case SENTINEL: { LineOpts o; line_opts(g, a, o); return any(o.fire, 5) || any(o.rot, 8); }
    case LAUNCHER: { LauncherOpts o; launcher_opts(g, a, o); return any(o.bot, 8); }
    default: return false;
    }
}

bool spike_action(const Game& g, int a, Candidate& out, bool first_turn) {
    const Entity& e = g.ents[a];
    if (e.type == SENTINEL) {
        // sentinel (scripted): fire only at a visible enemy core tile on its line
        LineOpts lo; line_opts(g, a, lo);
        for (int k = 0; k < 5; k++) {
            if (!lo.fire[k]) continue;
            const int b = g.tile_building(a, lo.fx[k], lo.fy[k]);
            if (b >= 0 && g.ents[b].type == CORE && g.ents[b].team != e.team) { out = {ActionKind::Fire, lo.fx[k], lo.fy[k]}; return true; }
        }
        return false;
    }
    if (e.type != SPIKE) return false;
    // spike: fire only when a visible enemy (builder or building) is on one of
    // the 8 neighbours (the shot then hits every enemy around, hidden mines
    // included); first_turn is no longer used (kept for the ABI)
    (void)first_turn;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const int x = e.x + dx, y = e.y + dy;
            if (!g.inb(x, y)) continue;
            const int u = g.bot[g.idx(x, y)], b = g.tile_building(a, x, y);
            const bool enemy = (u >= 0 && g.ents[u].team != e.team) || (b >= 0 && g.ents[b].team != e.team);
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
Decision decide(const Game& g, int a, const float* logits, const Chooser& choose, const int8_t* plan4, int own_launchers, const Targets* tg) {
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
        CoreOpts o; core_opts(g, a, o, tg);
        int s = run(HC_SPAWN, vec(o.spawn, 13));
        if (s > 0) d.actions.push_back({ActionKind::Spawn, o.sx[s], o.sy[s]});
        int c = run(HC_CONVERT, vec(o.conv, 5));
        if (c > 0) d.actions.push_back({ActionKind::Convert, o.amount[c]});
    } else if (p == P_BUILDER) {
        BuilderOpts o; builder_opts(g, a, o, plan4, own_launchers, tg);
        // crossable pit (1-2 wide) and no own launcher: build one on the pit's edge
        if (own_launchers == 0 && g.pit_axis >= 0 && g.pit_w >= 1 && g.pit_w <= 2) {
            const int lt = 5;                          // BUILD_ORDER index of LAUNCHER
            for (int i = 0; i < 4; i++) {
                const int c = g.pit_axis == 0 ? o.tx[i] : o.ty[i];
                if ((c == g.pit_at - 1 || c == g.pit_at + g.pit_w) && o.build[lt][i][0]) {
                    d.actions.push_back({ActionKind::Build, LAUNCHER, o.tx[i], o.ty[i], C});
                    return d;
                }
            }
        }
        // sentinel (scripted): a builder that can build one aimed at a known enemy
        // core tile builds it (aims() in builder_opts; at most MAX_SENTINELS); nothing recorded
        {
            const int st = 4;                          // BUILD_ORDER index of SENTINEL
            for (int i = 0; i < 4; i++)
                for (int dd = 0; dd < 8; dd++)
                    if (o.build[st][i][dd]) {
                        d.actions.push_back({ActionKind::Build, SENTINEL, o.tx[i], o.ty[i], dd + 1});
                        return d;
                    }
        }
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
