#include "obs.hpp"

#include <algorithm>
#include <tuple>
#include <cmath>

namespace fe {

void ObsMemory::init(const Game& g, int team_) {
    W = g.W;
    H = g.H;
    team = team_;
    size_t n = size_t(W) * H;
    terrain.assign(n, T_UNKNOWN);
    bld_id.assign(n, -1);
    bld_type.assign(n, -1);
    bld_team.assign(n, -1);
    bot_id.assign(n, -1);
    bot_team.assign(n, -1);
    seen_round.assign(n, -1);
    visit_round.assign(n, -1);
    bld_dir.assign(n, -1);
    bld_hp.assign(n, -1);
    bot_dir.assign(n, -1);
    bot_hp.assign(n, -1);
    attr_seen_round.assign(n, -1);
    known.clear(); pending.clear();
    sym_alive = 7; sym = -1;
    // a pit lies on the mirror axis (controller get_pit): x pit -> mirror x, y pit -> mirror y
    if (g_bus.use_pit && g.pit_axis >= 0) sym_alive = uint8_t(1u << (g.pit_axis == 0 ? 1 : 2));
}

// ---------------------------------------------------------------- report bus
// Identical in bots/econ/encoder.py (checked by engine/tools/obs_parity.py).
// A FACT is one tile: rep_pack(cat, tile, remove); `known` / `pending` hold facts.
// A REPORT (store word) is a fact, or for ore an anchor fact plus bits 11..25:
// the other cells of the 4x4 block right / below the anchor holding the same ore.
// Slots are not partitioned: a write in round r is read by every unit in round
// r + 1 (all read the same start-of-round snapshot), so overwriting it later
// loses nothing; a fresh fact that loses a same-round slot race stays pending
// and goes out again. Fresh reports: slot (id + round) % 16; rotation (idle
// units with (id + round) % 4 == 0): slot ((id + round) / 4) % 16.
static constexpr uint32_t REP_MARK = 1u << 31, REP_REMOVE = 1u << 30;
enum : int { REP_ORE_TI = 1, REP_ORE_VO, REP_CORE, REP_GUNNER, REP_SENTINEL, REP_LAUNCHER, REP_SPIKE,
             REP_OWN_LAUNCHER,            // 8: one of OUR launchers (for the launcher cap)
             REP_SYM };                   // 9: the map symmetry (tile field = sym); experiment knob
// experiment knobs (bus_stats); the defaults are the deployed behaviour
BusConfig g_bus;
static int rep_cat_of_type(int type) {
    switch (type) {
    case CORE: return REP_CORE; case GUNNER: return REP_GUNNER; case SENTINEL: return REP_SENTINEL;
    case LAUNCHER: return REP_LAUNCHER; case SPIKE: return REP_SPIKE; default: return 0;
    }
}
static int rep_type_of_cat(int cat) {
    switch (cat) {
    case REP_CORE: return CORE; case REP_GUNNER: return GUNNER; case REP_SENTINEL: return SENTINEL;
    case REP_LAUNCHER: return LAUNCHER; case REP_SPIKE: return SPIKE; default: return -1;
    }
}
static bool rep_is_ore(int cat) { return cat == REP_ORE_TI || cat == REP_ORE_VO; }
static uint32_t rep_pack(int cat, int t, bool remove) { return REP_MARK | (remove ? REP_REMOVE : 0u) | (uint32_t(cat) << 26) | uint32_t(t); }
static int rep_rank(uint32_t p) { int c = (p >> 26) & 15; return c == REP_CORE ? 0 : (c >= REP_GUNNER ? 1 : 2); }
int ObsMemory::own_count(int type) const {
    int n = 0;
    for (size_t t = 0; t < bld_id.size(); t++) n += bld_id[t] >= 0 && bld_team[t] == team && bld_type[t] == type;
    return n;
}
static void rep_note(std::set<uint32_t>& known, uint32_t f) { known.insert(f); known.erase(f ^ REP_REMOVE); }

int ObsMemory::mirror(int t, int s) const {
    const int x = t % W, y = t / W;
    return ((s == 1) ? y : H - 1 - y) * W + ((s == 2) ? x : W - 1 - x);
}

// A tile's terrain is known: rule out the symmetries it contradicts; once the
// symmetry is known, its mirror tile gets the same terrain.
void ObsMemory::learn_terrain(int t) {
    if (terrain[t] == T_UNKNOWN) return;
    for (int s = 0; s < 3; s++) {
        if (!(sym_alive >> s & 1)) continue;
        const int mt = mirror(t, s);
        if (terrain[mt] != T_UNKNOWN && terrain[mt] != terrain[t]) sym_alive &= uint8_t(~(1u << s));
    }
    if (sym >= 0) {
        const int mt = mirror(t, sym);
        if (terrain[mt] == T_UNKNOWN) terrain[mt] = terrain[t];
    }
}

// End of a turn's update: a symmetry that maps our core onto itself is out;
// when one symmetry is left it becomes THE symmetry (every known tile is
// mirrored once), and the enemy core is placed on the mirror of ours where that
// tile was never seen.
void ObsMemory::resolve_symmetry() {
    const int N = W * H;
    std::vector<int> own_core;
    for (int t = 0; t < N; t++) if (bld_id[t] >= 0 && bld_team[t] == team && bld_type[t] == CORE) own_core.push_back(t);
    for (int s = 0; s < 3; s++) {
        if (!(sym_alive >> s & 1)) continue;
        for (int t : own_core) {
            const int mt = mirror(t, s);
            if (bld_id[mt] >= 0 && bld_team[mt] == team && bld_type[mt] == CORE) { sym_alive &= uint8_t(~(1u << s)); break; }
        }
    }
    // a known enemy core tile (seen or reported) must be the mirror of one of ours
    if (g_bus.use_enemy_core && !own_core.empty() && sym < 0) {
        for (int t = 0; t < N; t++) {
            if (!(bld_id[t] >= 0 && bld_team[t] != team && bld_type[t] == CORE)) continue;
            for (int s = 0; s < 3; s++) {
                if (!(sym_alive >> s & 1)) continue;
                bool match = false;
                for (int o : own_core) if (mirror(o, s) == t) { match = true; break; }
                if (!match) sym_alive &= uint8_t(~(1u << s));
            }
        }
    }
    if (sym < 0 && (sym_alive == 1 || sym_alive == 2 || sym_alive == 4)) {
        sym = sym_alive == 1 ? 0 : (sym_alive == 2 ? 1 : 2);
        for (int t = 0; t < N; t++) {
            const int mt = mirror(t, sym);
            if (terrain[t] != T_UNKNOWN && terrain[mt] == T_UNKNOWN) terrain[mt] = terrain[t];
        }
    }
    if (sym >= 0) {
        const int enemy = 1 - team;
        for (int t : own_core) {
            const int mt = mirror(t, sym);
            if (bld_id[mt] < 0 && seen_round[mt] < 0) {
                bld_id[mt] = enemy + 1; bld_type[mt] = int8_t(CORE); bld_team[mt] = int8_t(enemy);
                bld_dir[mt] = -1; bld_hp[mt] = 100;
                if (terrain[mt] == T_UNKNOWN) terrain[mt] = T_EMPTY;
            }
        }
    }
}

void ObsMemory::merge_store(const int64_t* store) {
    snap_sym = false; snap_mask = 7;
    for (int i = 0; i < 16; i++) {
        const int64_t v = store[i];
        if (v < 0 || v > 0xFFFFFFFFll || !(uint64_t(v) & REP_MARK)) continue;
        const uint32_t p = uint32_t(v);
        const int cat = (p >> 26) & 15, t = p & 0x7FF;
        const uint32_t bits = (p >> 11) & 0x7FFF;
        const bool remove = p & REP_REMOVE;
        if (cat == REP_SYM) {
            // tile field: bits 0..2 = the sender's candidate set (1 << sym when it knows it)
            const uint8_t mask = uint8_t(t & 7);
            if (!remove && !bits && mask) {
                snap_mask &= mask;
                if (__builtin_popcount(mask) == 1) snap_sym = true;
                if (sym < 0 && (sym_alive & mask)) sym_alive &= mask;
            }
            continue;
        }
        if (cat < REP_ORE_TI || cat > REP_OWN_LAUNCHER || t >= W * H) continue;
        if (rep_is_ore(cat)) {
            if (remove) continue;
            const int x = t % W, y = t / W;
            for (int k = 0; k < 16; k++) {
                if (k > 0 && !(bits >> (k - 1) & 1)) continue;
                const int xx = x + k % 4, yy = y + k / 4;
                if (xx >= W || yy >= H) continue;
                const int tt = yy * W + xx;
                const uint32_t f = rep_pack(cat, tt, false);
                rep_note(known, f);
                pending.erase(f);
                if (terrain[tt] == T_UNKNOWN) terrain[tt] = cat == REP_ORE_TI ? T_ORE_TI : T_ORE_VO;
                learn_terrain(tt);
            }
            continue;
        }
        // non-ore reports carry the sender's map symmetry in bits 11..12 (0 unknown, else sym + 1)
        if (bits & ~3u) continue;
        if (const int ss = int(bits & 3u); ss) { snap_sym = true; snap_mask &= uint8_t(1u << (ss - 1)); if (sym < 0 && (sym_alive >> (ss - 1) & 1)) sym_alive = uint8_t(1u << (ss - 1)); }
        const uint32_t f = p & ~(0x7FFFu << 11);     // the fact itself
        rep_note(known, f);
        pending.erase(f);
        if (cat == REP_OWN_LAUNCHER) {
            if (remove) {
                if (bld_id[t] >= 0 && bld_team[t] == team && bld_type[t] == LAUNCHER) {
                    bld_id[t] = -1; bld_type[t] = -1; bld_team[t] = -1; bld_dir[t] = -1; bld_hp[t] = -1;
                }
            } else if (bld_id[t] < 0) {
                bld_id[t] = (1 << 29) + t; bld_type[t] = int8_t(LAUNCHER); bld_team[t] = int8_t(team);
                bld_dir[t] = -1; bld_hp[t] = 100;
                if (terrain[t] == T_UNKNOWN) { terrain[t] = T_EMPTY; learn_terrain(t); }
            }
            continue;
        }
        const int type = rep_type_of_cat(cat), enemy = 1 - team;
        if (remove) {
            if (bld_id[t] >= 0 && bld_team[t] == enemy && bld_type[t] == type) {
                bld_id[t] = -1; bld_type[t] = -1; bld_team[t] = -1; bld_dir[t] = -1; bld_hp[t] = -1;
            }
        } else if (bld_id[t] < 0) {
            // a reported enemy building: a core keeps its real id (team + 1), so
            // its 4 tiles count once; a turret gets a per-tile id. Direction and
            // HP unknown; attribute age untouched (not seen by this unit).
            bld_id[t] = type == CORE ? enemy + 1 : (1 << 30) + t;
            bld_type[t] = int8_t(type); bld_team[t] = int8_t(enemy);
            bld_dir[t] = -1; bld_hp[t] = 100;
            if (terrain[t] == T_UNKNOWN) { terrain[t] = T_EMPTY; learn_terrain(t); }
        }
    }
}

// The ore report for anchor fact f: f plus the 4x4 block's cells holding the same ore.
uint32_t ObsMemory::ore_report(uint32_t f) const {
    const int cat = (f >> 26) & 15, t = f & 0x7FF, x = t % W, y = t / W;
    const uint8_t ore = cat == REP_ORE_TI ? T_ORE_TI : T_ORE_VO;
    uint32_t bits = 0;
    for (int k = 1; k < 16; k++) {
        const int xx = x + k % 4, yy = y + k / 4;
        if (xx < W && yy < H && terrain[yy * W + xx] == ore) bits |= 1u << (k - 1);
    }
    return f | (bits << 11);
}

// The report to post this turn: the most important pending fact (core, turrets,
// ore; lowest tile) into slot (id + round) % 16; with nothing pending, on rounds
// where (id + round) % 4 == 0, a known fact still true in memory, cycling
// through them in order, into slot ((id + round) / 4) % 16.
bool ObsMemory::next_report(uint64_t unit_id, int round, int& slot, uint32_t& report) const {
    const uint64_t k = unit_id + uint64_t(round);
    if (!pending.empty()) {
        uint32_t best = 0;
        const auto key = [](uint32_t q) { return std::make_tuple(rep_rank(q), int(q & 0x7FF), int((q >> 30) & 1), int((q >> 26) & 15)); };
        for (uint32_t f : pending) if (!best || key(f) < key(best)) best = f;
        const int cat = (best >> 26) & 15;
        report = rep_is_ore(cat) ? ore_report(best) : (best | sym_bits());
        slot = int(k % 16);
        return true;
    }
    if (g_bus.sym_report && k % uint64_t(g_bus.sym_div) == 0) {
        // post our candidate set when it says more than last round's store did:
        // the symmetry itself, or (mode 2) eliminations the team has not shared
        const uint8_t mine = sym >= 0 ? uint8_t(1u << sym) : sym_alive;
        const bool news = sym >= 0 ? !snap_sym : (g_bus.sym_report >= 2 && mine && (snap_mask & ~mine & 7));
        if (news) {
            report = REP_MARK | (uint32_t(REP_SYM) << 26) | uint32_t(mine);
            slot = int(k % 16);
            return true;
        }
    }
    if (k % uint64_t(g_bus.rot_div) != 0) return false;
    std::vector<uint32_t> live;
    const int enemy = 1 - team;
    std::vector<uint8_t> covered;               // dedupe: ore tiles inside an earlier anchor's 4x4 block
    if (g_bus.rot_dedupe) covered.assign(size_t(W) * H * 3, 0);
    for (uint32_t f : known) {                  // std::set: ascending (by category, then tile)
        if (f & REP_REMOVE) continue;
        const int cat = (f >> 26) & 15, t = f & 0x7FF;
        if (rep_is_ore(cat)) {
            const uint8_t ore = cat == REP_ORE_TI ? T_ORE_TI : T_ORE_VO;
            if (terrain[t] != ore) continue;
            if (g_bus.rot_dedupe) {
                if (covered[size_t(cat) * W * H + t]) continue;
                const int x = t % W, y = t / W;
                for (int dy = 0; dy < 4; dy++) for (int dx = 0; dx < 4; dx++)
                    if (x + dx < W && y + dy < H) covered[size_t(cat) * W * H + (y + dy) * W + x + dx] = 1;
            }
            live.push_back(f);
        }
        else if (cat == REP_OWN_LAUNCHER) { if (bld_id[t] >= 0 && bld_team[t] == team && bld_type[t] == LAUNCHER) live.push_back(f); }
        else if (bld_id[t] >= 0 && bld_team[t] == enemy && bld_type[t] == rep_type_of_cat(cat)) live.push_back(f);
    }
    if (live.empty()) return false;
    const uint32_t f = live[(k / uint64_t(g_bus.rot_div)) % live.size()];
    report = rep_is_ore((f >> 26) & 15) ? ore_report(f) : (f | sym_bits());
    slot = int((k / uint64_t(g_bus.rot_div)) % 16);
    return true;
}

void ObsMemory::update(const Game& g, int actor) {
    std::vector<int> tiles;
    g.nearby_tiles(actor, -1, tiles);
    auto fact = [&](uint32_t p) { pending.erase(p ^ REP_REMOVE); if (!known.count(p)) pending.insert(p); };
    for (int t : tiles) {
        int x = t % g.W, y = t / g.W;
        const int old_id = bld_id[t], old_type = bld_type[t], old_team = bld_team[t];
        uint8_t e = g.env[t];
        terrain[t] = e == WALL ? T_WALL
                   : e == ORE_TI ? T_ORE_TI
                   : e == ORE_VO ? T_ORE_VO
                   : e == VOID ? T_VOID
                   : T_EMPTY;
        int b = g.tile_building(actor, x, y);   // hides enemy mines
        if (b >= 0) {
            bld_id[t] = g.ents[b].id;
            bld_type[t] = int8_t(g.ents[b].type);
            bld_team[t] = int8_t(g.ents[b].team);
            bld_dir[t] = g.ents[b].dir;
            bld_hp[t] = g.ents[b].max_hp ? (100 * g.ents[b].hp) / g.ents[b].max_hp : 0;
            attr_seen_round[t] = g.round;
        } else {
            bld_id[t] = -1;
            bld_type[t] = -1;
            bld_team[t] = -1;
            bld_dir[t] = -1;
            bld_hp[t] = -1;
        }
        int u = g.bot[t];
        if (u >= 0) {
            int32_t observed_id = g.ents[u].id;
            if (bot_id[t] != observed_id) visit_round[t] = g.round;
            bot_id[t] = observed_id;
            bot_team[t] = int8_t(g.ents[u].team);
            bot_dir[t] = g.ents[u].dir;
            bot_hp[t] = g.ents[u].max_hp ? (100 * g.ents[u].hp) / g.ents[u].max_hp : 0;
            // This is deliberately last *observed* visit. Under fog, production
            // cannot know about a later unseen visit; do not read hidden state.
            attr_seen_round[t] = g.round;
        } else {
            bot_id[t] = -1;
            bot_team[t] = -1;
            bot_dir[t] = -1;
            bot_hp[t] = -1;
        }
        seen_round[t] = g.round;
        learn_terrain(t);
        // reportable facts seen now
        if (terrain[t] == T_ORE_TI || terrain[t] == T_ORE_VO) fact(rep_pack(terrain[t] == T_ORE_TI ? REP_ORE_TI : REP_ORE_VO, t, false));
        const int cat_now = (bld_id[t] >= 0 && bld_team[t] != team) ? rep_cat_of_type(bld_type[t]) : 0;
        if (cat_now) fact(rep_pack(cat_now, t, false));
        const int cat_old = (old_id >= 0 && old_team != team) ? rep_cat_of_type(old_type) : 0;
        if (cat_old && cat_old != cat_now) fact(rep_pack(cat_old, t, true));
        // our own launchers (the launcher cap)
        const bool own_l_now = bld_id[t] >= 0 && bld_team[t] == team && bld_type[t] == LAUNCHER;
        const bool own_l_old = old_id >= 0 && old_team == team && old_type == LAUNCHER;
        if (own_l_now) fact(rep_pack(REP_OWN_LAUNCHER, t, false));
        else if (own_l_old) fact(rep_pack(REP_OWN_LAUNCHER, t, true));
    }
    resolve_symmetry();
}

// Walk `maxlen` tiles outward from (x0,y0) along (dx,dy), one rounded step at a
// time. The Python encoder must produce the identical path.
static void ray_path(int x0, int y0, int dx, int dy, int maxlen,
                     std::vector<std::pair<int, int>>& out) {
    out.clear();
    int L = std::max(std::abs(dx), std::abs(dy));
    if (L == 0) return;
    double sx = double(dx) / L, sy = double(dy) / L;
    double px = x0, py = y0;
    int lx = x0, ly = y0;
    for (int k = 1; k <= maxlen; k++) {
        px += sx;
        py += sy;
        int ix = int(std::floor(px + 0.5)), iy = int(std::floor(py + 0.5));
        if (ix == lx && iy == ly) continue;
        out.push_back({ix, iy});
        lx = ix;
        ly = iy;
    }
}

static bool has(const std::vector<int>& v, int x) {
    for (int e : v) if (e == x) return true;
    return false;
}

// Conveyor planner (identical in bots/econ/encoder.py, checked by obs_parity):
// only what this unit remembers. A BFS from the remembered own-core tiles
// (distance 0) over known EMPTY tiles that hold no building, or the team's own
// conveyor / splitter; neighbours in N, E, S, W order. Then for every known ore
// tile (row-major) that is free or holds an own harvester: from its neighbour
// closest to the core (first in N E S W order on ties) walk down the distance
// to the core; each cell on the way gets the direction to its next cell (the
// first path to reach a cell keeps it). An ore tile next to the core needs no
// conveyor. Unknown tiles, walls, void, ore and other buildings are not paths.
void ObsMemory::plan_paths(std::vector<int8_t>& plan) const {
    const int N = W * H;
    plan.assign(N, 0);
    static const int NX[4] = {0, 1, 0, -1}, NY[4] = {-1, 0, 1, 0};
    static const int8_t NDIR[4] = {1, 3, 5, 7};          // Dir codes of N, E, S, W (a local N below is the tile count)
    std::vector<int> dist(N, -1), q;
    q.reserve(N);
    for (int t = 0; t < N; t++)
        if (bld_id[t] >= 0 && bld_team[t] == team && bld_type[t] == CORE) { dist[t] = 0; q.push_back(t); }
    auto passable = [&](int t) {
        if (terrain[t] != T_EMPTY) return false;
        if (bld_id[t] < 0) return true;
        return bld_team[t] == team && (bld_type[t] == CONVEYOR || bld_type[t] == SPLITTER);
    };
    for (size_t h = 0; h < q.size(); h++) {
        const int t = q[h], x = t % W, y = t / W;
        for (int k = 0; k < 4; k++) {
            const int nx = x + NX[k], ny = y + NY[k];
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            const int nt = ny * W + nx;
            if (dist[nt] >= 0 || !passable(nt)) continue;
            dist[nt] = dist[t] + 1;
            q.push_back(nt);
        }
    }
    for (int s = 0; s < N; s++) {
        if (terrain[s] != T_ORE_TI && terrain[s] != T_ORE_VO) continue;
        if (bld_id[s] >= 0 && !(bld_team[s] == team && bld_type[s] == HARVESTER)) continue;
        const int sx = s % W, sy = s / W;
        int best = -1, bd = 1 << 30;
        for (int k = 0; k < 4; k++) {
            const int nx = sx + NX[k], ny = sy + NY[k];
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            const int nt = ny * W + nx;
            if (dist[nt] >= 0 && dist[nt] < bd) { best = nt; bd = dist[nt]; }
        }
        if (best < 0 || bd == 0) continue;
        int c = best;
        while (dist[c] > 0) {
            const int x = c % W, y = c / W;
            int nxt = -1, kk = -1;
            for (int k = 0; k < 4; k++) {
                const int nx = x + NX[k], ny = y + NY[k];
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                const int nt = ny * W + nx;
                if (dist[nt] == dist[c] - 1) { nxt = nt; kk = k; break; }
            }
            if (plan[c] == 0) plan[c] = NDIR[kk];
            c = nxt;
        }
    }
}

void encode_actor(const Game& g, const ObsMemory& m, int actor,
                  std::vector<int32_t>& ch, std::vector<int32_t>& scalars) {
    ch.assign(ACTOR_SPATIAL, 0);
    scalars.assign(ACTOR_SCALARS, 0);
    const Entity& e = g.ents[actor];
    const int team = e.team;
    const int cx = e.x, cy = e.y;
    const int cs = PATCH * PATCH;

    for (int dy = -HALF; dy <= HALF; dy++) {
        for (int dx = -HALF; dx <= HALF; dx++) {
            int x = cx + dx, y = cy + dy;
            if (!g.inb(x, y)) continue;       // outside the map: all channels zero
            int t = y * g.W + x;
            int cell = (dy + HALF) * PATCH + (dx + HALF);
            int ter = m.terrain[t];
            int tc = ter == T_UNKNOWN ? CH_TERRAIN - 1 : ter - 1;
            ch[(size_t)(C_TERRAIN + tc) * cs + cell] = 1;
            if (m.bld_id[t] >= 0) {
                int bt = m.bld_type[t];
                if (m.bld_team[t] == team) {
                    ch[(size_t)(C_OWN + bt) * cs + cell] = 1;
                } else if (bt != MINE) {
                    ch[(size_t)(C_ENEMY + bt) * cs + cell] = 1;
                }
            }
            if (m.bot_id[t] >= 0) {
                bool own = m.bot_team[t] == team;
                ch[(size_t)((own ? C_OWN : C_ENEMY) + BUILDER) * cs + cell] = 1;
                ch[(size_t)(C_OCC + (own ? 0 : 1)) * cs + cell] = 1;
            }
            int st = m.seen_round[t] < 0 ? 255 : std::min(255, g.round - m.seen_round[t]);
            ch[(size_t)C_STALE * cs + cell] = st;
            int visit = m.visit_round[t] < 0 ? 255 : std::min(255, g.round - m.visit_round[t]);
            ch[(size_t)C_VISIT * cs + cell] = visit;
            if (m.bld_id[t] >= 0) {
                int dir = m.bld_dir[t];
                if (dir >= 0 && dir < CH_DIR) ch[(size_t)(C_DIR + dir) * cs + cell] = 1;
                ch[(size_t)C_HP * cs + cell] = m.bld_hp[t];
                int age = m.attr_seen_round[t] < 0 ? 255 : std::min(255, g.round - m.attr_seen_round[t]);
                ch[(size_t)C_ATTR_STALE * cs + cell] = age;
            }
            if (m.bot_id[t] >= 0) {
                ch[(size_t)(C_HP + 1) * cs + cell] = m.bot_hp[t];
                int age = m.attr_seen_round[t] < 0 ? 255 : std::min(255, g.round - m.attr_seen_round[t]);
                ch[(size_t)(C_ATTR_STALE + 1) * cs + cell] = age;
            }
        }
    }

    // Remembered turret geometry. This is intentionally not an exact
    // can_fire answer: production cannot inspect hidden ammo/cooldowns or mines.
    // Channels 0/1 = possible own/enemy reach; 2/3 = confirmed own/enemy geometry.
    bool friendly_reaches_enemy_core = false, friendly_confirmed = false;
    bool enemy_reaches_friendly_core = false, enemy_confirmed = false;
    std::vector<int> turret_ids;
    for (int t = 0; t < m.W * m.H; t++) {
        int id = m.bld_id[t], type = m.bld_type[t], dir = m.bld_dir[t];
        if (id < 0 || dir <= C || dir > NW) continue;
        if (type != GUNNER && type != SENTINEL && type != SPIKE) continue;
        bool duplicate = false;
        for (int v : turret_ids) if (v == id) { duplicate = true; break; }
        if (duplicate) continue;
        turret_ids.push_back(id);
        int tx = t % m.W, ty = t / m.W;
        std::vector<std::pair<int, int>> targets;
        g.attack_tiles(type, tx, ty, dir, targets);
        for (const auto& target : targets) {
            int x = target.first, y = target.second;
            if (!g.inb(x, y)) continue;
            int dx = x - cx, dy = y - cy;
            bool friendly = m.bld_team[t] == team;
            bool confirmed = m.attr_seen_round[t] == g.round && m.seen_round[t] == g.round;
            bool blocked = false;
            if (type == GUNNER || type == SENTINEL) {
                int steps = std::max(std::abs(x - tx), std::abs(y - ty));
                int sx = (x == tx) ? 0 : (x > tx ? 1 : -1);
                int sy = (y == ty) ? 0 : (y > ty ? 1 : -1);
                for (int k = 1; k < steps; k++) {
                    int ix = tx + sx * k, iy = ty + sy * k;
                    int q = iy * m.W + ix;
                    bool fresh = m.seen_round[q] == g.round;
                    if (type == GUNNER && !fresh) confirmed = false;
                    if (type == GUNNER) {
                        bool static_block = m.terrain[q] == T_WALL || m.terrain[q] == T_VOID;
                        bool dynamic_block = fresh && (m.bld_id[q] >= 0 || m.bot_id[q] >= 0);
                        if (static_block || dynamic_block) {
                            blocked = true;
                            break;
                        }
                    }
                }
            }
            if (!blocked) {
                int target_t = y * m.W + x;
                if (m.bld_id[target_t] >= 0 && m.bld_type[target_t] == CORE) {
                    bool target_friendly = m.bld_team[target_t] == team;
                    bool core_confirmed = confirmed && m.seen_round[target_t] == g.round;
                    if (friendly && !target_friendly) {
                        friendly_reaches_enemy_core = true;
                        if (core_confirmed) friendly_confirmed = true;
                    } else if (!friendly && target_friendly) {
                        enemy_reaches_friendly_core = true;
                        if (core_confirmed) enemy_confirmed = true;
                    }
                }
                if (std::abs(dx) <= HALF && std::abs(dy) <= HALF) {
                    int cell = (dy + HALF) * PATCH + dx + HALF;
                    ch[(size_t)(C_THREAT + (friendly ? 0 : 1)) * cs + cell] = 1;
                    if (confirmed) ch[(size_t)(C_THREAT + (friendly ? 2 : 3)) * cs + cell] = 1;
                }
            }
        }
    }

    // edge rays: one per boundary cell of the patch, along the radial direction
    for (int dy = -HALF; dy <= HALF; dy++) {
        for (int dx = -HALF; dx <= HALF; dx++) {
            if (std::abs(dx) != HALF && std::abs(dy) != HALF) continue;
            int32_t v[CH_RAY] = {0};
            int bx = cx + dx, by = cy + dy;
            if (g.inb(bx, by)) {
                std::vector<std::pair<int, int>> path;
                ray_path(bx, by, dx, dy, RAY_LEN, path);
                std::vector<int> seen_b, seen_u;
                int step = 0;
                for (auto& p : path) {
                    if (!g.inb(p.first, p.second)) break;   // stop at the map edge
                    step++;
                    int t = p.second * g.W + p.first;
                    int ter = m.terrain[t];
                    if (!v[0] && (ter == T_WALL || ter == T_VOID)) v[0] = step;
                    if (!v[1] && ter == T_UNKNOWN) v[1] = step;
                    if (ter == T_ORE_TI) v[2]++;
                    if (ter == T_ORE_VO) v[3]++;
                    if (ter == T_UNKNOWN) v[10]++;
                    int bid = m.bld_id[t];
                    if (bid >= 0 && !has(seen_b, bid)) {
                        seen_b.push_back(bid);
                        bool own = m.bld_team[t] == team;
                        if (own) v[4]++; else v[5]++;
                        int bt = m.bld_type[t];
                        if (is_unit(bt)) { if (own) v[6]++; else v[7]++; }
                        if (bt == HARVESTER) { if (own) v[8]++; else v[9]++; }
                    }
                    int uid = m.bot_id[t];
                    if (uid >= 0 && !has(seen_u, uid)) {
                        seen_u.push_back(uid);
                        if (m.bot_team[t] == team) v[6]++; else v[7]++;
                    }
                }
            }
            int cell = (dy + HALF) * PATCH + (dx + HALF);
            for (int k = 0; k < CH_RAY; k++) ch[(size_t)(C_RAY + k) * cs + cell] = v[k];
        }
    }

    // planned conveyor paths (builders only)
    if (e.type == BUILDER) {
        std::vector<int8_t> plan;
        m.plan_paths(plan);
        for (int dy = -HALF; dy <= HALF; dy++)
            for (int dx = -HALF; dx <= HALF; dx++) {
                const int x = cx + dx, y = cy + dy;
                if (!g.inb(x, y)) continue;
                ch[(size_t)C_PATH * cs + (dy + HALF) * PATCH + (dx + HALF)] = plan[y * g.W + x];
            }
    }

    // scalars
    for (int k = 0; k < 6; k++) scalars[k] = (e.type == k) ? 1 : 0;
    scalars[6] = e.max_hp ? (100 * e.hp) / e.max_hp : 0;
    scalars[7] = e.acd;
    scalars[8] = e.mcd;
    scalars[9] = g.round;
    const Team& me = g.teams[team];
    scalars[10] = me.ti;
    scalars[11] = me.vo;
    scalars[12] = me.ammo;
    scalars[13] = me.scale;  // get_scale_percent()
    scalars[14] = me.units;
    scalars[15] = 0;  // opponent economy is not exposed to the deployed controller
    scalars[16] = 0;
    scalars[17] = 0;
    scalars[18] = 0;
    scalars[19] = 0;
    for (int i = 0; i < 16; i++) scalars[20 + i] = int32_t(me.store[i]);
    scalars[36] = friendly_reaches_enemy_core;
    scalars[37] = friendly_confirmed;
    scalars[38] = enemy_reaches_friendly_core;
    scalars[39] = enemy_confirmed;
    // current prices: conveyor, splitter, harvester, barrier, gunner, sentinel,
    // launcher, spike, mine, builder bot
    static const int COST_TYPES[10] = {CONVEYOR, SPLITTER, HARVESTER, BARRIER, GUNNER, SENTINEL, LAUNCHER, SPIKE, MINE, BUILDER};
    for (int i = 0; i < 10; i++) scalars[ACTOR_COSTS + i] = g.cost(team, COST_TYPES[i]);
}

// ---------------------------------------------------------------- critic
void encode_critic(const Game& g, std::vector<int32_t>& ch, std::vector<int32_t>& scalars) {
    ch.assign(CRITIC_SPATIAL, 0);
    scalars.assign(CRITIC_SCALARS, 0);
    const int cs = CRITIC_PAD * CRITIC_PAD;
    for (int y = 0; y < g.H && y < CRITIC_PAD; y++) {
        for (int x = 0; x < g.W && x < CRITIC_PAD; x++) {
            int t = y * g.W + x;
            int cell = y * CRITIC_PAD + x;
            int e = g.env[t];
            int tc = e == EMPTY ? 0 : e == WALL ? 1 : e == ORE_TI ? 2 : e == ORE_VO ? 3 : 4;
            ch[(size_t)(0 + tc) * cs + cell] = 1;
            int b = g.bld[t];
            if (b >= 0) {
                int bt = g.ents[b].type, tm = g.ents[b].team;
                ch[(size_t)(CC_TERRAIN + (tm == 0 ? 0 : CC_OWN) + bt) * cs + cell] = 1;
            }
            int u = g.bot[t];
            if (u >= 0) {
                int tm = g.ents[u].team;
                ch[(size_t)(CC_TERRAIN + (tm == 0 ? 0 : CC_OWN) + BUILDER) * cs + cell] = 1;
                ch[(size_t)(CC_TERRAIN + CC_OWN + CC_ENEMY + tm) * cs + cell] = 1;
            }
        }
    }
    scalars[0] = g.round;
    for (int t = 0; t < 2; t++) {
        const Team& m = g.teams[t];
        int o = 1 + 5 * t;
        scalars[o] = m.ti;
        scalars[o + 1] = m.vo;
        scalars[o + 2] = m.ammo;
        scalars[o + 3] = m.scale;
        scalars[o + 4] = m.units;
    }
    int base = 11;
    for (auto& ent : g.ents) {
        if (!ent.alive) continue;
        scalars[base + ent.team * NTYPES + ent.type]++;
    }
    const int e = base + 2 * NTYPES;
    scalars[e + 0] = std::max(0, g.max_turns - g.round);                  // rounds left
    scalars[e + 1] = g.teams[0].ti_collected;
    scalars[e + 2] = g.teams[1].ti_collected;
    scalars[e + 3] = g.teams[0].vo_collected;
    scalars[e + 4] = g.teams[1].vo_collected;
    scalars[e + 5] = g.teams[0].ti_collected - g.teams[1].ti_collected;   // first tie-break
    scalars[e + 6] = g.teams[0].ti - g.teams[1].ti;                        // third tie-break
}

}  // namespace fe
