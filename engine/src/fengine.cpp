#include "fengine.hpp"

#include <algorithm>
#include <random>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

namespace fe {

const int DX[9] = {0, 0, 1, 1, 1, 0, -1, -1, -1};
const int DY[9] = {0, -1, -1, 0, 1, 1, 1, 0, -1};
//                          BLD CORE GUN SEN LAU SPK CONV SPL HARV BAR MINE
const int BASE_COST[NTYPES] = {30, 0, 20, 30, 20, 20, 3, 6, 20, 3, 6};
const int SCALE_INC[NTYPES] = {20, 0, 20, 20, 10, 10, 1, 1, 5, 1, 1};
const int MAX_HP[NTYPES] = {40, 500, 25, 40, 30, 40, 20, 20, 30, 30, 5};
const int VISION[NTYPES] = {20, 36, 13, 32, 26, 8, 0, 0, 0, 0, 0};

bool is_unit(int t) { return t <= SPIKE; }
bool is_walkable(int t) { return t == CONVEYOR || t == SPLITTER || t == MINE; }

static int card_index(int dx, int dy) {  // N E S W -> 0..3, else -1
    if (dx == 0 && dy == -1) return 0;
    if (dx == 1 && dy == 0) return 1;
    if (dx == 0 && dy == 1) return 2;
    if (dx == -1 && dy == 0) return 3;
    return -1;
}
static bool cardinal(int d) { return d == N || d == E || d == S || d == W; }

// ------------------------------------------------------------------ map loading
namespace {
struct PB {
    const uint8_t* p; const uint8_t* e;
    bool more() const { return p < e; }
    uint64_t var() { uint64_t r = 0; int s = 0; while (p < e) { uint8_t c = *p++; r |= uint64_t(c & 0x7F) << s; s += 7; if (!(c & 0x80)) break; } return r; }
    bool field(int& f, int& wt, uint64_t& v, PB& sub) {
        if (!more()) return false;
        uint64_t k = var(); f = int(k >> 3); wt = int(k & 7);
        if (wt == 0) v = var();
        else if (wt == 2) { uint64_t l = var(); sub = {p, p + l}; p += l; }
        else if (wt == 5) p += 4;
        else if (wt == 1) p += 8;
        return true;
    }
};
}  // namespace

bool Game::load_map(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), {});
    return load_map_bytes(b.data(), b.size());
}


bool Game::load_map_bytes(const uint8_t* data, size_t n) {
    PB pb{data, data + n};
    int f, wt; uint64_t v; PB sub{};
    std::vector<std::vector<uint8_t>> rows;
    pit_axis = -1;
    bool have_core[2] = {false, false};
    while (pb.field(f, wt, v, sub)) {
        if (f == 1 && wt == 0) W = int(v);
        else if (f == 2 && wt == 0) H = int(v);
        else if (f == 3 && wt == 2) {
            std::vector<uint8_t> row;
            int ff, ww; uint64_t vv; PB s2{};
            while (sub.field(ff, ww, vv, s2)) {
                if (ff != 1) continue;
                if (ww == 2) { while (s2.more()) row.push_back(uint8_t(s2.var())); }
                else row.push_back(uint8_t(vv));
            }
            rows.push_back(row);
        } else if (f == 4 && wt == 2) {
            int ff, ww; uint64_t vv; PB s2{}; int team = 0, x = 0, y = 0;
            while (sub.field(ff, ww, vv, s2)) {
                if (ff == 2) team = int(vv);
                if (ff == 3) { int a, b2; uint64_t c; PB s3{}; while (s2.field(a, b2, c, s3)) { if (a == 1) x = int(c); if (a == 2) y = int(c); } }
            }
            core_xy[team][0] = x; core_xy[team][1] = y; have_core[team] = true;
        } else if (f == 8 && wt == 2) {
            int ff, ww; uint64_t vv; PB s2{};
            pit_axis = 0; pit_at = 0; pit_w = 0;
            while (sub.field(ff, ww, vv, s2)) {
                if (ff == 1) pit_axis = int(vv);
                if (ff == 2) pit_at = int(vv);
                if (ff == 3) pit_w = int(vv);
            }
        }
    }
    if (W <= 0 || H <= 0 || !have_core[0] || !have_core[1]) return false;
    env.assign(size_t(W) * H, EMPTY);
    for (int y = 0; y < H && y < int(rows.size()); y++)
        for (int x = 0; x < W && x < int(rows[y].size()); x++) env[idx(x, y)] = rows[y][x];
    reset();
    return true;
}

static const uint64_t ID_SALT = 0x1D5EED00C0DEF00Dull;

void Game::reset(uint64_t seed_) {
    seed = seed_;
    rng = seed ? seed : 1;
    rng_dist.seed(seed);
    rng_ids.seed(seed ^ ID_SALT);
    edge_last.clear();
    used_ids.clear();
    round = 0; winner = -1; over = false; win_condition.clear();
    ents.clear(); last_moves.clear();
    bld.assign(size_t(W) * H, -1);
    bot.assign(size_t(W) * H, -1);
    teams = {};
    next_seq = 0; next_id = 1;
    for (int t = 0; t < 2; t++) {
        int c = create(CORE, t, core_xy[t][0], core_xy[t][1], C);
        teams[t].core = c;
    }
}

int Game::find(int id) const {
    for (size_t i = 0; i < ents.size(); i++) if (ents[i].id == id && ents[i].alive) return int(i);
    return -1;
}

int Game::cost(int team, int type) const { return BASE_COST[type] * teams[team].scale / 100; }

bool Game::passable(int x, int y) const {
    if (!inb(x, y)) return false;
    int e = env[idx(x, y)];
    if (e == WALL || e == VOID) return false;
    int b = bld[idx(x, y)];
    return b < 0 || is_walkable(ents[b].type);
}

int Game::create(int type, int team, int x, int y, int dir) {
    Entity e;
    e.id = type == CORE ? team + 1 : int(new_id());
    e.type = uint8_t(type); e.team = uint8_t(team); e.dir = uint8_t(dir);
    e.x = x; e.y = y; e.hp = e.max_hp = MAX_HP[type];
    e.seq = next_seq++;
    if (type == HARVESTER) e.res = env[idx(x, y)] == ORE_VO ? RVO : RTI;
    int i = int(ents.size());
    ents.push_back(e);
    if (type == CORE) {
        for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) bld[idx(x + dx, y + dy)] = i;
    } else if (type == BUILDER) {
        bot[idx(x, y)] = i;
    } else {
        bld[idx(x, y)] = i;
    }
    teams[team].scale += SCALE_INC[type];
    if (is_unit(type)) teams[team].units++;
    return i;
}

uint32_t Game::new_id() {
    for (;;) {
        uint32_t id = rng_ids.gen_range(3, 1000000000);
        if (used_ids.insert(id).second) return id;
    }
}

void Game::kill(int i) {
    Entity& e = ents[i];
    if (!e.alive) return;
    e.alive = 0;
    if (e.type == CORE) {
        for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) bld[idx(e.x + dx, e.y + dy)] = -1;
        if (winner < 0) { winner = 1 - e.team; win_condition = "core_destroyed"; }
    } else if (e.type == BUILDER) {
        if (bot[idx(e.x, e.y)] == i) bot[idx(e.x, e.y)] = -1;
    } else {
        if (bld[idx(e.x, e.y)] == i) bld[idx(e.x, e.y)] = -1;
    }
    teams[e.team].scale -= SCALE_INC[e.type];
    if (is_unit(e.type)) teams[e.team].units--;
}

void Game::damage(int i, int amount) {
    Entity& e = ents[i];
    if (!e.alive) return;
    e.hp -= amount;
    if (e.hp <= 0) kill(i);
}

std::vector<int> Game::turn_order() const {
    std::vector<int> out;
    for (size_t i = 0; i < ents.size(); i++)
        if (ents[i].alive && is_unit(ents[i].type)) out.push_back(int(i));
    std::sort(out.begin(), out.end(), [&](int a, int b) { return ents[a].seq < ents[b].seq; });
    return out;
}

// ------------------------------------------------------------------ vision
int Game::dist2_from(int a, int x, int y) const {
    const Entity& e = ents[a];
    if (e.type == CORE) {  // from the nearest footprint tile
        int dx = x < e.x ? e.x - x : (x > e.x + 1 ? x - e.x - 1 : 0);
        int dy = y < e.y ? e.y - y : (y > e.y + 1 ? y - e.y - 1 : 0);
        return dx * dx + dy * dy;
    }
    int dx = x - e.x, dy = y - e.y;
    return dx * dx + dy * dy;
}

bool Game::in_vision(int a, int x, int y) const { return dist2_from(a, x, y) <= VISION[ents[a].type]; }

void Game::nearby_tiles(int a, int r2, std::vector<int>& out) const {
    out.clear();
    int v = VISION[ents[a].type];
    if (r2 < 0 || r2 > v) r2 = v;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (dist2_from(a, x, y) <= r2) out.push_back(idx(x, y));
}

bool Game::visible_building(int a, int b) const {
    return !(ents[b].type == MINE && ents[b].team != ents[a].team);
}

int Game::tile_building(int a, int x, int y) const {
    int b = bld[idx(x, y)];
    return (b >= 0 && visible_building(a, b)) ? b : -1;
}

bool Game::tile_empty(int a, int x, int y) const {
    int e = env[idx(x, y)];
    return e != WALL && e != VOID && tile_building(a, x, y) < 0;
}

bool Game::tile_passable(int a, int x, int y) const {
    int e = env[idx(x, y)];
    if (e == WALL || e == VOID || bot[idx(x, y)] >= 0) return false;
    int b = tile_building(a, x, y);
    return b < 0 || is_walkable(ents[b].type);
}

void Game::nearby_ids(int a, int r2, int kind, std::vector<int>& out) const {
    out.clear();
    std::vector<int> tiles;
    nearby_tiles(a, r2, tiles);
    std::vector<char> seen_core(2, 0);
    for (int t : tiles) {
        int x = t % W, y = t / W;
        int b = tile_building(a, x, y), u = bot[t];
        if (b >= 0) {
            bool core = ents[b].type == CORE;
            bool dup = core && seen_core[ents[b].team];
            if (core) seen_core[ents[b].team] = 1;
            if (!dup && (kind == 0 || kind == 1 || (kind == 2 && is_unit(ents[b].type)))) out.push_back(ents[b].id);
        }
        if (u >= 0 && (kind == 0 || kind == 2)) out.push_back(ents[u].id);
    }
}

// ------------------------------------------------------------------ helpers
static bool orth_adjacent(const Entity& e, int x, int y) {
    int dx = x - e.x, dy = y - e.y;
    return (dx == 0 && (dy == 1 || dy == -1)) || (dy == 0 && (dx == 1 || dx == -1));
}

static int core_dist2(const Entity& c, int x, int y) {  // squared distance to nearest core tile
    int dx = x < c.x ? c.x - x : (x > c.x + 1 ? x - c.x - 1 : 0);
    int dy = y < c.y ? c.y - y : (y > c.y + 1 ? y - c.y - 1 : 0);
    return dx * dx + dy * dy;
}

static bool crosses_pit(const Game& g, int x0, int y0, int x1, int y1) {
    if (g.pit_axis < 0 || g.pit_w <= 0) return false;
    int a = g.pit_axis == 0 ? x0 : y0, b = g.pit_axis == 0 ? x1 : y1;
    int lo = g.pit_at, hi = g.pit_at + g.pit_w - 1;
    return (a < lo && b > hi) || (b < lo && a > hi);
}

// arriving builder bot: enemy mine check. Returns false if the bot died.
static bool arrive(Game& g, int b) {
    Entity& e = g.ents[b];
    int m = g.bld[g.idx(e.x, e.y)];
    if (m >= 0 && g.ents[m].type == MINE && g.ents[m].team != e.team) {
        g.kill(m);
        g.damage(b, 40);
        return g.ents[b].alive;
    }
    return true;
}

static int builder_can_act(const Game& g, int a) {
    if (a < 0 || a >= int(g.ents.size()) || !g.ents[a].alive) return E_DEAD;
    if (g.over) return E_GAME_OVER;
    const Entity& e = g.ents[a];
    if (e.type != BUILDER) return E_WRONG_TYPE;
    if (e.acd > 0) return E_COOLDOWN;
    if (e.moved || e.acted) return E_ALREADY_ACTED;
    return OK;
}

// ------------------------------------------------------------------ actions
int Game::can_move(int a, int dir) const {
    if (a < 0 || a >= int(ents.size()) || !ents[a].alive) return E_DEAD;
    if (over) return E_GAME_OVER;
    const Entity& e = ents[a];
    if (e.type != BUILDER) return E_WRONG_TYPE;
    if (!cardinal(dir)) return E_BAD_DIR;
    if (e.mcd > 0) return E_COOLDOWN;
    if (e.acted || e.moved) return E_ALREADY_ACTED;
    int x = e.x + DX[dir], y = e.y + DY[dir];
    if (!passable(x, y)) return E_BLOCKED;
    if (bot[idx(x, y)] >= 0) return E_BLOCKED;
    return OK;
}

int Game::move(int a, int dir) {
    int r = can_move(a, dir);
    if (r) return r;
    Entity& e = ents[a];
    bot[idx(e.x, e.y)] = -1;
    e.x += DX[dir]; e.y += DY[dir];
    bot[idx(e.x, e.y)] = a;
    e.mcd = 1; e.moved = true;
    arrive(*this, a);
    return OK;
}

int Game::can_build(int a, int type, int x, int y, int dir) const {
    int r = builder_can_act(*this, a);
    if (r) return r;
    const Entity& e = ents[a];
    if (type == BUILDER || type == CORE || type < 0 || type >= NTYPES) return E_BAD_ARG;
    if (!inb(x, y) || !orth_adjacent(e, x, y)) return E_RANGE;
    int en = env[idx(x, y)];
    if (en == WALL || en == VOID) return E_BLOCKED;
    if (bld[idx(x, y)] >= 0) return E_BLOCKED;
    // official: only a conveyor or splitter may be built under a builder bot
    if (bot[idx(x, y)] >= 0 && type != CONVEYOR && type != SPLITTER) return E_BLOCKED;
    if (type == HARVESTER && en != ORE_TI && en != ORE_VO) return E_NOT_ORE;
    if (type == CONVEYOR || type == SPLITTER) { if (!cardinal(dir)) return E_BAD_DIR; }
    else if (type == GUNNER || type == SENTINEL) { if (dir == C || dir > NW) return E_BAD_DIR; }
    if (type == SPIKE) {
        const Entity& ec = ents[teams[1 - e.team].core];
        if (ec.alive && core_dist2(ec, x, y) <= 2) return E_SPIKE_COLLAR;
    }
    if (is_unit(type) && teams[e.team].units >= MAX_TEAM_UNITS) return E_UNIT_CAP;
    if (teams[e.team].ti < cost(e.team, type)) return E_FUNDS;
    return OK;
}

int Game::build(int a, int type, int x, int y, int dir) {
    int r = can_build(a, type, x, y, dir);
    if (r) return r;
    Entity& e = ents[a];
    teams[e.team].ti -= cost(e.team, type);
    int d = (type == CONVEYOR || type == SPLITTER || type == GUNNER || type == SENTINEL) ? dir : C;
    int team = e.team;
    ents[a].acd = 1; ents[a].acted = true;
    create(type, team, x, y, d);   // may reallocate ents
    // an enemy builder standing there steps on nothing; mines only trigger on arrival
    return OK;
}

int Game::can_attack(int a, int x, int y) const {
    int r = builder_can_act(*this, a);
    if (r) return r;
    const Entity& e = ents[a];
    if (!inb(x, y) || !orth_adjacent(e, x, y)) return E_RANGE;
    if (bld[idx(x, y)] < 0) return E_NO_TARGET;   // official can_fire: needs a building
    if (teams[e.team].ti < 2) return E_FUNDS;
    return OK;
}

int Game::attack(int a, int x, int y) {
    int r = can_attack(a, x, y);
    if (r) return r;
    Entity& e = ents[a];
    int b = bld[idx(x, y)];
    teams[e.team].ti -= 2;
    e.acd = 1; e.acted = true;
    damage(b, 2);
    return OK;
}

int Game::can_heal(int a, int x, int y) const {
    int r = builder_can_act(*this, a);
    if (r) return r;
    const Entity& e = ents[a];
    if (!inb(x, y) || !orth_adjacent(e, x, y)) return E_RANGE;
    int b = bld[idx(x, y)], u = bot[idx(x, y)];
    bool any = (b >= 0 && ents[b].team == e.team && ents[b].hp < ents[b].max_hp) ||
               (u >= 0 && ents[u].team == e.team && ents[u].hp < ents[u].max_hp);
    if (!any) return E_NO_TARGET;
    if (teams[e.team].ti < 1) return E_FUNDS;
    return OK;
}

int Game::heal(int a, int x, int y) {
    int r = can_heal(a, x, y);
    if (r) return r;
    Entity& e = ents[a];
    int b = bld[idx(x, y)], u = bot[idx(x, y)];
    teams[e.team].ti -= 1;
    e.acd = 1; e.acted = true;
    for (int t : {b, u})
        if (t >= 0 && ents[t].team == e.team) ents[t].hp = std::min(ents[t].max_hp, ents[t].hp + 4);
    return OK;
}

int Game::can_destroy(int a, int x, int y) const {
    if (a < 0 || a >= int(ents.size()) || !ents[a].alive) return E_DEAD;
    if (over) return E_GAME_OVER;
    const Entity& e = ents[a];
    if (e.type != BUILDER) return E_WRONG_TYPE;
    if (!inb(x, y) || !orth_adjacent(e, x, y)) return E_RANGE;
    int b = bld[idx(x, y)];
    if (b < 0 || ents[b].team != e.team || ents[b].type == CORE) return E_NO_TARGET;
    return OK;
}

int Game::destroy(int a, int x, int y) {
    int r = can_destroy(a, x, y);
    if (r) return r;
    kill(bld[idx(x, y)]);
    return OK;
}

int Game::self_destruct(int a) {
    if (a < 0 || !ents[a].alive) return E_DEAD;
    if (ents[a].type != BUILDER) return E_WRONG_TYPE;
    kill(a);
    return OK;
}

int Game::can_spawn(int c, int x, int y) const {
    if (c < 0 || !ents[c].alive) return E_DEAD;
    if (over) return E_GAME_OVER;
    const Entity& e = ents[c];
    if (e.type != CORE) return E_WRONG_TYPE;
    if (e.acd > 0) return E_COOLDOWN;
    if (!inb(x, y)) return E_RANGE;
    int d2 = core_dist2(e, x, y);
    if (d2 == 0 || d2 > 2) return E_RANGE;
    if (!passable(x, y) || bot[idx(x, y)] >= 0) return E_BLOCKED;
    if (teams[e.team].units >= MAX_TEAM_UNITS) return E_UNIT_CAP;
    if (teams[e.team].ti < cost(e.team, BUILDER)) return E_FUNDS;
    return OK;
}

int Game::spawn(int c, int x, int y) {
    int r = can_spawn(c, x, y);
    if (r) return r;
    int team = ents[c].team;
    teams[team].ti -= cost(team, BUILDER);
    ents[c].acd = 1;
    int b = create(BUILDER, team, x, y, C);
    arrive(*this, b);
    return OK;
}

int Game::can_convert(int c, int amount) const {
    if (c < 0 || c >= int(ents.size()) || !ents[c].alive) return E_DEAD;
    if (over) return E_GAME_OVER;
    if (ents[c].type != CORE) return E_WRONG_TYPE;
    const Team& t = teams[ents[c].team];
    if (t.converted) return E_ONCE_PER_TURN;
    if (amount <= 0 || amount > t.vo) return E_FUNDS;
    return OK;
}

int Game::convert_ammo(int c, int amount) {
    int r = can_convert(c, amount);
    if (r) return r;
    Team& t = teams[ents[c].team];
    t.vo -= amount; t.ammo += 2 * amount; t.converted = true;
    return OK;
}

void Game::attack_tiles(int type, int x, int y, int dir, std::vector<std::pair<int, int>>& out) const {
    out.clear();
    if (type == GUNNER || type == SENTINEL) {
        int len = type == GUNNER ? 3 : 5;
        if (dir == C || dir > NW) return;
        bool diag = !cardinal(dir);
        if (diag) len = type == GUNNER ? 2 : 4;  // measured: r^2 13 / 32 along diagonals
        for (int k = 1; k <= len; k++) out.push_back({x + DX[dir] * k, y + DY[dir] * k});
    } else if (type == SPIKE) {
        for (int d = 1; d <= 8; d++) out.push_back({x + DX[d], y + DY[d]});
    } else if (type == LAUNCHER) {
        for (int dy = -5; dy <= 5; dy++)
            for (int dx = -5; dx <= 5; dx++)
                if ((dx || dy) && dx * dx + dy * dy <= 26) out.push_back({x + dx, y + dy});
    }
}

int Game::can_fire(int a, int x, int y) const {
    if (a < 0 || a >= int(ents.size()) || !ents[a].alive) return E_DEAD;
    if (over) return E_GAME_OVER;
    const Entity& e = ents[a];
    if (e.type != GUNNER && e.type != SENTINEL && e.type != SPIKE) return E_WRONG_TYPE;
    if (e.acd > 0) return E_COOLDOWN;
    if (!inb(x, y)) return E_RANGE;
    std::vector<std::pair<int, int>> tiles;
    attack_tiles(e.type, e.x, e.y, e.dir, tiles);
    bool in = false;
    for (auto& t : tiles) if (t.first == x && t.second == y) { in = true; break; }
    if (!in) return E_RANGE;
    if (e.type == SPIKE) return OK;
    // measured: firing at an empty tile is legal (ammo is still spent)
    if (e.type == GUNNER) {  // blocked by anything in between
        for (auto& t : tiles) {
            if (t.first == x && t.second == y) break;
            if (!inb(t.first, t.second)) return E_BLOCKED;
            int i = idx(t.first, t.second);
            if (env[i] == WALL || bld[i] >= 0 || bot[i] >= 0) return E_BLOCKED;
        }
    }
    int need = e.type == GUNNER ? 4 : 10;
    if (teams[e.team].ammo < need) return E_NO_AMMO;
    return OK;
}

int Game::fire(int a, int x, int y) {
    int r = can_fire(a, x, y);
    if (r) return r;
    Entity& e = ents[a];
    int team = e.team;
    if (e.type == SPIKE) {
        e.acd = 4;
        std::vector<int> hit;
        for (int d = 1; d <= 8; d++) {
            int xx = e.x + DX[d], yy = e.y + DY[d];
            if (!inb(xx, yy)) continue;
            int b = bld[idx(xx, yy)], u = bot[idx(xx, yy)];
            // measured: one entity per tile, the builder bot before the building
            if (u >= 0 && ents[u].team != team) hit.push_back(u);
            else if (b >= 0 && ents[b].team != team) hit.push_back(b);
        }
        std::sort(hit.begin(), hit.end());
        hit.erase(std::unique(hit.begin(), hit.end()), hit.end());  // a core spans several tiles
        for (int h : hit) damage(h, 12);
        return OK;
    }
    int dmg = e.type == GUNNER ? 7 : 18;
    teams[team].ammo -= e.type == GUNNER ? 4 : 10;
    e.acd = e.type == GUNNER ? 1 : 2;
    int u = bot[idx(x, y)], b = bld[idx(x, y)];
    int hit = u >= 0 ? u : b;          // measured: a builder bot on the tile takes the shot
    if (hit >= 0) damage(hit, dmg);
    return OK;
}

int Game::can_rotate(int a, int dir) const {
    if (a < 0 || a >= int(ents.size()) || !ents[a].alive) return E_DEAD;
    if (over) return E_GAME_OVER;
    const Entity& e = ents[a];
    if (e.type != GUNNER) return E_WRONG_TYPE;
    if (e.acd > 0) return E_COOLDOWN;
    if (dir == C || dir > NW || dir == e.dir) return E_BAD_DIR;   // must be a different compass direction
    if (teams[e.team].ti < 10) return E_FUNDS;
    return OK;
}

int Game::rotate(int a, int dir) {
    int r = can_rotate(a, dir);
    if (r) return r;
    Entity& e = ents[a];
    teams[e.team].ti -= 10;
    e.dir = uint8_t(dir); e.acd = 1;
    return OK;
}

int Game::can_launch(int a, int bx, int by, int tx, int ty) const {
    if (a < 0 || a >= int(ents.size()) || !ents[a].alive) return E_DEAD;
    if (over) return E_GAME_OVER;
    const Entity& e = ents[a];
    if (e.type != LAUNCHER) return E_WRONG_TYPE;
    if (e.acd > 0) return E_COOLDOWN;
    if (!inb(bx, by) || std::max(std::abs(bx - e.x), std::abs(by - e.y)) != 1) return E_RANGE;
    if (bot[idx(bx, by)] < 0) return E_NO_TARGET;
    if (!inb(tx, ty)) return E_RANGE;
    int dx = tx - e.x, dy = ty - e.y;
    if (dx * dx + dy * dy > 26) return E_RANGE;
    if (bot[idx(tx, ty)] >= 0) return E_BLOCKED;
    int en = env[idx(tx, ty)];
    if (en == WALL || en == VOID) return E_BLOCKED;   // official: never aimed at a void tile
    int b = bld[idx(tx, ty)];
    if (b >= 0 && !is_walkable(ents[b].type)) return E_BLOCKED;
    return OK;
}

int Game::launch(int a, int bx, int by, int tx, int ty) {
    int r = can_launch(a, bx, by, tx, ty);
    if (r) return r;
    int u = bot[idx(bx, by)];
    ents[a].acd = 1;
    bot[idx(bx, by)] = -1;
    Entity& b = ents[u];
    bool falls = env[idx(tx, ty)] == VOID ||
                 (crosses_pit(*this, bx, by, tx, ty) && pit_w > 2);
    b.x = tx; b.y = ty;
    if (falls) { b.alive = 1; kill(u); return OK; }
    bot[idx(tx, ty)] = u;
    arrive(*this, u);
    return OK;
}

int Game::write_store(int team, int i, int64_t v) {
    if (i < 0 || i >= 16) return E_BAD_ARG;
    teams[team].store_next[i] = v;
    return OK;
}

void Game::resign(int team) {
    if (winner < 0) { winner = 1 - team; win_condition = "resigned"; }
    kill(teams[team].core);
}

// ------------------------------------------------------------------ distribution
// Reproduces the official engine's Game::distribute_resources (read from its
// binary and checked against traces of it):
//  1. Scan tiles row by row. Every building lists its output tiles (conveyor:
//     front; splitter: front/left/right; harvester: its 4 neighbours) that are
//     in bounds, on its side of a pit and hold a building; each such edge is
//     recorded under its target (a target -> sources list, kept in source scan
//     order) and counted for its source. A building with nothing to send (an
//     empty conveyor/splitter, a harvester on cooldown, anything else) is idle.
//  2. Targets in (x, y) order: if the target can take a stack now (a core, or
//     an empty conveyor/splitter), push every non-idle source's edge that it
//     accepts. Priority: i32::MAX if the source has one edge and the target one
//     source, else -(round this edge last moved a stack; 0 if never); plus a
//     uniform [0,1) draw from StdRng(seed).
//  3. Pop the highest edge. If the source still has its stack and the target
//     can still take one, move it (a harvester draws a new stack id and goes on
//     cooldown 4), record the round for the edge, mark the source idle, and push
//     the edges of sources waiting on the tile just vacated.
void Game::distribute() {
    last_moves.clear();
    const int NT = W * H;
    std::vector<char> idle(NT, 1);
    std::vector<int> outcount(NT, 0);
    std::vector<std::vector<int>> incoming(NT);
    auto same_side = [&](int x0, int y0, int x1, int y1) {
        if (pit_axis < 0 || pit_w <= 0) return true;
        int a = pit_axis == 0 ? x0 : y0, b = pit_axis == 0 ? x1 : y1;
        return (a < pit_at) == (b < pit_at);
    };
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int bi = bld[idx(x, y)];
            if (bi < 0) continue;
            const Entity& e = ents[bi];
            int outs[4]; int n = 0;
            if (e.type == CONVEYOR) {
                outs[n++] = e.dir;
                if (e.stored != RNONE) idle[idx(x, y)] = 0;
            } else if (e.type == SPLITTER) {
                outs[n++] = e.dir;
                outs[n++] = e.dir == N ? int(fe::W) : e.dir - 2;
                outs[n++] = e.dir == fe::W ? int(N) : e.dir + 2;
                if (e.stored != RNONE) idle[idx(x, y)] = 0;
            } else if (e.type == HARVESTER) {
                outs[n++] = N; outs[n++] = E; outs[n++] = S; outs[n++] = fe::W;
                if (e.hcd == 0) idle[idx(x, y)] = 0;
            }
            int cnt = 0;
            for (int k = 0; k < n; k++) {
                int tx = x + DX[outs[k]], ty = y + DY[outs[k]];
                if (!inb(tx, ty) || !same_side(x, y, tx, ty)) continue;
                if (bld[idx(tx, ty)] < 0) continue;
                incoming[idx(tx, ty)].push_back(idx(x, y));
                cnt++;
            }
            outcount[idx(x, y)] = cnt;
        }
    }
    auto can_take = [&](int t) {
        int b = bld[t];
        if (b < 0) return false;
        const Entity& d = ents[b];
        if (d.type == CORE) return true;
        return (d.type == CONVEYOR || d.type == SPLITTER) && d.stored == RNONE;
    };
    auto accepts = [&](int s, int t) {
        const Entity& d = ents[bld[t]];
        int sx = s % W, sy = s / W;
        if (d.type == CORE) return true;
        if (d.type == CONVEYOR) return !(d.x + DX[d.dir] == sx && d.y + DY[d.dir] == sy);
        if (d.type == SPLITTER) return d.x - DX[d.dir] == sx && d.y - DY[d.dir] == sy;
        return false;
    };
    auto ekey = [](int s, int t) { return (uint64_t(uint32_t(s)) << 32) | uint32_t(t); };
    struct HE { double p; int s, t; };
    // std::push_heap / pop_heap with the same max-heap sift rules as Rust's BinaryHeap
    std::vector<HE> heap;
    auto push = [&](int s, int t) {
        int prio;
        if (outcount[s] == 1 && incoming[t].size() == 1) prio = 2147483647;
        else { auto it = edge_last.find(ekey(s, t)); prio = it == edge_last.end() ? 0 : -it->second; }
        double p = double(prio) + rng_dist.f64();
        // Rust BinaryHeap::push: sift_up while strictly greater than the parent
        heap.push_back({p, s, t});
        size_t i = heap.size() - 1;
        HE x = heap[i];
        while (i > 0) {
            size_t par = (i - 1) / 2;
            if (!(x.p > heap[par].p)) break;
            heap[i] = heap[par]; i = par;
        }
        heap[i] = x;
    };
    auto pop = [&](HE& out) {
        if (heap.empty()) return false;
        // Rust BinaryHeap::pop: swap last into the root, sift_down_to_bottom, then sift_up
        HE last = heap.back(); heap.pop_back();
        if (heap.empty()) { out = last; return true; }
        out = heap[0];
        size_t end = heap.size(), pos = 0, child = 1;
        while (child + 1 < end) {
            if (!(heap[child].p > heap[child + 1].p)) child++;   // pick the greater (right on ties)
            heap[pos] = heap[child]; pos = child; child = 2 * pos + 1;
        }
        if (child == end - 1) { heap[pos] = heap[child]; pos = child; }
        heap[pos] = last;
        while (pos > 0) {
            size_t par = (pos - 1) / 2;
            if (!(last.p > heap[par].p)) break;
            heap[pos] = heap[par]; pos = par;
        }
        heap[pos] = last;
        return true;
    };
    for (int x = 0; x < W; x++)
        for (int y = 0; y < H; y++) {
            int t = idx(x, y);
            if (incoming[t].empty() || !can_take(t)) continue;
            for (int s : incoming[t])
                if (!idle[s] && accepts(s, t)) push(s, t);
        }
    HE e;
    while (pop(e)) {
        if (idle[e.s] || !can_take(e.t)) continue;
        Entity& src = ents[bld[e.s]];
        uint8_t res; uint32_t sid;
        if (src.type == HARVESTER) { res = src.res; sid = new_id(); src.hcd = 4; }
        else { res = src.stored; sid = src.stack_id; src.stored = RNONE; src.stack_id = 0; }
        Entity& d = ents[bld[e.t]];
        last_moves.push_back({e.s % W, e.s / W, e.t % W, e.t / W, res, sid});
        if (d.type == CORE) {
            Team& tm = teams[d.team];
            if (res == RTI) { tm.ti += STACK; tm.ti_collected += STACK; }
            else { tm.vo += STACK; tm.vo_collected += STACK; }
        } else { d.stored = res; d.stack_id = sid; }
        edge_last[ekey(e.s, e.t)] = round;
        idle[e.s] = 1;
        if (can_take(e.s))
            for (int s2 : incoming[e.s])
                if (!idle[s2] && accepts(s2, e.s)) push(s2, e.s);
    }
}

void Game::end_round() {
    if (over) return;
    distribute();
    if (round % 4 == 3) { teams[0].ti += 10; teams[1].ti += 10; }
    for (auto& e : ents) {
        if (!e.alive) continue;
        if (e.acd > 0) e.acd--;
        if (e.mcd > 0) e.mcd--;
        if (e.hcd > 0) e.hcd--;
        e.acted = e.moved = false;
    }
    for (auto& t : teams) { t.converted = false; t.store = t.store_next; }
    round++;
    if (round >= max_turns && winner < 0) {
        auto harv = [&](int team) { int n = 0; for (auto& e : ents) if (e.alive && e.team == team && e.type == HARVESTER) n++; return n; };
        int a0 = teams[0].ti_collected, b0 = teams[1].ti_collected;
        int a1 = harv(0), b1 = harv(1);
        int a2 = teams[0].ti, b2 = teams[1].ti;
        if (a0 != b0) { winner = a0 > b0 ? 0 : 1; win_condition = "titanium_collected"; }
        else if (a1 != b1) { winner = a1 > b1 ? 0 : 1; win_condition = "harvesters"; }
        else if (a2 != b2) { winner = a2 > b2 ? 0 : 1; win_condition = "titanium_stored"; }
        else {
            // official: rng.gen::<bool>() on the distribution generator (top bit of next_u32), true = A
            winner = (rng_dist.next_u32() >> 31) ? 0 : 1; win_condition = "coinflip";
        }
    }
    if (winner >= 0) over = true;
}

void Game::limit_ore(int radius, uint64_t seed) {
    if (radius < 0 || teams[0].core < 0 || teams[1].core < 0) return;
    const Entity& c0 = ents[teams[0].core];
    const Entity& c1 = ents[teams[1].core];
    // the map symmetry that carries core A's footprint onto core B's
    auto tr = [&](int k, int x, int y) {
        if (k == 0) return std::pair<int, int>(W - 1 - x, H - 1 - y);   // 180 rotation
        if (k == 1) return std::pair<int, int>(W - 1 - x, y);           // mirror x
        return std::pair<int, int>(x, H - 1 - y);                       // mirror y
    };
    int sym = -1, best_match = -1;
    for (int k = 0; k < 3; k++) {
        auto a = tr(k, c0.x, c0.y), b = tr(k, c0.x + 1, c0.y + 1);
        if (std::min(a.first, b.first) != c1.x || std::min(a.second, b.second) != c1.y) continue;
        int match = 0;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            auto m = tr(k, x, y);
            match += env[idx(x, y)] == env[idx(m.first, m.second)];
        }
        if (match > best_match) { best_match = match; sym = k; }
    }
    if (sym < 0) sym = 0;
    int n_ti = 0, n_vo = 0;
    for (auto& t : env) {
        if (t == ORE_TI) { n_ti++; t = EMPTY; }
        else if (t == ORE_VO) { n_vo++; t = EMPTY; }
    }
    // candidate tiles around core A whose mirror is also free and distinct
    auto free_tile = [&](int x, int y) {
        int i = idx(x, y);
        return env[i] == EMPTY && bld[i] < 0 && bot[i] < 0;
    };
    std::vector<std::pair<int, int>> cand;
    for (int y = c0.y - radius; y <= c0.y + 1 + radius; y++)
        for (int x = c0.x - radius; x <= c0.x + 1 + radius; x++) {
            if (!inb(x, y) || !free_tile(x, y)) continue;
            auto m = tr(sym, x, y);
            if (m == std::pair<int, int>(x, y) || !free_tile(m.first, m.second)) continue;
            cand.push_back({x, y});
        }
    std::mt19937_64 rng(seed ^ 0x0BE5EEDULL);
    std::shuffle(cand.begin(), cand.end(), rng);
    int want_ti = n_ti / 2, want_vo = n_vo / 2;
    for (auto& [x, y] : cand) {
        auto m = tr(sym, x, y);
        if (env[idx(x, y)] != EMPTY || env[idx(m.first, m.second)] != EMPTY) continue;  // taken as a mirror
        uint8_t t;
        if (want_ti > 0) { t = ORE_TI; want_ti--; }
        else if (want_vo > 0) { t = ORE_VO; want_vo--; }
        else break;
        env[idx(x, y)] = t;
        env[idx(m.first, m.second)] = t;
    }
}

}  // namespace fe
