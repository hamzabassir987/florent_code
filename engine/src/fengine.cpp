#include "fengine.hpp"

#include <algorithm>
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

void Game::reset(uint64_t seed) {
    rng = seed ? seed : 1;
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
    e.id = next_id++;
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
bool Game::in_vision(int a, int x, int y) const {
    const Entity& e = ents[a];
    int r2 = VISION[e.type];
    if (e.type == CORE) {  // from the nearest footprint tile
        int dx = x < e.x ? e.x - x : (x > e.x + 1 ? x - e.x - 1 : 0);
        int dy = y < e.y ? e.y - y : (y > e.y + 1 ? y - e.y - 1 : 0);
        return dx * dx + dy * dy <= r2;
    }
    int dx = x - e.x, dy = y - e.y;
    return dx * dx + dy * dy <= r2;
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

int Game::attack(int a, int x, int y) {
    int r = builder_can_act(*this, a);
    if (r) return r;
    Entity& e = ents[a];
    if (!inb(x, y) || !orth_adjacent(e, x, y)) return E_RANGE;
    int b = bld[idx(x, y)];   // measured: attacking an empty tile is legal (and still costs 2 Ti)
    if (teams[e.team].ti < 2) return E_FUNDS;
    teams[e.team].ti -= 2;
    e.acd = 1; e.acted = true;
    if (b >= 0) damage(b, 2);
    return OK;
}

int Game::heal(int a, int x, int y) {
    int r = builder_can_act(*this, a);
    if (r) return r;
    Entity& e = ents[a];
    if (!inb(x, y) || !orth_adjacent(e, x, y)) return E_RANGE;
    int b = bld[idx(x, y)], u = bot[idx(x, y)];
    bool any = (b >= 0 && ents[b].team == e.team && ents[b].hp < ents[b].max_hp) ||
               (u >= 0 && ents[u].team == e.team && ents[u].hp < ents[u].max_hp);
    if (!any) return E_NO_TARGET;
    if (teams[e.team].ti < 1) return E_FUNDS;
    teams[e.team].ti -= 1;
    e.acd = 1; e.acted = true;
    for (int t : {b, u})
        if (t >= 0 && ents[t].team == e.team) ents[t].hp = std::min(ents[t].max_hp, ents[t].hp + 4);
    return OK;
}

int Game::destroy(int a, int x, int y) {
    if (a < 0 || !ents[a].alive) return E_DEAD;
    const Entity& e = ents[a];
    if (e.type != BUILDER) return E_WRONG_TYPE;
    if (!inb(x, y) || !orth_adjacent(e, x, y)) return E_RANGE;
    int b = bld[idx(x, y)];
    if (b < 0 || ents[b].team != e.team || ents[b].type == CORE) return E_NO_TARGET;
    kill(b);
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

int Game::convert_ammo(int c, int amount) {
    if (c < 0 || !ents[c].alive) return E_DEAD;
    if (ents[c].type != CORE) return E_WRONG_TYPE;
    Team& t = teams[ents[c].team];
    if (t.converted) return E_ONCE_PER_TURN;
    if (amount <= 0 || amount > t.vo) return E_FUNDS;
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

int Game::rotate(int a, int dir) {
    if (a < 0 || !ents[a].alive) return E_DEAD;
    Entity& e = ents[a];
    if (e.type != GUNNER) return E_WRONG_TYPE;
    if (e.acd > 0) return E_COOLDOWN;
    if (dir == C || dir > NW) return E_BAD_DIR;
    if (teams[e.team].ti < 10) return E_FUNDS;
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
    if (en == WALL) return E_BLOCKED;
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
// Every round, after all units acted. A stack moves one tile. Measured rules:
//  - a conveyor holds one stack and passes it to the tile in front;
//  - a receiver takes from at most one source per round, the input side it
//    used least recently (strict alternation in the replays);
//  - a harvester outputs every 4 rounds (first on the round it is built) to
//    its least-recently-used accepting neighbour; if none is free it waits;
//  - a chain advances as a whole in one round (a vacated tile can be refilled).
void Game::distribute() {
    last_moves.clear();
    struct Src { int e; int x, y; int cand[4]; int nc; };
    std::vector<Src> srcs;
    auto accepts = [&](int sx, int sy, int tx, int ty) -> bool {
        if (!inb(tx, ty)) return false;
        int t = bld[idx(tx, ty)];
        if (t < 0) return false;
        const Entity& d = ents[t];
        if (d.type == CORE) return true;
        if (d.type == CONVEYOR) return !(d.x + DX[d.dir] == sx && d.y + DY[d.dir] == sy);
        if (d.type == SPLITTER) return d.x - DX[d.dir] == sx && d.y - DY[d.dir] == sy;
        return false;
    };
    for (size_t i = 0; i < ents.size(); i++) {
        const Entity& e = ents[i];
        if (!e.alive) continue;
        Src s{int(i), e.x, e.y, {0, 0, 0, 0}, 0};
        if (e.type == CONVEYOR && e.stored != RNONE) {
            int tx = e.x + DX[e.dir], ty = e.y + DY[e.dir];
            if (accepts(e.x, e.y, tx, ty)) { s.cand[0] = idx(tx, ty); s.nc = 1; }
        } else if (e.type == SPLITTER && e.stored != RNONE) {
            int ds[3] = {int(e.dir), (e.dir == N ? int(fe::W) : e.dir - 2), (e.dir == fe::W ? int(N) : e.dir + 2)};
            for (int d : ds) { int tx = e.x + DX[d], ty = e.y + DY[d]; if (accepts(e.x, e.y, tx, ty)) s.cand[s.nc++] = idx(tx, ty); }
        } else if (e.type == HARVESTER && round - e.last_out >= 4) {
            for (int d : {int(N), int(E), int(S), int(fe::W)}) { int tx = e.x + DX[d], ty = e.y + DY[d]; if (accepts(e.x, e.y, tx, ty)) s.cand[s.nc++] = idx(tx, ty); }
        }
        if (s.nc) srcs.push_back(s);
    }
    std::vector<int> target(srcs.size(), -1);           // chosen tile per source
    std::vector<char> taken(size_t(W) * H, 0), moving(size_t(W) * H, 0);
    auto is_core = [&](int t) { int b = bld[t]; return b >= 0 && ents[b].type == CORE; };
    auto free_tile = [&](int t) {
        if (is_core(t)) return true;
        if (taken[t]) return false;
        const Entity& d = ents[bld[t]];
        return d.stored == RNONE || moving[t];
    };
    auto side = [&](int fromx, int fromy, int tox, int toy) { return card_index(fromx - tox, fromy - toy); };
    bool changed = true;
    std::vector<std::vector<int>> props(size_t(W) * H);
    std::vector<int> touched;
    while (changed) {
        changed = false;
        touched.clear();
        for (size_t s = 0; s < srcs.size(); s++) {
            if (target[s] >= 0) continue;
            const Src& S = srcs[s];
            const Entity& se = ents[S.e];
            int best = -1; long bk = 0;
            for (int c = 0; c < S.nc; c++) {
                int t = S.cand[c];
                if (!free_tile(t)) continue;
                int sd = card_index(t % W - S.x, t / W - S.y);
                long key = long(se.out_last[sd]) * 8 + sd;
                if (best < 0 || key < bk) { best = t; bk = key; }
            }
            if (best < 0) continue;
            if (props[best].empty()) touched.push_back(best);
            props[best].push_back(int(s));
        }
        for (int t : touched) {
            auto& ps = props[t];
            if (is_core(t)) {
                for (int s : ps) { target[s] = t; if (ents[srcs[s].e].type != HARVESTER) moving[idx(srcs[s].x, srcs[s].y)] = 1; }
            } else {
                const Entity& d = ents[bld[t]];
                int best = -1; long bk = 0;
                for (int s : ps) {
                    int sd = side(srcs[s].x, srcs[s].y, t % W, t / W);
                    long key = long(d.in_last[sd]) * 8 + sd;
                    if (best < 0 || key < bk) { best = s; bk = key; }
                }
                target[best] = t; taken[t] = 1;
                if (ents[srcs[best].e].type != HARVESTER) moving[idx(srcs[best].x, srcs[best].y)] = 1;
            }
            ps.clear();
            changed = true;
        }
    }
    // apply simultaneously
    std::vector<uint8_t> carried(srcs.size(), RNONE);
    for (size_t s = 0; s < srcs.size(); s++) {
        if (target[s] < 0) continue;
        Entity& e = ents[srcs[s].e];
        int t = target[s];
        int sd = card_index(t % W - e.x, t / W - e.y);
        if (e.type == HARVESTER) { carried[s] = e.res; e.last_out = round; }
        else { carried[s] = e.stored; e.stored = RNONE; }
        if (sd >= 0) e.out_last[sd] = round;
    }
    for (size_t s = 0; s < srcs.size(); s++) {
        if (target[s] < 0) continue;
        int t = target[s];
        Entity& d = ents[bld[t]];
        last_moves.push_back({srcs[s].x, srcs[s].y, t % W, t / W, carried[s]});
        if (d.type == CORE) {
            Team& tm = teams[d.team];
            if (carried[s] == RTI) { tm.ti += STACK; tm.ti_collected += STACK; }
            else { tm.vo += STACK; tm.vo_collected += STACK; }
        } else {
            d.stored = carried[s];
            int sd = side(srcs[s].x, srcs[s].y, d.x, d.y);
            if (sd >= 0) d.in_last[sd] = round;
        }
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
        e.acted = e.moved = false;
    }
    for (auto& t : teams) { t.converted = false; t.store = t.store_next; }
    round++;
    if (round >= MAX_TURNS && winner < 0) {
        auto harv = [&](int team) { int n = 0; for (auto& e : ents) if (e.alive && e.team == team && e.type == HARVESTER) n++; return n; };
        int a = teams[0].ti_collected, b = teams[1].ti_collected;
        if (a != b) { winner = a > b ? 0 : 1; win_condition = "titanium_collected"; return; }
        a = harv(0); b = harv(1);
        if (a != b) { winner = a > b ? 0 : 1; win_condition = "harvesters"; return; }
        a = teams[0].ti; b = teams[1].ti;
        if (a != b) { winner = a > b ? 0 : 1; win_condition = "titanium_stored"; return; }
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        winner = int(rng & 1); win_condition = "coinflip";
    }
    if (winner >= 0) over = true;
}

}  // namespace fe
