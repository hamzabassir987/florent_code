#include "obs.hpp"

#include <algorithm>
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
}

void ObsMemory::update(const Game& g, int actor) {
    std::vector<int> tiles;
    g.nearby_tiles(actor, -1, tiles);
    for (int t : tiles) {
        int x = t % g.W, y = t / g.W;
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
    }
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
