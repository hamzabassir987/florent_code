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
    stood.assign(n, 0);
    visit_round.assign(n, -1);
    bld_dir.assign(n, -1);
    bld_hp.assign(n, -1);
    bot_dir.assign(n, -1);
    bot_hp.assign(n, -1);
    attr_seen_round.assign(n, -1);
    for (auto& q : bus_q) q.clear();
    snap.clear(); snap_ore.assign(n, 0);
    rep_bot_tile.assign(128, -1); rep_bot_round.assign(128, -1);
    born = -1; last_spawn_react = -1000; snap_spawn = false; core_alarm_round = -1;
    last_hp = -1; spawned_now = false; core_alarm_now = false; is_builder = false;
    sym_alive = 7; sym = -1;
    // a pit lies on the mirror axis (controller get_pit): x pit -> mirror x, y pit -> mirror y
    if (g_bus.use_pit && g.pit_axis >= 0) sym_alive = uint8_t(1u << (g.pit_axis == 0 ? 1 : 2));
}

// ---------------------------------------------------------------- report bus
// Identical in bots/econ/encoder.py (checked by engine/tools/obs_parity.py).
// The 16 slots are split in groups (BusGroup, obs.hpp); the slot's group tells
// what a report means. A report (u32):
//   bit 31 marker; bit 30 spawn (a builder was just spawned); bits 28..29 the
//   sender's map symmetry + 1 (0 unknown); bits 0..10 tile; then per group:
//   G_MYB / G_EB  bits 11..16 tag (unit id % 63; 63 = the core's status),
//                 bits 17..22 round % 64, bit 27 core under attack (tag 63)
//   G_ORE         bit 11 kind (0 Ti, 1 Vo), bits 12..26 the other cells of the
//                 4x4 block right / below holding the same ore
//   G_EATK        bits 11..13 type (core, gunner, sentinel, launcher, spike), bit 14 removal
//   G_MINE        bits 11..13 type (launcher, spike, gunner, sentinel, mine, harvester), bit 14 removal
// Each unit keeps a send queue per group: one write per group per turn (slot
// base + (id + round) % size); an item is delivered when a snapshot shows its
// content, whoever wrote it; a sent item missing from the next snapshot is
// sent again; a delivered item is due again `refresh` rounds later; never-
// delivered items go first (enemy attacking buildings: nearest to our core
// first). A spawn bit makes everything due (once per spawn_cooldown rounds);
// a builder spawned after quiet_after writes nothing for its first quiet_rounds.
BusConfig g_bus;
static constexpr uint32_t W_MARK = 1u << 31, W_SPAWN = 1u << 30, W_ALARM = 1u << 27, W_REMOVE = 1u << 14;
static constexpr int TAG_CORE = 63;
static const int EATK_TYPES[5] = {CORE, GUNNER, SENTINEL, LAUNCHER, SPIKE};
static const int MINE_TYPES[6] = {LAUNCHER, SPIKE, GUNNER, SENTINEL, MINE, HARVESTER};
static int code_of(const int* types, int n, int type) { for (int i = 0; i < n; i++) if (types[i] == type) return i; return -1; }
static uint32_t content(uint32_t w) { return w & ~(W_MARK | W_SPAWN | (3u << 28)); }
static int round_from6(int now, int r6) { return now - ((now - r6) & 63); }
static int cheb(int a, int b, int W) { return std::max(std::abs(a % W - b % W), std::abs(a / W - b / W)); }

int ObsMemory::own_count(int type) const {
    int n = 0;
    for (size_t t = 0; t < bld_id.size(); t++) n += bld_id[t] >= 0 && bld_team[t] == team && bld_type[t] == type;
    return n;
}
uint32_t ObsMemory::extra_bits() const { return sym >= 0 ? uint32_t(sym + 1) << 28 : 0u; }

int ObsMemory::mirror(int t, int s) const {
    const int x = t % W, y = t / W;
    return ((s == 1) ? y : H - 1 - y) * W + ((s == 2) ? x : W - 1 - x);
}
// A tile's terrain is known: rule out the symmetries it contradicts; once the
// symmetry is known, its mirror tile gets the same terrain.
void ObsMemory::learn_terrain(int t) {
    if (terrain[t] == T_UNKNOWN) return;
    const bool blocked = terrain[t] == T_WALL || terrain[t] == T_VOID;
    for (int s = 0; s < 3; s++) {
        if (!(sym_alive >> s & 1)) continue;
        const int mt = mirror(t, s);
        if ((terrain[mt] != T_UNKNOWN && terrain[mt] != terrain[t]) || (blocked && stood[mt])) sym_alive &= uint8_t(~(1u << s));
    }
    if (sym >= 0) {
        const int mt = mirror(t, sym);
        if (terrain[mt] == T_UNKNOWN) terrain[mt] = terrain[t];
    }
}
// A reported unit / building stands on t (terrain not known: ore or empty):
// a symmetry mapping t onto a known wall or void is out.
void ObsMemory::learn_passable(int t) {
    stood[t] = 1;
    for (int s = 0; s < 3; s++) {
        if (!(sym_alive >> s & 1)) continue;
        const uint8_t mt = terrain[mirror(t, s)];
        if (mt == T_WALL || mt == T_VOID) sym_alive &= uint8_t(~(1u << s));
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
    // the cores mirror each other: a candidate is out when it maps a known
    // enemy core tile onto a tile this unit saw without our core, or one of our
    // core tiles onto a tile it saw without the enemy core
    if (g_bus.use_enemy_core && sym < 0) {
        for (int t = 0; t < N; t++) {
            if (bld_id[t] < 0 || bld_type[t] != CORE) continue;
            const int want = bld_team[t] != team ? team : 1 - team;
            for (int s = 0; s < 3; s++) {
                if (!(sym_alive >> s & 1)) continue;
                const int mt = mirror(t, s);
                if (seen_round[mt] >= 0 && !(bld_id[mt] >= 0 && bld_type[mt] == CORE && bld_team[mt] == want))
                    sym_alive &= uint8_t(~(1u << s));
            }
        }
    }
    // a known enemy core tile must mirror a tile of our 2x2 core: a known one,
    // or an unseen tile next to every known one (part of our core not seen yet)
    if (g_bus.use_enemy_core && !own_core.empty() && sym < 0) {
        for (int t = 0; t < N; t++) {
            if (!(bld_id[t] >= 0 && bld_team[t] != team && bld_type[t] == CORE)) continue;
            for (int s = 0; s < 3; s++) {
                if (!(sym_alive >> s & 1)) continue;
                bool match = false;
                const int mt = mirror(t, s);
                for (int o : own_core) if (o == mt) { match = true; break; }
                if (!match && seen_round[mt] < 0) {
                    match = true;
                    for (int o : own_core) if (std::max(std::abs(o % W - mt % W), std::abs(o / W - mt / W)) > 1) { match = false; break; }
                }
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

// A reported builder (team tm, tag): moved to tile t, the round it was seen.
static void place_reported_bot(ObsMemory& m, int tm, int tag, int t, int rr) {
    const int idx = tm * 64 + tag, sid = (1 << 28) | idx;
    const int old = m.rep_bot_tile[idx];
    if (old >= 0 && old != t && m.bot_id[old] == sid) { m.bot_id[old] = -1; m.bot_team[old] = -1; m.bot_dir[old] = -1; m.bot_hp[old] = -1; }
    if (m.bot_id[t] < 0 || m.bot_id[t] == sid) {
        m.bot_id[t] = sid; m.bot_team[t] = int8_t(tm); m.bot_dir[t] = -1; m.bot_hp[t] = -1;
        m.visit_round[t] = rr;   // terrain left as is: a builder can stand on ore
        m.learn_passable(t);
    }
    m.rep_bot_tile[idx] = t; m.rep_bot_round[idx] = rr;
}

void ObsMemory::merge_store(const int64_t* store, int round) {
    const int N = W * H, enemy = 1 - team;
    snap.clear(); snap_ore.assign(N, 0); snap_spawn = false;
    for (int i = 0; i < 16; i++) {
        const int64_t v = store[i];
        if (v < 0 || v > 0xFFFFFFFFll || !(uint64_t(v) & W_MARK)) continue;
        const uint32_t w = uint32_t(v);
        int grp = 0; while (grp + 1 < BUS_GROUPS && i >= BUS_BASE[grp + 1]) grp++;
        const int t = w & 0x7FF;
        if (t >= N) continue;
        if (const int ss = int(w >> 28 & 3u); ss && sym < 0 && (sym_alive >> (ss - 1) & 1)) sym_alive = uint8_t(1u << (ss - 1));
        if (w & W_SPAWN) snap_spawn = true;
        snap.insert(uint32_t(grp) << 28 | content(w));
        if (grp == G_MYB || grp == G_EB) {
            const int tag = (w >> 11) & 63, rr = round_from6(round, (w >> 17) & 63);
            if (grp == G_MYB && tag == TAG_CORE) { if (w & W_ALARM) core_alarm_round = std::max(core_alarm_round, rr); continue; }
            if (round - rr > g_bus.enemy_bot_ttl) continue;
            place_reported_bot(*this, grp == G_MYB ? team : enemy, tag, t, rr);
        } else if (grp == G_ORE) {
            const int kind = (w >> 11) & 1, x = t % W, y = t / W;
            const uint32_t bits = (w >> 12) & 0x7FFF;
            for (int k = 0; k < 16; k++) {
                if (k > 0 && !(bits >> (k - 1) & 1)) continue;
                const int xx = x + k % 4, yy = y + k / 4;
                if (xx >= W || yy >= H) continue;
                const int tt = yy * W + xx;
                snap_ore[tt] |= uint8_t(1u << kind);
                if (terrain[tt] == T_UNKNOWN) terrain[tt] = kind ? T_ORE_VO : T_ORE_TI;
                learn_terrain(tt);
            }
        } else {
            const bool eatk = grp == G_EATK;
            const int code = (w >> 11) & 7;
            if (code >= (eatk ? 5 : 6)) continue;
            const int type = eatk ? EATK_TYPES[code] : MINE_TYPES[code], owner = eatk ? enemy : team;
            if (w & W_REMOVE) {
                if (bld_id[t] >= 0 && bld_team[t] == owner && bld_type[t] == type) {
                    bld_id[t] = -1; bld_type[t] = -1; bld_team[t] = -1; bld_dir[t] = -1; bld_hp[t] = -1;
                }
            } else if (bld_id[t] < 0) {
                // reported: a core keeps its real id (team + 1: its 4 tiles count once);
                // others get a per-tile id; direction and HP unknown, attribute age untouched
                bld_id[t] = type == CORE ? owner + 1 : (eatk ? (1 << 30) : (1 << 29)) + t;
                bld_type[t] = int8_t(type); bld_team[t] = int8_t(owner);
                bld_dir[t] = -1; bld_hp[t] = 100;   // terrain left as is (a harvester stands on ore)
                learn_passable(t);
            }
        }
    }
    // reported builders older than the ttl are forgotten
    for (int idx = 0; idx < 128; idx++) {
        const int t = rep_bot_tile[idx];
        if (t < 0 || round - rep_bot_round[idx] <= g_bus.enemy_bot_ttl) continue;
        if (bot_id[t] == ((1 << 28) | idx)) { bot_id[t] = -1; bot_team[t] = -1; bot_dir[t] = -1; bot_hp[t] = -1; }
        rep_bot_tile[idx] = -1;
    }
}

void ObsMemory::update(const Game& g, int actor) {
    std::vector<int> tiles;
    g.nearby_tiles(actor, -1, tiles);
    const int round = g.round, enemy = 1 - team;
    const Entity& me = g.ents[actor];
    if (born < 0) born = round;
    is_builder = me.type == BUILDER;
    std::vector<std::pair<int, int>> gone_eatk, gone_mine;   // (tile, code) of buildings seen gone now
    std::vector<std::pair<int, int>> enemy_bots;             // (tile, tag) visible now
    bool enemy_near = false;
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
            if (g.ents[u].team != team) { enemy_bots.push_back({t, int(uint32_t(observed_id) % 63u)}); enemy_near = true; }
        } else {
            bot_id[t] = -1;
            bot_team[t] = -1;
            bot_dir[t] = -1;
            bot_hp[t] = -1;
        }
        seen_round[t] = g.round;
        learn_terrain(t);
        if (bld_id[t] >= 0 && bld_team[t] != team) enemy_near = true;
        // buildings seen gone (or replaced) now
        if (old_id >= 0) {
            const bool same = bld_id[t] >= 0 && bld_type[t] == old_type && bld_team[t] == old_team;
            if (!same) {
                if (old_team == enemy) { const int c = code_of(EATK_TYPES, 5, old_type); if (c >= 0) gone_eatk.push_back({t, c}); }
                if (old_team == team) { const int c = code_of(MINE_TYPES, 6, old_type); if (c >= 0) gone_mine.push_back({t, c}); }
            }
        }
    }
    resolve_symmetry();
    // the core's status: under attack when its HP dropped or an enemy is in view
    if (me.type == CORE) {
        core_alarm_now = (last_hp >= 0 && me.hp < last_hp) || enemy_near;
        last_hp = me.hp;
        if (core_alarm_now) core_alarm_round = round;
    }

    // ---- send queues: the items this unit should keep posting
    const int N = W * H;
    auto find = [](std::vector<BusItem>& q, uint32_t key) -> BusItem* { for (auto& it : q) if (it.key == key) return &it; return nullptr; };
    auto want = [&](int grp, uint32_t key, uint32_t word, bool remove) -> BusItem& {
        BusItem* it = find(bus_q[grp], key);
        if (!it) { BusItem n; n.key = key; n.word = word; n.due = round; n.remove = remove; n.anchor = -1; bus_q[grp].push_back(n); it = &bus_q[grp].back(); }
        it->word = word;
        return *it;
    };
    std::set<uint32_t> keep[BUS_GROUPS];
    // our builders: own position (moved 2+ tiles since the last delivered post: due now); core status
    {
        std::vector<BusItem> q;
        for (auto& it : bus_q[G_MYB]) {
            if (it.key == 0 && is_builder) q.push_back(it);
            if (it.key == uint32_t(TAG_CORE) && me.type == CORE && (core_alarm_now || spawned_now)) q.push_back(it);
        }
        bus_q[G_MYB] = q;
        const uint32_t r6 = uint32_t(round & 63) << 17;
        if (is_builder) {
            const int t = me.y * W + me.x;
            BusItem& it = want(G_MYB, 0, uint32_t(t) | (uint32_t(uint32_t(me.id) % 63u) << 11) | r6, false);
            if (it.anchor >= 0 && cheb(it.anchor, t, W) >= 2) it.due = std::min(it.due, round);
        }
        if (me.type == CORE && (core_alarm_now || spawned_now)) {
            BusItem& it = want(G_MYB, TAG_CORE, uint32_t(me.y * W + me.x) | (uint32_t(TAG_CORE) << 11) | r6 | (core_alarm_now ? W_ALARM : 0u), false);
            it.due = std::min(it.due, round);       // every turn while there is something to say
        }
    }
    // enemy builders in view now (others are dropped)
    {
        std::vector<BusItem> q;
        for (auto& it : bus_q[G_EB]) for (auto& eb : enemy_bots) if (it.key == uint32_t(eb.second)) { q.push_back(it); break; }
        bus_q[G_EB] = q;
        for (auto& [t, tag] : enemy_bots) {
            BusItem& it = want(G_EB, uint32_t(tag), uint32_t(t) | (uint32_t(tag) << 11) | (uint32_t(round & 63) << 17), false);
            if (it.anchor >= 0 && cheb(it.anchor, t, W) >= 2) it.due = std::min(it.due, round);
        }
    }
    // ore, enemy attacking buildings, our buildings: everything memory holds
    for (int t = 0; t < N; t++) {
        if (terrain[t] == T_ORE_TI || terrain[t] == T_ORE_VO) {
            const uint32_t key = uint32_t(t) | (terrain[t] == T_ORE_VO ? 1u << 11 : 0u);
            want(G_ORE, key, key, false); keep[G_ORE].insert(key);
        }
        if (bld_id[t] < 0) continue;
        if (bld_team[t] == enemy) {
            const int c = code_of(EATK_TYPES, 5, bld_type[t]);
            if (c >= 0) { const uint32_t key = uint32_t(t) | uint32_t(c) << 11; want(G_EATK, key, key, false); keep[G_EATK].insert(key); }
        } else if (bld_team[t] == team) {
            const int c = code_of(MINE_TYPES, 6, bld_type[t]);
            if (c >= 0) { const uint32_t key = uint32_t(t) | uint32_t(c) << 11; want(G_MINE, key, key, false); keep[G_MINE].insert(key); }
        }
    }
    for (auto& [t, c] : gone_eatk) { const uint32_t key = uint32_t(t) | uint32_t(c) << 11 | W_REMOVE; want(G_EATK, key, key, true); keep[G_EATK].insert(key); }
    for (auto& [t, c] : gone_mine) { const uint32_t key = uint32_t(t) | uint32_t(c) << 11 | W_REMOVE; want(G_MINE, key, key, true); keep[G_MINE].insert(key); }
    for (int grp : {G_ORE, G_EATK, G_MINE}) {
        std::vector<BusItem> q;
        for (auto& it : bus_q[grp]) if (keep[grp].count(it.key) || it.remove) q.push_back(it);   // removals stay until delivered
        bus_q[grp] = q;
    }

    // ---- delivery checks against the snapshot merged this turn
    for (int grp = 0; grp < BUS_GROUPS; grp++) {
        std::vector<BusItem> q;
        for (auto& it : bus_q[grp]) {
            bool delivered;
            if (grp == G_ORE) delivered = snap_ore[it.key & 0x7FF] >> ((it.key >> 11) & 1) & 1;
            else if (grp == G_MYB || grp == G_EB) delivered = it.sent >= 0 && snap.count(uint32_t(grp) << 28 | it.sent_word);
            else delivered = snap.count(uint32_t(grp) << 28 | it.word);
            if (delivered) {
                if (it.remove) continue;                                   // a delivered removal is done
                if (it.sent >= 0 || it.due <= round) it.due = round + (grp <= G_EB ? g_bus.refresh : g_bus.refresh_facts);
                it.fresh = false; it.sent = -1; it.anchor = int(it.word & 0x7FF);
            } else if (it.sent >= 0 && it.sent < round) {
                it.due = round; it.sent = -1;                               // overwritten: send again
            }
            q.push_back(it);
        }
        bus_q[grp] = q;
    }
    if (snap_spawn && round - last_spawn_react >= g_bus.spawn_cooldown) {
        last_spawn_react = round;
        for (auto& q : bus_q) for (auto& it : q) it.due = std::min(it.due, round);
    }
}

void ObsMemory::next_reports(uint64_t unit_id, int round, std::vector<std::pair<int, uint32_t>>& out) {
    out.clear();
    if (is_builder && born > g_bus.quiet_after && round - born < g_bus.quiet_rounds) return;   // a late newcomer listens first
    const int N = W * H;
    std::vector<int> own_core;
    for (int t = 0; t < N; t++) if (bld_id[t] >= 0 && bld_team[t] == team && bld_type[t] == CORE) own_core.push_back(t);
    const uint64_t k = unit_id + uint64_t(round);
    const uint32_t spread = uint32_t((unit_id * 2654435761ull + uint64_t(round) * 40503ull) & 0xFFFFFFFFull);
    for (int grp = 0; grp < BUS_GROUPS; grp++) {
        BusItem* best = nullptr;
        std::tuple<int, int, uint32_t> bk;
        const bool refresh_ok = grp <= G_EB || k % uint64_t(g_bus.refresh_gate) == 0;
        for (auto& it : bus_q[grp]) {
            if (it.due > round) continue;
            if (!it.fresh && !refresh_ok) continue;     // refreshes are gated: fresh facts keep the slots
            int prio = 0;
            if (grp == G_EATK && !own_core.empty()) {
                prio = 1 << 20;
                for (int c : own_core) prio = std::min(prio, cheb(c, int(it.key & 0x7FF), W));
            }
            // spread: different units prefer different items (a per-unit key hash)
            const auto key = std::make_tuple(it.fresh ? 0 : 1, prio, it.key ^ spread);
            if (!best || key < bk) { best = &it; bk = key; }
        }
        if (!best) continue;
        uint32_t w = best->word;
        if (grp == G_ORE) {                                     // the 4x4 cluster from the anchor
            const int t = w & 0x7FF, x = t % W, y = t / W;
            const uint8_t ore = (w >> 11 & 1) ? T_ORE_VO : T_ORE_TI;
            for (int c = 1; c < 16; c++) {
                const int xx = x + c % 4, yy = y + c / 4;
                if (xx < W && yy < H && terrain[yy * W + xx] == ore) w |= 1u << (11 + c);
            }
        }
        best->sent = round;
        best->sent_word = w;
        uint32_t full = W_MARK | w | extra_bits();
        if (grp == G_MYB && best->key == uint32_t(TAG_CORE) && spawned_now) { full |= W_SPAWN; spawned_now = false; }
        out.push_back({BUS_BASE[grp] + int(k % uint64_t(BUS_SIZE[grp])), full});
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
    // rounds since our core last raised the under-attack alarm (team bus), cap 15
    scalars[15] = m.core_alarm_round < 0 ? 15 : std::min(15, g.round - m.core_alarm_round);
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
