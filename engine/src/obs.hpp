// Observation encoder.
//
// Two observations are defined here:
//
//  * the ACTOR observation: a 17x17 patch centred on the acting unit, built
//    only from information that unit's controller is allowed to see (fog of
//    war), plus "edge rays" that summarise what lies beyond the patch in the
//    direction of every boundary cell. This is what the deployed Python bot
//    computes, so the C++ and Python versions must agree bit for bit
//    (engine/tools/obs_parity.py checks that).
//
//  * the CRITIC observation: the whole board at native resolution, without
//    fog. Training only; never deployed. Computed once per round, not once
//    per unit.
#pragma once
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "fengine.hpp"

namespace fe {

// ---------------------------------------------------------------- actor
constexpr int PATCH = 17;
constexpr int HALF = PATCH / 2;      // 8
constexpr int RAY_LEN = 32;          // max tiles a ray walks

// terrain codes kept in per-unit memory (0 = never seen)
enum ObsTerrain : uint8_t { T_UNKNOWN = 0, T_EMPTY, T_WALL, T_ORE_TI, T_ORE_VO, T_VOID };

constexpr int CH_TERRAIN = 6;        // empty, wall, ore_ti, ore_vo, void, unknown
constexpr int CH_OWN = 11;           // own entity one-hot, index = Type
constexpr int CH_ENEMY = 10;         // enemy entity one-hot, index = Type (mine omitted)
constexpr int CH_OCC = 2;            // own / enemy builder bot on the tile
constexpr int CH_STALE = 1;          // rounds since last seen (0 = visible now), cap 255
constexpr int CH_VISIT = 1;          // rounds since last observed builder presence
constexpr int CH_ATTR_STALE = 2;     // age of remembered building / bot attributes, cap 255
constexpr int CH_DIR = 9;            // building direction one-hot: centre, then compass directions
constexpr int CH_HP = 2;             // remembered building / builder HP percentage
constexpr int CH_THREAT = 4;         // own/enemy remembered turret reach: possible / confirmed
constexpr int CH_RAY = 11;           // ray aggregates, written on boundary cells only
constexpr int CH_PATH = 1;           // builders only: planned conveyor cell, value = conveyor direction (Dir 1..8), 0 none
constexpr int ACTOR_CH = CH_TERRAIN + CH_OWN + CH_ENEMY + CH_OCC + CH_STALE + CH_VISIT + CH_ATTR_STALE + CH_DIR + CH_HP + CH_THREAT + CH_RAY + CH_PATH;  // 60
constexpr int ACTOR_SPATIAL = ACTOR_CH * PATCH * PATCH;
// 40 base scalars, then the team's current price of each buildable type
// (controller get_*_cost(), in ACTOR_COST_TYPES order)
constexpr int ACTOR_SCALARS = 50;
constexpr int ACTOR_COSTS = 40;

constexpr int C_TERRAIN = 0;
constexpr int C_OWN = C_TERRAIN + CH_TERRAIN;
constexpr int C_ENEMY = C_OWN + CH_OWN;
constexpr int C_OCC = C_ENEMY + CH_ENEMY;
constexpr int C_STALE = C_OCC + CH_OCC;
constexpr int C_VISIT = C_STALE + CH_STALE;
constexpr int C_ATTR_STALE = C_VISIT + CH_VISIT;
constexpr int C_DIR = C_ATTR_STALE + CH_ATTR_STALE;
constexpr int C_HP = C_DIR + CH_DIR;
constexpr int C_THREAT = C_HP + CH_HP;
constexpr int C_RAY = C_THREAT + CH_THREAT;
constexpr int C_PATH = C_RAY + CH_RAY;
constexpr int LINE_CAP = 4;          // conveyor planner: harvesters per line into the core

// report-bus parameters (experiments: rl/tests/bus_stats.cpp); defaults = deployed
struct BusConfig {
    int refresh = 4;              // builder groups: a delivered item is due again after this many rounds
    int refresh_facts = 16;       // ore / buildings: the same (chosen with bus_stats)
    int refresh_gate = 4;         // ore / buildings: a refresh (non-fresh item) only on rounds where (id + round) % gate == 0
    int spawn_cooldown = 4;       // a unit reacts to the spawn bit at most once per this many rounds
    int quiet_after = 40;         // builders spawned after this round ...
    int quiet_rounds = 4;         // ... write nothing for their first rounds
    int enemy_bot_ttl = 10;       // reported enemy builders are forgotten after this many rounds
    bool use_pit = true, use_enemy_core = true;   // symmetry from the pit / a known enemy core
};
extern BusConfig g_bus;

// Team store layout (slot groups; the group tells what a slot holds):
constexpr int BUS_GROUPS = 5;
enum BusGroup { G_MYB = 0,    // slots 0..3: our builders' positions (+ the core's status, tag 63)
                G_EB,         // slots 4..5: enemy builder sightings (with the round)
                G_ORE,        // slots 6..9: ore clusters
                G_EATK,       // slots 10..13: enemy attacking buildings (core, gunner, sentinel, launcher, spike)
                G_MINE };     // slots 14..15: our buildings (launcher, spike, gunner, sentinel, mine, harvester)
constexpr int BUS_BASE[BUS_GROUPS] = {0, 4, 6, 10, 14}, BUS_SIZE[BUS_GROUPS] = {4, 2, 4, 4, 2};

// per-unit fog memory (one instance per unit)
struct ObsMemory {
    int W = 0, H = 0, team = 0;
    std::vector<uint8_t> terrain;       // ObsTerrain
    std::vector<int32_t> bld_id;        // remembered building, -1 = none
    std::vector<int8_t> bld_type;       // Type, when bld_id >= 0
    std::vector<int8_t> bld_team;
    std::vector<int32_t> bot_id;        // remembered builder bot, -1 = none
    std::vector<int8_t> bot_team;
    std::vector<int32_t> seen_round;    // last round in vision, -1 = never
    std::vector<int32_t> visit_round;   // last observed round with a builder on tile, -1 = never
    std::vector<int8_t> bld_dir;        // last observed direction, -1 = unavailable/centre
    std::vector<int16_t> bld_hp;        // last observed HP percentage
    std::vector<int8_t> bot_dir;
    std::vector<int16_t> bot_hp;
    std::vector<int32_t> attr_seen_round; // last round entity attributes were observed

    // ---- team report bus (the 16-slot team store; docs/team-memory.md).
    // Each turn a unit: merge_store (apply the snapshot), update (own vision;
    // refreshes its send queues, checks what was delivered), then next_reports
    // (at most one write per slot group; the caller writes them). Identical in
    // bots/econ/encoder.py, checked by engine/tools/obs_parity.py.
    struct BusItem {
        uint32_t key = 0;          // identity inside its group
        uint32_t word = 0;         // report content (without the symmetry / spawn bits)
        int due = 0;               // round it should be (re)sent from
        int sent = -1;             // round it was last written, -1 not awaiting a check
        bool fresh = true;         // never seen delivered yet
        bool remove = false;       // a removal: dropped once delivered
        uint32_t sent_word = 0;    // what was written (the delivery check)
        int anchor = -1;           // builder groups: tile at the last delivered post
    };
    std::vector<BusItem> bus_q[BUS_GROUPS];
    std::set<uint32_t> snap;               // the last snapshot's report contents (per group: group << 28 | content)
    std::vector<uint8_t> snap_ore;         // the last snapshot's ore coverage per tile (bit 0 Ti, bit 1 Vo)
    std::vector<int32_t> rep_bot_tile;     // builder tag (team * 64 + tag) -> tile of its reported position, -1
    std::vector<int32_t> rep_bot_round;    //   and the round it was reported at
    int born = -1;                         // round of this unit's first turn
    int last_spawn_react = -1000;          // round this unit last reacted to a spawn bit
    bool snap_spawn = false;               // the last snapshot carried a spawn bit
    int core_alarm_round = -1;             // latest round our core raised the alarm (reported), -1 never
    // the core's own status (its unit only)
    int last_hp = -1;                      // its HP at its previous turn
    bool spawned_now = false;              // it spawned a builder (set by the caller after the action)
    bool is_builder = false;
    void merge_store(const int64_t* store16, int round);
    // the reports to write this turn: (slot, word) pairs, at most one per group
    void next_reports(uint64_t unit_id, int round, std::vector<std::pair<int, uint32_t>>& out);
    int own_count(int type) const;          // own buildings of a type in memory (seen or reported)
    uint32_t extra_bits() const;            // symmetry bits carried by every report
    // map symmetry (0: 180 rotation, 1: mirror x, 2: mirror y): candidates
    // still consistent with the known terrain (or adopted from a teammate's
    // report); once one is left, known tiles are mirrored and the enemy core is
    // placed on the mirror of ours.
    uint8_t sym_alive = 7;
    int sym = -1;
    int mirror(int t, int s) const;
    void learn_terrain(int t);
    void learn_passable(int t);   // a reported unit stands on t
    std::vector<uint8_t> stood;   // a reported unit / building stood here (so not wall / void)
    void resolve_symmetry();

    void init(const Game& g, int team_);
    // Conveyor plan from this memory (see plan_paths in obs.cpp): per tile the
    // planned conveyor direction (Dir N/E/S/W), 0 = not on a planned path.
    void plan_paths(std::vector<int8_t>& plan) const;
    void update(const Game& g, int actor);
    // the core's status inputs (its own turn): HP now, enemy in view
    bool core_alarm_now = false;
};

// flat int32 buffer; channel-major, index = channel * PATCH*PATCH + (dy+HALF)*PATCH + (dx+HALF)
void encode_actor(const Game& g, const ObsMemory& m, int actor,
                  std::vector<int32_t>& ch, std::vector<int32_t>& scalars);

// ---------------------------------------------------------------- critic
constexpr int CRITIC_PAD = 48;       // maps up to 48x48; larger ones are clipped
constexpr int CC_TERRAIN = 5;        // empty, wall, ore_ti, ore_vo, void
constexpr int CC_OWN = 11;           // own (team 0 of the encoder call) entity one-hot
constexpr int CC_ENEMY = 11;
constexpr int CC_OCC = 2;            // builder bots
constexpr int CRITIC_CH = CC_TERRAIN + CC_OWN + CC_ENEMY + CC_OCC;   // 29
constexpr int CRITIC_SPATIAL = CRITIC_CH * CRITIC_PAD * CRITIC_PAD;
// round, 5 per team (Ti, Vo, ammo, scale, units), entity counts per team and
// type, then what decides a game at the round limit: rounds left, Ti / Vo
// collected per team, and the A - B differences of Ti collected and Ti stored.
constexpr int CRITIC_EXTRA = 7;
constexpr int CRITIC_SCALARS = 1 + 10 + 2 * NTYPES + CRITIC_EXTRA;

void encode_critic(const Game& g, std::vector<int32_t>& ch, std::vector<int32_t>& scalars);

}  // namespace fe
