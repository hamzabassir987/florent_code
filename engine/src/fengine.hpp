// fengine: a native re-implementation of the Florent Code League game rules.
//
// Rules were measured from official replays (see engine/RULES.md and the
// verifier engine/tools/verify.py). One Game is one match; it is not
// thread-safe, use one per thread.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rustrng.hpp"

namespace fe {

enum Env : uint8_t { EMPTY = 0, WALL = 1, ORE_TI = 2, ORE_VO = 4, VOID = 5 };
enum Dir : uint8_t { C = 0, N, NE, E, SE, S, SW, W, NW };
enum Type : uint8_t {
    BUILDER = 0, CORE, GUNNER, SENTINEL, LAUNCHER, SPIKE,
    CONVEYOR, SPLITTER, HARVESTER, BARRIER, MINE, NTYPES
};
enum Res : uint8_t { RNONE = 0, RTI = 1, RVO = 4 };

// Error codes returned by actions (0 = ok).
enum Err : int {
    OK = 0, E_DEAD, E_WRONG_TYPE, E_COOLDOWN, E_ALREADY_ACTED, E_RANGE, E_BLOCKED,
    E_FUNDS, E_NOT_ORE, E_BAD_DIR, E_UNIT_CAP, E_SPIKE_COLLAR, E_NO_TARGET,
    E_NO_AMMO, E_TEAM, E_GAME_OVER, E_BAD_ARG, E_ONCE_PER_TURN
};

constexpr int MAX_TURNS = 1000;
constexpr int MAX_TEAM_UNITS = 50;
constexpr int STACK = 10;

extern const int DX[9], DY[9];
extern const int BASE_COST[NTYPES], SCALE_INC[NTYPES], MAX_HP[NTYPES], VISION[NTYPES];
bool is_unit(int t);        // counts toward the 50-unit cap (core, builder, turrets)
bool is_walkable(int t);    // a builder bot may stand on this building

struct Entity {
    int id = 0;
    uint8_t type = 0, team = 0, dir = 0, alive = 1;
    int x = 0, y = 0, hp = 0, max_hp = 0;
    int acd = 0, mcd = 0;          // action / move cooldown
    bool acted = false, moved = false;  // this round (builder: mutually exclusive)
    long seq = 0;                  // creation order = turn order
    // economy
    uint8_t stored = RNONE;        // conveyor / splitter
    uint8_t res = RNONE;           // harvester product
    uint32_t stack_id = 0;         // conveyor / splitter: id of the stack held
    int hcd = 0;                   // harvester cooldown (outputs when 0, then 4)
    int last_out = -1000000;       // unused (kept for ABI)
    std::array<int, 4> in_last{{-1, -1, -1, -1}};   // per side (N,E,S,W): round last received from
    std::array<int, 4> out_last{{-1, -1, -1, -1}};  // harvester / splitter: round last output to
};

struct Team {
    int ti = 500, vo = 0, ammo = 0;
    int ti_collected = 0, vo_collected = 0;
    int scale = 100;               // percent
    int units = 0;
    bool converted = false;        // convert_ammo used this round
    std::array<int64_t, 16> store{}, store_next{};
    int core = -1;                 // entity index
};

struct Move { int fx, fy, tx, ty; uint8_t res; uint32_t stack_id; };

class Game {
public:
    // ---- setup
    bool load_map(const std::string& path);          // .map26 file
    bool load_map_bytes(const uint8_t* data, size_t n);
    void reset(uint64_t seed = 1);

    // ---- map / state queries
    int W = 0, H = 0;
    std::vector<uint8_t> env;
    int pit_axis = -1, pit_at = 0, pit_w = 0;         // -1: no pit
    int round = 0;
    int winner = -1;                                   // -1 running, 0 A, 1 B
    bool over = false;                                 // set at the end of the round the game was decided
    std::string win_condition;
    std::vector<Entity> ents;                          // index = handle; dead entries kept
    std::vector<int> bld;                              // building index per tile (-1)
    std::vector<int> bot;                              // builder bot index per tile (-1)
    std::array<Team, 2> teams;
    std::vector<Move> last_moves;                      // resource moves of the last round

    int idx(int x, int y) const { return y * W + x; }
    bool inb(int x, int y) const { return x >= 0 && y >= 0 && x < W && y < H; }
    int find(int id) const;                            // entity index by id, -1
    int cost(int team, int type) const;
    bool passable(int x, int y) const;                 // a builder could stand here (ignores bots)

    // ---- the round
    // Units in turn order for this round (creation order). Units created during
    // the round act from the next round.
    std::vector<int> turn_order() const;
    void end_round();                                  // distribution, income, cooldowns, store, end check

    // ---- actions (actor = entity index). All return an Err.
    int move(int a, int dir);
    int build(int a, int type, int x, int y, int dir);
    int attack(int a, int x, int y);                   // builder: 2 Ti -> 2 dmg
    int heal(int a, int x, int y);                     // builder: 1 Ti -> +4 to own entities on tile
    int destroy(int a, int x, int y);                  // builder: own building, free
    int self_destruct(int a);
    int spawn(int core, int x, int y);
    int convert_ammo(int core, int amount);
    int fire(int a, int x, int y);                     // gunner / sentinel / spike
    int rotate(int a, int dir);                        // gunner: 10 Ti, acd 1
    int launch(int a, int bx, int by, int tx, int ty); // launcher
    int write_store(int team, int i, int64_t v);
    void resign(int team);

    // checks without side effects
    int can_move(int a, int dir) const;
    int can_build(int a, int type, int x, int y, int dir) const;
    int can_fire(int a, int x, int y) const;
    int can_spawn(int core, int x, int y) const;
    int can_launch(int a, int bx, int by, int tx, int ty) const;
    int can_attack(int a, int x, int y) const;
    int can_heal(int a, int x, int y) const;
    int can_destroy(int a, int x, int y) const;
    int can_rotate(int a, int dir) const;
    int can_convert(int core, int amount) const;

    // attack pattern of a turret placed at (x,y) facing dir
    void attack_tiles(int type, int x, int y, int dir, std::vector<std::pair<int,int>>& out) const;

    // ---- vision
    bool in_vision(int a, int x, int y) const;

    // ---- internals (public for the verifier)
    int create(int type, int team, int x, int y, int dir);
    void damage(int e, int amount);
    void kill(int e);
    void distribute();
    long next_seq = 0;
    int next_id = 3;
    uint64_t rng = 1;
    uint64_t seed = 1;
    StdRng rng_dist;                                   // StdRng(seed): distribution tie-breaks
    StdRng rng_ids;                                    // StdRng(seed ^ ID_SALT): entity and stack ids
    std::unordered_map<uint64_t, int> edge_last;       // (src tile, dst tile) -> round last used
    std::unordered_set<uint32_t> used_ids;             // every entity / stack id ever issued (never reused)
    uint32_t new_id();                                 // official id rule: redraw until unused
    int core_xy[2][2] = {{0, 0}, {0, 0}};
};

}  // namespace fe
