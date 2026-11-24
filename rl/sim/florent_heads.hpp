// Factorized action heads (checkpoint ABI). Mirrored exactly by
// bots/econ/heads.py; rl/tests/heads_parity.py checks both against each other.
//
// Each acting type has a "mode" head plus sub-heads that participate only for
// the chosen mode. Every head is its own categorical distribution with its own
// legal mask (masks of later heads are conditional on earlier choices). PPO
// stores one record per participating head (head id, mask, choice, old logp).
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

#include "florent_actions.hpp"

namespace florent_rl {

enum HeadId : int {
    // core: both heads every turn (convert does not use the action cooldown)
    HC_SPAWN = 0,      // 13: none + 12 ring tiles
    HC_CONVERT,        // 5: none, 10%, 25%, 50%, 100% of current Voltium
    // builder
    HB_MODE,           // 6: wait, move, build, attack, heal, destroy
    HB_MOVE,           // 4: N E S W
    HB_BTYPE,          // 9: CONVEYOR SPLITTER HARVESTER GUNNER SENTINEL LAUNCHER SPIKE BARRIER MINE
    HB_BTILE,          // 4: adjacent N E S W
    HB_BDIR,           // 8: N NE E SE S SW W NW (directional building types only)
    HB_ATTACK,         // 4: adjacent N E S W
    HB_HEAL,           // 4
    HB_DESTROY,        // 4
    // gunner
    HG_MODE,           // 3: wait, fire, rotate
    HG_FIRE,           // 5: steps 1..5 along the facing line
    HG_ROTATE,         // 8
    // sentinel
    HS_MODE,           // 2: wait, fire
    HS_FIRE,           // 5
    // launcher
    HL_MODE,           // 2: wait, launch
    HL_BOT,            // 8: neighbours, row-major over dy, dx in [-1, 1]
    HL_DEST,           // 120: row-major over dy, dx in [-5, 5], centre excluded
    NUM_HEAD_IDS
};

constexpr int HEAD_SIZE[NUM_HEAD_IDS] = {13, 5, 6, 4, 9, 4, 8, 4, 4, 4, 3, 5, 8, 2, 5, 2, 8, 120};

// Policy (model) per acting type, and the heads each policy owns, in logit order.
enum PolicyId : int { P_CORE = 0, P_BUILDER, P_GUNNER, P_SENTINEL, P_LAUNCHER, NUM_POLICIES };
const std::vector<int>& policy_heads(int policy);
int policy_logit_count(int policy);
int head_offset(int policy, int head);          // offset of a head inside its policy's logit row
int policy_of_type(int type);                   // -1: no learned policy (spike: deterministic; buildings)

extern const int BUILD_ORDER[9];                // engine Type per HB_BTYPE index
extern const int CARD4[4];                      // N E S W (engine Dir)
extern const int CONVERT_TENTHS[5];             // 0 = none, then 1, 2.5, 5, 10 tenths (x10: 0,10,25,50,100 %)

struct HeadRecord {
    int head = 0, choice = 0;
    std::vector<uint8_t> mask;
    float logp = 0.f;
};

struct Decision {
    std::vector<Candidate> actions;             // executed in order (core: spawn then convert)
    std::vector<HeadRecord> records;            // participating heads
};

// Chooses one entry of a head. Gets the head's logits (HEAD_SIZE entries) and
// legal mask; returns the index and writes its log-probability under the
// masked softmax.
using Chooser = std::function<int(int head, const float* logits, const std::vector<uint8_t>& mask, float& logp)>;

// Conveyor planner mask (docs/rl-layout.md): plan4[i] is the planned conveyor
// direction (Dir code, 0 none) of the builder's i-th adjacent tile (CARD4
// order N E S W), read from the actor observation's path channel. With a plan,
// a conveyor / splitter may be built only on a planned tile, facing its planned
// direction. nullptr: no planner restriction.
void plan4_from_obs(const int32_t* spatial, int8_t plan4[4]);
void plan4_from_obs(const float* spatial, int8_t plan4[4]);
void plan4_from_obs(const uint8_t* spatial, int8_t plan4[4]);

// Does the unit have any legal action besides waiting? (if not: no forward)
bool has_choice(const fe::Game& g, int actor, const int8_t* plan4 = nullptr);

// Deterministic spike rule: fire at the first neighbour (row-major) holding a
// visible enemy entity if the spike can fire; else wait.
bool spike_action(const fe::Game& g, int actor, Candidate& out);

// Decode one turn. `logits` is this unit's policy logit row (policy_logit_count).
Decision decide(const fe::Game& g, int actor, const float* logits, const Chooser& choose, const int8_t* plan4 = nullptr);

// Masked log-softmax helpers (shared by sampling and argmax).
int argmax_choice(const float* logits, const std::vector<uint8_t>& mask, float& logp);
float masked_logp(const float* logits, const std::vector<uint8_t>& mask, int choice);

}  // namespace florent_rl
