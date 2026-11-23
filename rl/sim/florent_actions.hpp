#pragma once
#include "../../engine/src/fengine.hpp"
#include <cstdint>
#include <vector>

namespace florent_rl {

enum class UnitPolicy : uint8_t { Core, Builder, Gunner, Sentinel, Launcher, Spike, None };
enum class ActionKind : uint8_t {
    Wait, Move, Spawn, Convert, Build, Attack, Heal, Destroy,
    Fire, Rotate, Launch
};
struct Candidate { ActionKind kind; int a=0,b=0,c=0,d=0; };

// Largest fixed candidate list is the launcher (1 wait + 8 builders x 120
// destinations = 961). Keep the checkpoint head width above that and constant.
constexpr int MAX_CANDIDATES = 1024;
UnitPolicy policy_for_type(int type);
const char* policy_name(UnitPolicy p);

// Fixed ABI candidate list. Legal filtering never changes candidate indices.
void enumerate_candidates(const fe::Game& g, int actor, std::vector<Candidate>& out);
bool legal_candidate(const fe::Game& g, int actor, const Candidate& c);
int apply_candidate(fe::Game& g, int actor, const Candidate& c);

} // namespace florent_rl
