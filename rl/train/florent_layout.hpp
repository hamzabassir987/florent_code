#pragma once
#include "../sim/florent_actions.hpp"
#include "../../engine/src/obs.hpp"
#include <array>
#include <cstddef>
#include <string>

namespace florent_rl {
constexpr int ACTOR_INPUT = fe::ACTOR_SPATIAL + fe::ACTOR_SCALARS;
constexpr int CRITIC_INPUT = fe::CRITIC_SPATIAL + fe::CRITIC_SCALARS;
constexpr float GAMMA = 0.9992f;
constexpr float LAMBDA = 0.95f;
constexpr float CLIP_EPS = 0.16f;
constexpr float END_REWARD = 1.0f;

enum Head : uint8_t {
    H_WAIT, H_MOVE, H_SPAWN, H_CONVERT, H_BUILD, H_ATTACK,
    H_HEAL, H_DESTROY, H_FIRE, H_ROTATE, H_LAUNCH, NUM_HEADS
};
struct Reward { static int terminal(const fe::Game& g, int team) { if(!g.over || g.winner<0) return 0; return g.winner==team ? 1 : -1; } };
}
