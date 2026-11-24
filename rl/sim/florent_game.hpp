#pragma once
#include "../../engine/src/fengine.hpp"
#include "florent_actions.hpp"
#include "../../engine/src/obs.hpp"
#include <string>
#include <unordered_map>

namespace florent_rl {
struct Match {
    fe::Game game;
    std::string map_path;
    uint64_t seed=1;
    int critic_round=-1;
    std::vector<int32_t> critic_channels, critic_scalars;
    std::unordered_map<int, fe::ObsMemory> memories;
    bool load(const std::string& path, uint64_t s);
    bool load_bytes(const std::string& map26, uint64_t s);   // a generated map (rl/sim/map_gen.hpp)
    void reset();
    bool round_critic();
    int terminal_reward(int team) const;
    std::vector<int> acting_units() const;
    int act(int entity, const Candidate& c);
    void actor_observation(int entity, std::vector<float>& out);
    void observe(int entity);            // update this unit's fog memory (every turn it runs)
    void encode_only(int entity, std::vector<float>& out);   // encode without updating
    // Encode without updating, straight into the trainer's compact input:
    // spatial clamped to 0..255 (uint8), scalars to the int16 range (the same
    // values as encode_only followed by a float clamp). Returns the number of
    // spatial values that had to be clamped.
    long encode_compact(int entity, uint8_t* sp, int16_t* sc);
    std::vector<int32_t> enc_ch, enc_sc;   // encoder scratch (reused)
};
}
