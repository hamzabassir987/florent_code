// Float teacher model (locked architecture, see SESSION_HANDOFF.md).
//
// Actor, one per acting type (core, builder, gunner, sentinel, launcher):
//   17x17x59 patch (scaled) -> 1x1 stem to 32 -> 4 x [1x1 pointwise -> ReLU ->
//   depthwise 3x3 -> + residual -> ReLU] -> centre 32 ++ 40 scaled scalars ->
//   Linear 72->128 -> ReLU -> Linear 128 -> that type's factorized head logits.
// Critic (training only): 48x48x29 board -> 1x1 stem to 32 ->
//   3 x [residual block -> 2x2 max-pool] (48->24->12->6) -> flatten ++ 33 scaled
//   scalars -> value (team-0 frame; team 1 uses -value).
#pragma once
#include <torch/torch.h>

#include "../../engine/src/obs.hpp"
#include "../sim/florent_heads.hpp"

namespace florent_rl {

constexpr int ACTOR_HIDDEN = 128;
constexpr int TRUNK_CH = 32;

// Fixed input scaling (part of the checkpoint ABI; the deployed student uses
// its own integer input map).
inline torch::Tensor actor_channel_scale() {
    std::vector<float> s(fe::ACTOR_CH, 1.f);
    s[fe::C_STALE] = s[fe::C_VISIT] = 1.f / 255.f;
    s[fe::C_ATTR_STALE] = s[fe::C_ATTR_STALE + 1] = 1.f / 255.f;
    s[fe::C_HP] = s[fe::C_HP + 1] = 1.f / 100.f;
    for (int k = 0; k < fe::CH_RAY; k++) s[fe::C_RAY + k] = 1.f / 32.f;
    return torch::tensor(s).view({1, fe::ACTOR_CH, 1, 1});
}
inline torch::Tensor actor_scalar_scale() {
    std::vector<float> s(fe::ACTOR_SCALARS, 0.f);
    for (int i = 0; i <= 5; i++) s[i] = 1.f;            // type one-hot
    s[6] = 1.f / 100.f;                                  // own hp %
    s[7] = s[8] = 1.f / 4.f;                             // cooldowns
    s[9] = 1.f / 1000.f;                                 // round
    s[10] = s[11] = s[12] = 1.f / 1000.f;                // Ti, Vo, ammo
    s[14] = 1.f / 50.f;                                  // unit count
    for (int i = 36; i <= 39; i++) s[i] = 1.f;           // turret/core reach flags
    // 13 (scale, always 0), 15-19 (opponent, always 0), 20-35 (packed store words): ignored
    return torch::tensor(s).view({1, fe::ACTOR_SCALARS});
}
inline torch::Tensor critic_scalar_scale() {
    std::vector<float> s(fe::CRITIC_SCALARS, 1.f / 10.f);  // per-type entity counts
    s[0] = 1.f / 1000.f;                                     // round
    for (int i = 1; i <= 10; i++) s[i] = 1.f / 1000.f;       // both teams' Ti/Vo/ammo/scale/units
    return torch::tensor(s).view({1, fe::CRITIC_SCALARS});
}

struct ResBlockImpl : torch::nn::Module {
    // two (1x1 -> depthwise 3x3) pairs, then residual: +-2 tiles per block
    torch::nn::Conv2d pw{nullptr}, dw{nullptr}, pw2{nullptr}, dw2{nullptr};
    explicit ResBlockImpl(int c) {
        pw = register_module("pw", torch::nn::Conv2d(torch::nn::Conv2dOptions(c, c, 1)));
        dw = register_module("dw", torch::nn::Conv2d(torch::nn::Conv2dOptions(c, c, 3).padding(1).groups(c)));
        pw2 = register_module("pw2", torch::nn::Conv2d(torch::nn::Conv2dOptions(c, c, 1)));
        dw2 = register_module("dw2", torch::nn::Conv2d(torch::nn::Conv2dOptions(c, c, 3).padding(1).groups(c)));
    }
    torch::Tensor forward(torch::Tensor x) {
        auto z = torch::relu(pw->forward(x));
        z = torch::relu(dw->forward(z));
        z = torch::relu(pw2->forward(z));
        z = dw2->forward(z);
        return torch::relu(x + z);
    }
};
TORCH_MODULE(ResBlock);

struct ActorImpl : torch::nn::Module {
    int policy;
    torch::nn::Conv2d stem{nullptr};
    torch::nn::ModuleList blocks;
    torch::nn::Linear fc{nullptr}, out{nullptr};
    explicit ActorImpl(int p) : policy(p) {
        stem = register_module("stem", torch::nn::Conv2d(torch::nn::Conv2dOptions(fe::ACTOR_CH, TRUNK_CH, 1)));
        for (int b = 0; b < 4; b++) blocks->push_back(ResBlock(TRUNK_CH));
        register_module("blocks", blocks);
        fc = register_module("fc", torch::nn::Linear(TRUNK_CH + fe::ACTOR_SCALARS, ACTOR_HIDDEN));
        out = register_module("out", torch::nn::Linear(ACTOR_HIDDEN, policy_logit_count(p)));
    }
    // spatial: [B, 59, 17, 17] already scaled; scalars: [B, 40] already scaled
    torch::Tensor forward(torch::Tensor spatial, torch::Tensor scalars) {
        auto h = torch::relu(stem->forward(spatial));
        for (auto& b : *blocks) h = b->as<ResBlock>()->forward(h);
        auto centre = h.index({torch::indexing::Slice(), torch::indexing::Slice(), fe::HALF, fe::HALF});
        auto f = torch::relu(fc->forward(torch::cat({centre, scalars}, 1)));
        return out->forward(f);
    }
};
TORCH_MODULE(Actor);

// Privileged critic: 48x48 board -> stem 1x1 -> [ResBlock, pool] x 3
// (2x2: 48 -> 24, 2x2: 24 -> 12, 4x4: 12 -> 3) -> 32*3*3 = 288 ++ scalars
// -> Linear 80 -> ReLU -> Linear 1.
struct CriticImpl : torch::nn::Module {
    static constexpr int CRITIC_HIDDEN = 80;
    torch::nn::Conv2d stem{nullptr};
    torch::nn::ModuleList blocks;
    torch::nn::Linear hidden{nullptr}, value{nullptr};
    CriticImpl() {
        stem = register_module("stem", torch::nn::Conv2d(torch::nn::Conv2dOptions(fe::CRITIC_CH, TRUNK_CH, 1)));
        for (int b = 0; b < 3; b++) blocks->push_back(ResBlock(TRUNK_CH));
        register_module("blocks", blocks);
        hidden = register_module("hidden", torch::nn::Linear(TRUNK_CH * 3 * 3 + fe::CRITIC_SCALARS, CRITIC_HIDDEN));
        value = register_module("value", torch::nn::Linear(CRITIC_HIDDEN, 1));
    }
    torch::Tensor forward(torch::Tensor spatial, torch::Tensor scalars) {
        static const int POOL[3] = {2, 2, 4};
        auto h = torch::relu(stem->forward(spatial));
        for (int b = 0; b < 3; b++)
            h = torch::max_pool2d(blocks[b]->as<ResBlock>()->forward(h), {POOL[b], POOL[b]}, {POOL[b], POOL[b]});
        auto f = torch::relu(hidden->forward(torch::cat({h.flatten(1), scalars}, 1)));
        return value->forward(f).squeeze(1);
    }
};
TORCH_MODULE(Critic);

struct TeacherImpl : torch::nn::Module {
    torch::nn::ModuleList actors;
    Critic critic{nullptr};
    torch::Tensor ch_scale, sc_scale, crit_scale;
    TeacherImpl() {
        for (int p = 0; p < NUM_POLICIES; p++) actors->push_back(Actor(p));
        register_module("actors", actors);
        critic = register_module("critic", Critic());
        ch_scale = register_buffer("ch_scale", actor_channel_scale());
        sc_scale = register_buffer("sc_scale", actor_scalar_scale());
        crit_scale = register_buffer("crit_scale", critic_scalar_scale());
    }
    // raw: [B, ACTOR_SPATIAL + ACTOR_SCALARS] unscaled encoder output
    torch::Tensor logits(int p, torch::Tensor raw) {
        auto sp = raw.narrow(1, 0, fe::ACTOR_SPATIAL).view({raw.size(0), fe::ACTOR_CH, fe::PATCH, fe::PATCH}) * ch_scale;
        auto sc = raw.narrow(1, fe::ACTOR_SPATIAL, fe::ACTOR_SCALARS) * sc_scale;
        return actors[p]->as<Actor>()->forward(sp, sc);
    }
    // raw: [B, CRITIC_SPATIAL + CRITIC_SCALARS]
    torch::Tensor value(torch::Tensor raw) {
        auto sp = raw.narrow(1, 0, fe::CRITIC_SPATIAL).view({raw.size(0), fe::CRITIC_CH, fe::CRITIC_PAD, fe::CRITIC_PAD});
        auto sc = raw.narrow(1, fe::CRITIC_SPATIAL, fe::CRITIC_SCALARS) * crit_scale;
        return critic->forward(sp, sc);
    }
};
TORCH_MODULE(Teacher);

}  // namespace florent_rl
