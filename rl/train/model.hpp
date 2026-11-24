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
// Scalars that can reach ~1000 enter at several scales: x becomes
// clamp(x / d, -SCALAR_CLIP, SCALAR_CLIP) for each d in SCALAR_SCALES, so a
// difference of 3 Ti shows at the fine scales and 400 vs 4000 stays apart at
// the coarse ones. Every other scalar is one feature at its fixed scale.
constexpr float SCALAR_SCALES[] = {1.f, 10.f, 100.f, 1000.f};
constexpr int N_SCALAR_SCALES = 4;
constexpr float SCALAR_CLIP = 2.f;
inline torch::Tensor multi_scale() {                          // [1, 1, N_SCALAR_SCALES]: 1/d
    std::vector<float> s;
    for (float d : SCALAR_SCALES) s.push_back(1.f / d);
    return torch::tensor(s).view({1, 1, N_SCALAR_SCALES});
}
// actor: round, Ti, Vo, ammo
inline std::vector<int64_t> actor_big() { return {9, 10, 11, 12}; }
// critic: round; per team Ti, Vo, ammo, scale; rounds left; Ti / Vo collected
// per team; Ti collected A - B; Ti stored A - B
inline std::vector<int64_t> critic_big() {
    const int e = fe::CRITIC_SCALARS - fe::CRITIC_EXTRA;
    return {0, 1, 2, 3, 4, 6, 7, 8, 9, e, e + 1, e + 2, e + 3, e + 4, e + 5, e + 6};
}
// The single-scale scalars: (index, scale). Actor scalars with scale 0 are dropped.
inline void single_scalars(const torch::Tensor& scale, const std::vector<int64_t>& big,
                           std::vector<int64_t>& idx, std::vector<float>& sc) {
    auto s = scale.flatten().contiguous();
    for (int64_t i = 0; i < s.numel(); i++) {
        const float v = s[i].item<float>();
        if (v == 0.f || std::find(big.begin(), big.end(), i) != big.end()) continue;
        idx.push_back(i); sc.push_back(v);
    }
}
inline torch::Tensor critic_single_scale() {
    std::vector<float> s(fe::CRITIC_SCALARS, 1.f / 10.f);    // entity counts per team and type
    s[5] = s[10] = 1.f / 50.f;                               // units per team
    return torch::tensor(s);
}
inline int actor_features() {
    std::vector<int64_t> i; std::vector<float> v;
    single_scalars(actor_scalar_scale(), actor_big(), i, v);
    return int(i.size() + actor_big().size() * N_SCALAR_SCALES);
}
inline int critic_features() {
    std::vector<int64_t> i; std::vector<float> v;
    single_scalars(critic_single_scale(), critic_big(), i, v);
    return int(i.size() + critic_big().size() * N_SCALAR_SCALES);
}

// raw scalars [B, N] -> features: the single ones scaled, then the big ones at every scale
struct ScalarFeatures {
    torch::Tensor single_idx, single_scale, big_idx;
    static ScalarFeatures make(const torch::Tensor& scale, const std::vector<int64_t>& big) {
        std::vector<int64_t> i; std::vector<float> v;
        single_scalars(scale, big, i, v);
        ScalarFeatures f;
        f.single_idx = torch::tensor(i, torch::kLong);
        f.single_scale = torch::tensor(v).view({1, -1});
        f.big_idx = torch::tensor(big, torch::kLong);
        return f;
    }
};

struct ResBlockImpl : torch::nn::Module {
    // two (1x1 -> depthwise 3x3) pairs, then residual: +-2 tiles per block
    torch::nn::Conv2d pw{nullptr}, dw{nullptr}, pw2{nullptr}, dw2{nullptr};
    explicit ResBlockImpl(int c) {
        pw = register_module("pw", torch::nn::Conv2d(torch::nn::Conv2dOptions(c, c, 1)));
        dw = register_module("dw", torch::nn::Conv2d(torch::nn::Conv2dOptions(c, c, 3).padding(1).groups(c)));
        pw2 = register_module("pw2", torch::nn::Conv2d(torch::nn::Conv2dOptions(c, c, 1)));
        dw2 = register_module("dw2", torch::nn::Conv2d(torch::nn::Conv2dOptions(c, c, 3).padding(1).groups(c)));
    }
    // in-place ReLUs / residual add: each conv output is reused, not copied
    torch::Tensor forward(torch::Tensor x) {
        auto z = pw->forward(x).relu_();
        z = dw->forward(z).relu_();
        z = pw2->forward(z).relu_();
        z = dw2->forward(z);
        return z.add_(x).relu_();
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
        fc = register_module("fc", torch::nn::Linear(TRUNK_CH + actor_features(), ACTOR_HIDDEN));
        out = register_module("out", torch::nn::Linear(ACTOR_HIDDEN, policy_logit_count(p)));
    }
    // spatial: [B, 59, 17, 17] already scaled; scalars: [B, actor_features()]
    torch::Tensor forward(torch::Tensor spatial, torch::Tensor scalars) {
        auto h = stem->forward(spatial).relu_();
        for (auto& b : *blocks) h = b->as<ResBlock>()->forward(h);
        auto centre = h.index({torch::indexing::Slice(), torch::indexing::Slice(), fe::HALF, fe::HALF});
        auto f = fc->forward(torch::cat({centre, scalars}, 1)).relu_();
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
        hidden = register_module("hidden", torch::nn::Linear(TRUNK_CH * 3 * 3 + critic_features(), CRITIC_HIDDEN));
        value = register_module("value", torch::nn::Linear(CRITIC_HIDDEN, 1));
    }
    torch::Tensor forward(torch::Tensor spatial, torch::Tensor scalars) {
        static const int POOL[3] = {2, 2, 4};
        auto h = stem->forward(spatial).relu_();
        for (int b = 0; b < 3; b++)
            h = torch::max_pool2d(blocks[b]->as<ResBlock>()->forward(h), {POOL[b], POOL[b]}, {POOL[b], POOL[b]});
        auto f = hidden->forward(torch::cat({h.flatten(1), scalars}, 1)).relu_();
        return value->forward(f).squeeze(1);
    }
};
TORCH_MODULE(Critic);

struct TeacherImpl : torch::nn::Module {
    torch::nn::ModuleList actors;
    Critic critic{nullptr};
    torch::Tensor ch_scale, scales;
    torch::Tensor a_idx, a_scale, a_big, c_idx, c_scale, c_big;   // scalar feature maps (buffers)
    TeacherImpl() {
        for (int p = 0; p < NUM_POLICIES; p++) actors->push_back(Actor(p));
        register_module("actors", actors);
        critic = register_module("critic", Critic());
        ch_scale = register_buffer("ch_scale", actor_channel_scale());
        auto af = ScalarFeatures::make(actor_scalar_scale(), actor_big());
        auto cf = ScalarFeatures::make(critic_single_scale(), critic_big());
        a_idx = register_buffer("a_idx", af.single_idx); a_scale = register_buffer("a_scale", af.single_scale);
        a_big = register_buffer("a_big", af.big_idx);
        c_idx = register_buffer("c_idx", cf.single_idx); c_scale = register_buffer("c_scale", cf.single_scale);
        c_big = register_buffer("c_big", cf.big_idx);
        scales = register_buffer("scales", multi_scale());
    }
    // raw: [B, ACTOR_SPATIAL + ACTOR_SCALARS] unscaled encoder output
    torch::Tensor logits(int p, torch::Tensor raw) {
        auto sp = raw.narrow(1, 0, fe::ACTOR_SPATIAL).view({raw.size(0), fe::ACTOR_CH, fe::PATCH, fe::PATCH}) * ch_scale;
        auto sc = actor_features(raw.narrow(1, fe::ACTOR_SPATIAL, fe::ACTOR_SCALARS));
        return actors[p]->as<Actor>()->forward(sp, sc);
    }
    // [B, CRITIC_SCALARS] raw -> [B, CRITIC_FEATURES]: each scalar at every scale, clamped
    torch::Tensor features(const torch::Tensor& sc, const torch::Tensor& idx, const torch::Tensor& scale, const torch::Tensor& big) {
        auto single = sc.index_select(1, idx) * scale;
        auto multi = (sc.index_select(1, big).unsqueeze(2) * scales).clamp_(-SCALAR_CLIP, SCALAR_CLIP).flatten(1);
        return torch::cat({single, multi}, 1);
    }
    torch::Tensor critic_features(const torch::Tensor& sc) { return features(sc, c_idx, c_scale, c_big); }
    torch::Tensor actor_features(const torch::Tensor& sc) { return features(sc, a_idx, a_scale, a_big); }
    // Compact inputs, already on the device: spatial uint8 [B, ACTOR_SPATIAL],
    // scalars any integer type [B, ACTOR_SCALARS]; converted to float here.
    torch::Tensor logits_split(int p, const torch::Tensor& sp, const torch::Tensor& sc) {
        auto x = sp.to(torch::kFloat32).view({sp.size(0), fe::ACTOR_CH, fe::PATCH, fe::PATCH}).mul_(ch_scale);
        auto y = actor_features(sc.to(torch::kFloat32));
        return actors[p]->as<Actor>()->forward(x, y);
    }
    torch::Tensor value_split(const torch::Tensor& sp, const torch::Tensor& sc) {
        auto x = sp.to(torch::kFloat32).view({sp.size(0), fe::CRITIC_CH, fe::CRITIC_PAD, fe::CRITIC_PAD});
        auto y = critic_features(sc.to(torch::kFloat32));
        return critic->forward(x, y);
    }
    // raw: [B, CRITIC_SPATIAL + CRITIC_SCALARS]
    torch::Tensor value(torch::Tensor raw) {
        auto sp = raw.narrow(1, 0, fe::CRITIC_SPATIAL).view({raw.size(0), fe::CRITIC_CH, fe::CRITIC_PAD, fe::CRITIC_PAD});
        auto sc = critic_features(raw.narrow(1, fe::CRITIC_SPATIAL, fe::CRITIC_SCALARS));
        return critic->forward(sp, sc);
    }
};
TORCH_MODULE(Teacher);

}  // namespace florent_rl
