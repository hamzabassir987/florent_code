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
#include "trunk_fn.hpp"

namespace florent_rl {

constexpr int ACTOR_HIDDEN = 128;
// The fused CUDA trunk (trunk_kernels.cu) on CUDA inputs; env FUSED_TRUNK=0
// uses the PyTorch ops instead.
inline bool use_fused_trunk() {
    static const bool on = fused_trunk_built() && (!std::getenv("FUSED_TRUNK") || std::atoi(std::getenv("FUSED_TRUNK")) != 0);
    return on;
}
// The same for the critic's spatial part (critic_kernels.cu); env FUSED_CRITIC=0.
inline bool use_fused_critic() {
    static const bool on = fused_trunk_built() && (!std::getenv("FUSED_CRITIC") || std::atoi(std::getenv("FUSED_CRITIC")) != 0);
    return on;
}
constexpr int TRUNK_CH = 32;

// Fixed input scaling (part of the checkpoint ABI; the deployed student uses
// its own integer input map).
inline torch::Tensor actor_channel_scale() {
    std::vector<float> s(fe::ACTOR_CH, 1.f);
    s[fe::C_STALE] = s[fe::C_VISIT] = 1.f / 255.f;
    s[fe::C_ATTR_STALE] = s[fe::C_ATTR_STALE + 1] = 1.f / 255.f;
    s[fe::C_HP] = s[fe::C_HP + 1] = 1.f / 100.f;
    for (int k = 0; k < fe::CH_RAY; k++) s[fe::C_RAY + k] = 1.f / 32.f;
    s[fe::C_PATH] = 1.f / 8.f;                           // planned conveyor direction (Dir 1..8)
    s[fe::C_HIT] = 1.f / fe::HIT_CAP;                    // rounds since seen losing HP (cap 10)
    return torch::tensor(s).view({1, fe::ACTOR_CH, 1, 1});
}
inline torch::Tensor actor_scalar_scale() {
    std::vector<float> s(fe::ACTOR_SCALARS, 0.f);
    for (int i = 0; i <= 5; i++) s[i] = 1.f;            // type one-hot
    s[6] = 1.f / 100.f;                                  // own hp %
    s[7] = s[8] = 1.f / 4.f;                             // cooldowns
    s[9] = 1.f / 1000.f;                                 // round
    s[10] = s[11] = s[12] = 1.f / 1000.f;                // Ti, Vo, ammo
    s[13] = 1.f / 100.f;                                 // scale percent
    s[14] = 1.f / 50.f;                                  // unit count
    s[15] = 1.f / 15.f;                                  // rounds since our core's under-attack alarm (cap 15)
    for (int i = fe::ACTOR_COSTS; i < fe::ACTOR_COSTS + 10; i++) s[i] = 1.f / 50.f;   // current prices
    for (int i = 36; i <= 39; i++) s[i] = 1.f;           // turret/core reach flags
    // 15-19 (opponent, always 0), 20-35 (packed store words): ignored
    return torch::tensor(s).view({1, fe::ACTOR_SCALARS});
}
// Scalars that can reach ~1000 enter at several scales: x becomes
// clamp(x / d, -SCALAR_CLIP, SCALAR_CLIP) for each d, so a difference of 3 Ti
// shows at the fine scales and 400 vs 4000 stays apart at the coarse ones.
// Ti and Vo (stock, collected, differences) also get /10^4 and /10^5. Every
// other scalar is one feature at its fixed scale.
constexpr float SCALAR_CLIP = 2.f;
constexpr int N_BIG = 4, N_HUGE = 6;
inline torch::Tensor scales_of(int n) {                       // [1, 1, n]: 1, 1/10, 1/100, ...
    std::vector<float> s; float d = 1.f;
    for (int i = 0; i < n; i++, d *= 10.f) s.push_back(1.f / d);
    return torch::tensor(s).view({1, 1, n});
}
// actor: round, ammo (x4) | Ti, Vo (x6)
inline std::vector<int64_t> actor_big() { return {9, 12}; }
inline std::vector<int64_t> actor_huge() { return {10, 11}; }
// critic: round, ammo and scale per team, rounds left (x4) | Ti, Vo per team,
// Ti / Vo collected per team, Ti collected A - B, Ti stored A - B (x6)
inline std::vector<int64_t> critic_big() {
    const int e = fe::CRITIC_SCALARS - fe::CRITIC_EXTRA;
    return {0, 3, 4, 8, 9, e};
}
inline std::vector<int64_t> critic_huge() {
    const int e = fe::CRITIC_SCALARS - fe::CRITIC_EXTRA;
    return {1, 2, 6, 7, e + 1, e + 2, e + 3, e + 4, e + 5, e + 6};
}
// The single-scale scalars: (index, scale). Scalars with scale 0 are dropped.
inline void single_scalars(const torch::Tensor& scale, const std::vector<int64_t>& multi,
                           std::vector<int64_t>& idx, std::vector<float>& sc) {
    auto s = scale.flatten().contiguous();
    for (int64_t i = 0; i < s.numel(); i++) {
        const float v = s[i].item<float>();
        if (v == 0.f || std::find(multi.begin(), multi.end(), i) != multi.end()) continue;
        idx.push_back(i); sc.push_back(v);
    }
}
inline torch::Tensor critic_single_scale() {
    std::vector<float> s(fe::CRITIC_SCALARS, 1.f / 10.f);    // entity counts per team and type
    s[5] = s[10] = 1.f / 50.f;                               // units per team
    return torch::tensor(s);
}

// raw scalars [B, N] -> features: the single ones scaled, then big (x4), then huge (x6)
struct ScalarFeatures {
    torch::Tensor single_idx, single_scale, big_idx, huge_idx;
    int count = 0;
    static ScalarFeatures make(const torch::Tensor& scale, const std::vector<int64_t>& big, const std::vector<int64_t>& huge) {
        std::vector<int64_t> multi = big; multi.insert(multi.end(), huge.begin(), huge.end());
        std::vector<int64_t> i; std::vector<float> v;
        single_scalars(scale, multi, i, v);
        ScalarFeatures f;
        f.single_idx = torch::tensor(i, torch::kLong);
        f.single_scale = torch::tensor(v).view({1, -1});
        f.big_idx = torch::tensor(big, torch::kLong);
        f.huge_idx = torch::tensor(huge, torch::kLong);
        f.count = int(i.size() + big.size() * N_BIG + huge.size() * N_HUGE);
        return f;
    }
};
inline int actor_features() { return ScalarFeatures::make(actor_scalar_scale(), actor_big(), actor_huge()).count; }
inline int critic_features() { return ScalarFeatures::make(critic_single_scale(), critic_big(), critic_huge()).count; }

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
    // The same block without padding: each depthwise conv only produces the
    // positions whose 3x3 window lies inside its input, so [B, C, S, S] ->
    // [B, C, S-4, S-4] and the residual is the input's centre crop. Inside
    // that window every value equals forward()'s (zero padding only reaches
    // the outer ring of each conv), so a trunk read at its centre is
    // unchanged: the actor computes only what its centre output depends on.
    torch::Tensor forward_valid(torch::Tensor x) {
        static const std::vector<int64_t> one{1, 1}, zero{0, 0};
        const at::IntArrayRef ONE(one), ZERO(zero);
        const int64_t c = x.size(1), s = x.size(2);
        auto z = pw->forward(x).relu_();
        z = torch::conv2d(z, dw->weight, dw->bias, ONE, ZERO, ONE, c).relu_();
        z = pw2->forward(z).relu_();
        z = torch::conv2d(z, dw2->weight, dw2->bias, ONE, ZERO, ONE, c);
        return z.add_(x.narrow(2, 2, s - 4).narrow(3, 2, s - 4)).relu_();
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
    // spatial: [B, ACTOR_CH, 17, 17] already scaled; scalars: [B, actor_features()]
    torch::Tensor forward(torch::Tensor spatial, torch::Tensor scalars) {
        return head(trunk_centre(stem->forward(spatial).relu_()), scalars);
    }
    // spatial unscaled: the per-channel input scale is folded into the stem
    // weights (one small multiply instead of a pass over the whole input)
    torch::Tensor forward_unscaled(torch::Tensor spatial, const torch::Tensor& ch_scale, torch::Tensor scalars) {
        auto w = stem->weight * ch_scale.view({1, -1, 1, 1});
        return head(trunk_centre(torch::conv2d(spatial, w, stem->bias).relu_()), scalars);
    }
    // The four blocks reach exactly HALF = 8 tiles (two 3x3 convs each) and only
    // the centre is read, so they run unpadded on shrinking windows:
    // 17 -> 13 -> 9 -> 5 -> 1 (see ResBlock::forward_valid). Same output as
    // the padded trunk's centre, ~2.5x less work.
    static_assert(fe::PATCH == 17, "trunk_centre assumes a 17x17 patch and 4 blocks");
    torch::Tensor trunk_centre(torch::Tensor h) {
        for (auto& b : *blocks) h = b->as<ResBlock>()->forward_valid(h);
        return h.flatten(1);                                     // [B, 32, 1, 1] -> [B, 32]
    }
    torch::Tensor head(const torch::Tensor& centre, const torch::Tensor& scalars) {
        auto f = fc->forward(torch::cat({centre, scalars}, 1)).relu_();
        return out->forward(f);
    }
    // Every trunk parameter as one vector, in trunk_kernels.cu's layout (built
    // with differentiable ops: the flat gradient flows back to the parameters).
    // The stem weight includes the per-channel input scale.
    torch::Tensor trunk_flat(const torch::Tensor& ch_scale) {
        std::vector<torch::Tensor> v;
        v.push_back((stem->weight.view({TRUNK_CH, fe::ACTOR_CH}) * ch_scale.view({1, fe::ACTOR_CH})).t().reshape(-1));
        v.push_back(stem->bias);
        for (auto& m : *blocks) {
            auto* b = m->as<ResBlock>();
            auto pw = b->pw->weight.view({TRUNK_CH, TRUNK_CH}), pw2 = b->pw2->weight.view({TRUNK_CH, TRUNK_CH});
            v.push_back(pw.t().reshape(-1)); v.push_back(pw.reshape(-1)); v.push_back(b->pw->bias);
            v.push_back(b->dw->weight.reshape(-1)); v.push_back(b->dw->bias);
            v.push_back(pw2.t().reshape(-1)); v.push_back(pw2.reshape(-1)); v.push_back(b->pw2->bias);
            v.push_back(b->dw2->weight.reshape(-1)); v.push_back(b->dw2->bias);
        }
        return torch::cat(v);
    }
    // reference: the padded trunk read at its centre (parity checks)
    torch::Tensor forward_padded(torch::Tensor spatial, torch::Tensor scalars) {
        auto h = stem->forward(spatial).relu_();
        for (auto& b : *blocks) h = b->as<ResBlock>()->forward(h);
        auto centre = h.index({torch::indexing::Slice(), torch::indexing::Slice(), fe::HALF, fe::HALF});
        return head(centre, scalars);
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
        return head(spatial_out(spatial), scalars);
    }
    torch::Tensor spatial_out(torch::Tensor spatial) {             // [B, 29, 48, 48] -> [B, 288]
        static const int POOL[3] = {2, 2, 4};
        auto h = stem->forward(spatial).relu_();
        for (int b = 0; b < 3; b++)
            h = torch::max_pool2d(blocks[b]->as<ResBlock>()->forward(h), {POOL[b], POOL[b]}, {POOL[b], POOL[b]});
        return h.flatten(1);
    }
    torch::Tensor head(const torch::Tensor& spatial, const torch::Tensor& scalars) {
        auto f = hidden->forward(torch::cat({spatial, scalars}, 1)).relu_();
        return value->forward(f).squeeze(1);
    }
    // the spatial parameters as one vector, in critic_kernels.cu's layout
    torch::Tensor spatial_flat() {
        std::vector<torch::Tensor> v;
        v.push_back(stem->weight.view({TRUNK_CH, fe::CRITIC_CH}).t().reshape(-1));
        v.push_back(stem->bias);
        for (auto& m : *blocks) {
            auto* b = m->as<ResBlock>();
            auto pw = b->pw->weight.view({TRUNK_CH, TRUNK_CH}), pw2 = b->pw2->weight.view({TRUNK_CH, TRUNK_CH});
            v.push_back(pw.t().reshape(-1)); v.push_back(pw.reshape(-1)); v.push_back(b->pw->bias);
            v.push_back(b->dw->weight.reshape(-1)); v.push_back(b->dw->bias);
            v.push_back(pw2.t().reshape(-1)); v.push_back(pw2.reshape(-1)); v.push_back(b->pw2->bias);
            v.push_back(b->dw2->weight.reshape(-1)); v.push_back(b->dw2->bias);
        }
        return torch::cat(v);
    }
};
TORCH_MODULE(Critic);

struct TeacherImpl : torch::nn::Module {
    torch::nn::ModuleList actors;
    Critic critic{nullptr};
    torch::Tensor ch_scale, scales4, scales6;
    torch::Tensor a_idx, a_scale, a_big, a_huge, c_idx, c_scale, c_big, c_huge;   // scalar feature maps (buffers)
    TeacherImpl() {
        for (int p = 0; p < NUM_POLICIES; p++) actors->push_back(Actor(p));
        register_module("actors", actors);
        critic = register_module("critic", Critic());
        ch_scale = register_buffer("ch_scale", actor_channel_scale());
        auto af = ScalarFeatures::make(actor_scalar_scale(), actor_big(), actor_huge());
        auto cf = ScalarFeatures::make(critic_single_scale(), critic_big(), critic_huge());
        a_idx = register_buffer("a_idx", af.single_idx); a_scale = register_buffer("a_scale", af.single_scale);
        a_big = register_buffer("a_big", af.big_idx); a_huge = register_buffer("a_huge", af.huge_idx);
        c_idx = register_buffer("c_idx", cf.single_idx); c_scale = register_buffer("c_scale", cf.single_scale);
        c_big = register_buffer("c_big", cf.big_idx); c_huge = register_buffer("c_huge", cf.huge_idx);
        scales4 = register_buffer("scales4", scales_of(N_BIG));
        scales6 = register_buffer("scales6", scales_of(N_HUGE));
    }
    // raw: [B, ACTOR_SPATIAL + ACTOR_SCALARS] unscaled encoder output
    torch::Tensor logits(int p, torch::Tensor raw) {
        auto sp = raw.narrow(1, 0, fe::ACTOR_SPATIAL).view({raw.size(0), fe::ACTOR_CH, fe::PATCH, fe::PATCH}) * ch_scale;
        auto sc = actor_features(raw.narrow(1, fe::ACTOR_SPATIAL, fe::ACTOR_SCALARS));
        return actors[p]->as<Actor>()->forward(sp, sc);
    }
    // [B, CRITIC_SCALARS] raw -> [B, CRITIC_FEATURES]: each scalar at every scale, clamped
    torch::Tensor features(const torch::Tensor& sc, const torch::Tensor& idx, const torch::Tensor& scale,
                           const torch::Tensor& big, const torch::Tensor& huge) {
        auto single = sc.index_select(1, idx) * scale;
        auto b = (sc.index_select(1, big).unsqueeze(2) * scales4).clamp_(-SCALAR_CLIP, SCALAR_CLIP).flatten(1);
        auto h = (sc.index_select(1, huge).unsqueeze(2) * scales6).clamp_(-SCALAR_CLIP, SCALAR_CLIP).flatten(1);
        return torch::cat({single, b, h}, 1);
    }
    torch::Tensor critic_features(const torch::Tensor& sc) { return features(sc, c_idx, c_scale, c_big, c_huge); }
    torch::Tensor actor_features(const torch::Tensor& sc) { return features(sc, a_idx, a_scale, a_big, a_huge); }
    // Compact inputs, already on the device: spatial uint8 [B, ACTOR_SPATIAL],
    // scalars any integer type [B, ACTOR_SCALARS]; converted to float here.
    // The trunk's flat parameter vector for no-grad forwards (rollouts):
    // kept per policy and rebuilt, in place, only when a trunk parameter has
    // changed (version counters and storage, so optimizer steps, copies and
    // loads are all seen). In place: a captured CUDA graph that reads it stays
    // valid. Call it before replaying such a graph.
    std::vector<torch::Tensor> flat_inf = std::vector<torch::Tensor>(NUM_POLICIES);
    std::vector<std::vector<std::pair<const void*, int64_t>>> flat_key = std::vector<std::vector<std::pair<const void*, int64_t>>>(NUM_POLICIES);
    const torch::Tensor& inference_flat(int p) {
        auto a = actors[p]->as<Actor>();
        std::vector<std::pair<const void*, int64_t>> key;
        for (auto& t : a->stem->parameters()) key.push_back({t.data_ptr(), t._version()});
        for (auto& t : a->blocks->parameters()) key.push_back({t.data_ptr(), t._version()});
        key.push_back({ch_scale.data_ptr(), ch_scale._version()});
        if (!flat_inf[p].defined() || key != flat_key[p]) {
            torch::NoGradGuard ng;
            auto f = a->trunk_flat(ch_scale);
            if (!flat_inf[p].defined() || flat_inf[p].device() != f.device()) {
                c10::InferenceMode off(false);       // a normal tensor, updated in place later
                flat_inf[p] = f.clone();
            } else flat_inf[p].copy_(f);
            flat_key[p] = std::move(key);
        }
        return flat_inf[p];
    }
    torch::Tensor logits_split(int p, const torch::Tensor& sp, const torch::Tensor& sc) {
        if (sp.is_cuda() && use_fused_trunk()) {
            auto a = actors[p]->as<Actor>();
            auto y = actor_features(sc.to(torch::kFloat32));
            const bool grad = torch::GradMode::is_enabled();
            return a->head(fused_trunk(sp.contiguous(), grad ? a->trunk_flat(ch_scale) : inference_flat(p)), y);
        }
        auto x = sp.to(torch::kFloat32).view({sp.size(0), fe::ACTOR_CH, fe::PATCH, fe::PATCH});
        auto y = actor_features(sc.to(torch::kFloat32));
        return actors[p]->as<Actor>()->forward_unscaled(x, ch_scale, y);
    }
    // the critic's flat spatial parameters for no-grad forwards (see inference_flat)
    torch::Tensor cflat_inf;
    std::vector<std::pair<const void*, int64_t>> cflat_key;
    const torch::Tensor& critic_inference_flat() {
        std::vector<std::pair<const void*, int64_t>> key;
        for (auto& t : critic->stem->parameters()) key.push_back({t.data_ptr(), t._version()});
        for (auto& t : critic->blocks->parameters()) key.push_back({t.data_ptr(), t._version()});
        if (!cflat_inf.defined() || key != cflat_key) {
            torch::NoGradGuard ng;
            auto f = critic->spatial_flat();
            if (!cflat_inf.defined() || cflat_inf.device() != f.device()) { c10::InferenceMode off(false); cflat_inf = f.clone(); }
            else cflat_inf.copy_(f);
            cflat_key = std::move(key);
        }
        return cflat_inf;
    }
    torch::Tensor value_split(const torch::Tensor& sp, const torch::Tensor& sc) {
        if (sp.is_cuda() && use_fused_critic()) {
            auto y = critic_features(sc.to(torch::kFloat32));
            const bool grad = torch::GradMode::is_enabled();
            return critic->head(fused_critic(sp.contiguous(), grad ? critic->spatial_flat() : critic_inference_flat()), y);
        }
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
