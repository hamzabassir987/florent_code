// Integer-exact int8 student (distilled from the float teacher; deployed as
// SWAR big-integer code in bots/econ/student.py, which must match bit for bit).
//
// Same architecture as the teacher actor: stem 1x1 -> 4 x [1x1 -> dw3x3 ->
// residual] -> centre ++ scalars -> hidden -> factorized head logits.
// Every value in the forward pass is an integer (stored exactly in float32):
//   inputs:        0..255 via a fixed per-channel map (quantize_actor)
//   weights:       int8 (-127..127) everywhere; biases integer at accumulator
//                  scale (|b| <= ST_BIAS_MAX)
//   conv output:   clamp(floor(acc / 2^q), 0, 255)
//   block output:  clamp(floor((dw_acc + (h << q)) / 2^q), 0, 255) after every 2nd layer
//   hidden:        like a conv output; logits = integer accumulator / 2^ST_LOGIT_SHIFT
// Training uses latent float weights (x ST_WSCALE / x ST_BSCALE) with
// straight-through rounding.
#pragma once
#include <torch/torch.h>

#include <cstdlib>

#include "../../engine/src/obs.hpp"
#include "../sim/florent_heads.hpp"

namespace florent_rl {

// trunk width: env ST_CH (default 16). 8 channels: ~4x cheaper 1x1 layers, the
// Python forward fits the server's 10 ms turn (16 channels: ~8 ms on the server)
inline const int ST_CH = [] { const char* v = std::getenv("ST_CH"); return v ? std::atoi(v) : 16; }();
constexpr int ST_HIDDEN = 128, ST_LOGIT_SHIFT = 12;
constexpr int ST_AMAX = 255;                 // activations and inputs: 0..255
// the sentinel gate (a separate P_SENTINEL network in the teacher) is merged into
// the student's builder network as 2 extra logits after the builder heads
// (0 no, 1 place): one forward per builder turn in the deployed bot
constexpr int ST_GATE = 2;
inline int student_logit_count(int p) { return policy_logit_count(p) + (p == P_BUILDER ? ST_GATE : 0); }
constexpr double ST_WMAX = 127, ST_WSCALE = 32, ST_BSCALE = 1024, ST_BIAS_MAX = 1 << 22;
// 4 residual blocks x 2 (1x1 -> dw3x3) layers = 8 layers; residual after odd layers
constexpr int ST_LAYERS = 8, ST_NSHIFT = 2 + 2 * ST_LAYERS, ST_FC_SHIFT = ST_NSHIFT - 1;

// raw encoder ints -> 0..255 (identical to bots/econ/student.py quantize()); integer
// divisions truncate, every result is clamped to 0..255
inline void quantize_actor(const int16_t* raw, int32_t* q) {
    using namespace fe;
    const int P2 = PATCH * PATCH;
    for (int c = 0; c < ACTOR_CH; c++) {
        for (int i = 0; i < P2; i++) {
            int v = raw[c * P2 + i], o;
            if (c == C_STALE || c == C_VISIT || c == C_ATTR_STALE || c == C_ATTR_STALE + 1) o = v;   // ages 0..255
            else if (c == C_HP || c == C_HP + 1) o = v * 255 / 100;                                 // HP percent
            else if (c == C_HIT) o = v * 25;                                                        // 0..10
            else if (c == C_PATH) o = v * 32;                                                       // Dir 1..8
            else if (c >= C_RAY && c < C_RAY + CH_RAY) o = v * 8;                                   // rays 0..32
            else o = v ? 255 : 0;                                                                   // one-hot
            q[c * P2 + i] = o < 0 ? 0 : (o > ST_AMAX ? ST_AMAX : o);
        }
    }
    const int16_t* s = raw + ACTOR_SPATIAL;
    int32_t* qs = q + ACTOR_SPATIAL;
    for (int i = 0; i < ACTOR_SCALARS; i++) {
        int v = s[i], o = 0;
        if (i <= 5 || (i >= 36 && i <= 39)) o = v ? 255 : 0;          // type one-hot, reach flags
        else if (i == 6) o = v * 255 / 100;                            // HP percent
        else if (i == 7 || i == 8) o = v * 32;                         // cooldowns
        else if (i == 9) o = v / 4;                                    // round
        else if (i == 10 || i == 11 || i == 16) o = v / 4;             // our Ti, Vo; enemy Vo
        else if (i == 12 || i == 17) o = v / 2;                        // ammo
        else if (i == 13 || i == 18) o = (v - 100) / 2;                // price scale percent
        else if (i == 14 || i == 19) o = v * 4;                        // unit counts
        else if (i == 15) o = v * 16;                                  // core alarm age 0..15
        else if (i >= fe::ACTOR_COSTS) o = v / 2;                      // current prices
        qs[i] = o < 0 ? 0 : (o > ST_AMAX ? ST_AMAX : o);              // 20..35 (store words): 0
    }
}

// straight-through helpers
inline torch::Tensor ste_round(const torch::Tensor& x) { return x + (torch::round(x) - x).detach(); }
inline torch::Tensor ste_floor(const torch::Tensor& x) { return x + (torch::floor(x) - x).detach(); }
inline torch::Tensor qw(const torch::Tensor& w) { return ste_round(torch::clamp(w * ST_WSCALE, -ST_WMAX, ST_WMAX)); }        // int8 weight
inline torch::Tensor qb(const torch::Tensor& b) { return ste_round(torch::clamp(b * ST_BSCALE, -ST_BIAS_MAX, ST_BIAS_MAX)); } // integer bias
inline torch::Tensor act(const torch::Tensor& acc, int q) {          // clamp(floor(acc/2^q), 0, 255)
    return torch::clamp(ste_floor(acc / double(1 << q)), 0.0, double(ST_AMAX));
}

struct StudentActorImpl : torch::nn::Module {
    int policy, nlogits;
    torch::Tensor stem_w, stem_b, fc_w, fc_b, out_w, out_b;
    std::vector<torch::Tensor> pw_w, pw_b, dw_w, dw_b;
    // shifts: stem, then per layer (pw, dw), then fc
    std::vector<int64_t> shifts;
    explicit StudentActorImpl(int p) : policy(p), nlogits(student_logit_count(p)) {
        auto u = [](std::vector<int64_t> shape, double a) { return torch::empty(shape).uniform_(-a, a); };
        stem_w = register_parameter("stem_w", u({ST_CH, fe::ACTOR_CH, 1, 1}, 0.5));
        stem_b = register_parameter("stem_b", torch::zeros({ST_CH}));
        for (int b = 0; b < ST_LAYERS; b++) {
            pw_w.push_back(register_parameter("pw_w" + std::to_string(b), u({ST_CH, ST_CH, 1, 1}, 0.5)));
            pw_b.push_back(register_parameter("pw_b" + std::to_string(b), torch::zeros({ST_CH})));
            dw_w.push_back(register_parameter("dw_w" + std::to_string(b), u({ST_CH, 1, 3, 3}, 0.5)));
            dw_b.push_back(register_parameter("dw_b" + std::to_string(b), torch::zeros({ST_CH})));
        }
        fc_w = register_parameter("fc_w", u({ST_HIDDEN, ST_CH + fe::ACTOR_SCALARS}, 0.6));
        fc_b = register_parameter("fc_b", torch::zeros({ST_HIDDEN}));
        out_w = register_parameter("out_w", u({nlogits, ST_HIDDEN}, 0.6));
        out_b = register_parameter("out_b", torch::zeros({nlogits}));
        shifts = std::vector<int64_t>(ST_NSHIFT, 4);
        register_buffer("shifts_t", torch::zeros({ST_NSHIFT}, torch::kLong));
    }
    void sync_shifts_to_buffer() { auto b = named_buffers()["shifts_t"]; for (int i = 0; i < ST_NSHIFT; i++) b[i] = shifts[i]; }
    void sync_shifts_from_buffer() { auto b = named_buffers()["shifts_t"]; for (int i = 0; i < ST_NSHIFT; i++) shifts[i] = b[i].item<int64_t>(); }

    // q: quantized input [B, ACTOR_SPATIAL + ACTOR_SCALARS] (0..15).
    // Returns integer logit accumulators [B, nlogits] (divide by 16 for softmax).
    // If `accs` is given, records each layer's pre-shift accumulator (calibration).
    torch::Tensor forward(torch::Tensor q, std::vector<torch::Tensor>* accs = nullptr) {
        long B = q.size(0);
        auto x = q.narrow(1, 0, fe::ACTOR_SPATIAL).view({B, fe::ACTOR_CH, fe::PATCH, fe::PATCH});
        auto sc = q.narrow(1, fe::ACTOR_SPATIAL, fe::ACTOR_SCALARS);
        auto rec = [&](const torch::Tensor& a) { if (accs) accs->push_back(a.detach()); };
        auto acc = torch::conv2d(x, qw(stem_w), qb(stem_b));
        rec(acc);
        auto h = act(acc, int(shifts[0]));
        torch::Tensor hb = h;                      // block input (residual source)
        for (int b = 0; b < ST_LAYERS; b++) {
            auto a1 = torch::conv2d(h, qw(pw_w[b]), qb(pw_b[b]));
            rec(a1);
            auto t = act(a1, int(shifts[1 + 2 * b]));
            auto a2 = torch::conv2d(t, qw(dw_w[b]), qb(dw_b[b]), 1, 1, 1, ST_CH);
            int qd = int(shifts[2 + 2 * b]);
            rec(a2);
            if (b % 2) { h = act(a2 + hb * double(1 << qd), qd); hb = h; }
            else h = act(a2, qd);
        }
        auto centre = h.index({torch::indexing::Slice(), torch::indexing::Slice(), fe::HALF, fe::HALF});
        auto f = torch::cat({centre, sc}, 1);
        auto a3 = torch::nn::functional::linear(f, qw(fc_w), qb(fc_b));
        rec(a3);
        auto hid = act(a3, int(shifts[ST_FC_SHIFT]));
        return torch::nn::functional::linear(hid, qw(out_w), qb(out_b));
    }
    // choose each shift so that ~99% of positive accumulators land within 0..255
    void calibrate(torch::Tensor q) {
        torch::NoGradGuard ng;
        for (int pass = 0; pass < ST_NSHIFT; pass++) {      // layer by layer (each depends on earlier shifts)
            std::vector<torch::Tensor> accs;
            forward(q, &accs);
            auto& a = accs[pass];
            auto pos = a.flatten();
            pos = pos.index({pos > 0});
            double p99 = pos.numel() ? std::get<0>(pos.sort()).index({long(pos.numel() * 0.99)}).item<double>() : ST_AMAX;
            int s = 0;
            while ((p99 / double(1 << s)) > ST_AMAX && s < 20) s++;
            if (pass >= 1 && pass < ST_FC_SHIFT && pass % 4 == 0) s = std::max(s, 1);  // dw: room for residual
            shifts[pass] = s;
        }
        sync_shifts_to_buffer();
    }
};
TORCH_MODULE(StudentActor);

struct StudentImpl : torch::nn::Module {
    torch::nn::ModuleList actors;
    StudentImpl() {
        for (int p = 0; p < NUM_POLICIES; p++) actors->push_back(StudentActor(p));
        register_module("actors", actors);
    }
    StudentActorImpl* a(int p) { return actors[p]->as<StudentActorImpl>(); }
};
TORCH_MODULE(Student);

}  // namespace florent_rl
