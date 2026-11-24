// Integer-exact int4 student (distilled from the float teacher; deployed as
// SWAR big-integer code in bots/econ/student.py, which must match bit for bit).
//
// Same architecture as the teacher actor: stem 1x1 -> 4 x [1x1 -> dw3x3 ->
// residual] -> centre ++ scalars -> hidden -> factorized head logits.
// Every value in the forward pass is an integer (stored exactly in float32):
//   inputs:        0..15 via a fixed per-channel map (quantize_actor)
//   conv weights:  int4 (-7..7), biases integer (-127..127)
//   conv output:   clamp(floor(acc / 2^q), 0, 15)
//   block output:  clamp(floor((dw_acc + (h << q)) / 2^q), 0, 15) after every 2nd layer
//   head:          int8 weights (-127..127); hidden like a conv output;
//                  logits = integer accumulator / 16
// Training uses latent float weights with straight-through rounding.
#pragma once
#include <torch/torch.h>

#include "../../engine/src/obs.hpp"
#include "../sim/florent_heads.hpp"

namespace florent_rl {

constexpr int ST_CH = 32, ST_HIDDEN = 128, ST_LOGIT_SHIFT = 4;
// 4 residual blocks x 2 (1x1 -> dw3x3) layers = 8 layers; residual after odd layers
constexpr int ST_LAYERS = 8, ST_NSHIFT = 2 + 2 * ST_LAYERS, ST_FC_SHIFT = ST_NSHIFT - 1;

// raw encoder ints -> 0..15 (identical to bots/econ/student.py quantize())
inline void quantize_actor(const int16_t* raw, int32_t* q) {
    using namespace fe;
    const int P2 = PATCH * PATCH;
    for (int c = 0; c < ACTOR_CH; c++) {
        for (int i = 0; i < P2; i++) {
            int v = raw[c * P2 + i], o;
            if (c == C_STALE || c == C_VISIT || c == C_ATTR_STALE || c == C_ATTR_STALE + 1) o = v >> 4;
            else if (c == C_HP || c == C_HP + 1) o = v * 15 / 100;
            else if (c >= C_RAY) o = (c - C_RAY) <= 1 ? (v + 1) >> 1 : v;
            else o = v ? 15 : 0;
            q[c * P2 + i] = o < 0 ? 0 : (o > 15 ? 15 : o);
        }
    }
    const int16_t* s = raw + ACTOR_SPATIAL;
    int32_t* qs = q + ACTOR_SPATIAL;
    for (int i = 0; i < ACTOR_SCALARS; i++) {
        int v = s[i], o = 0;
        if (i <= 5 || (i >= 36 && i <= 39)) o = v ? 15 : 0;
        else if (i == 6) o = v * 15 / 100;
        else if (i == 7 || i == 8) o = v * 3;
        else if (i == 9) o = v / 67;
        else if (i == 10 || i == 11) o = v / 100;
        else if (i == 12) o = v / 50;
        else if (i == 14) o = v / 4;
        else if (i == 15) o = v;                               // core alarm age (0..15)
        else if (i == 13) o = (v - 100) / 25;                 // scale percent: 100 -> 0, +25 % a step
        else if (i >= fe::ACTOR_COSTS) o = v / 4;              // current prices
        qs[i] = o < 0 ? 0 : (o > 15 ? 15 : o);
    }
}

// straight-through helpers
inline torch::Tensor ste_round(const torch::Tensor& x) { return x + (torch::round(x) - x).detach(); }
inline torch::Tensor ste_floor(const torch::Tensor& x) { return x + (torch::floor(x) - x).detach(); }
inline torch::Tensor qw(const torch::Tensor& w, double lim) { return ste_round(torch::clamp(w, -lim, lim)); }
inline torch::Tensor act(const torch::Tensor& acc, int q) {          // clamp(floor(acc/2^q), 0, 15)
    return torch::clamp(ste_floor(acc / double(1 << q)), 0.0, 15.0);
}

struct StudentActorImpl : torch::nn::Module {
    int policy, nlogits;
    torch::Tensor stem_w, stem_b, fc_w, fc_b, out_w, out_b;
    std::vector<torch::Tensor> pw_w, pw_b, dw_w, dw_b;
    // shifts: stem, then per layer (pw, dw), then fc
    std::vector<int64_t> shifts;
    explicit StudentActorImpl(int p) : policy(p), nlogits(policy_logit_count(p)) {
        auto u = [](std::vector<int64_t> shape, double a) { return torch::empty(shape).uniform_(-a, a); };
        stem_w = register_parameter("stem_w", u({ST_CH, fe::ACTOR_CH, 1, 1}, 3));
        stem_b = register_parameter("stem_b", torch::zeros({ST_CH}));
        for (int b = 0; b < ST_LAYERS; b++) {
            pw_w.push_back(register_parameter("pw_w" + std::to_string(b), u({ST_CH, ST_CH, 1, 1}, 3)));
            pw_b.push_back(register_parameter("pw_b" + std::to_string(b), torch::zeros({ST_CH})));
            dw_w.push_back(register_parameter("dw_w" + std::to_string(b), u({ST_CH, 1, 3, 3}, 3)));
            dw_b.push_back(register_parameter("dw_b" + std::to_string(b), torch::zeros({ST_CH})));
        }
        fc_w = register_parameter("fc_w", u({ST_HIDDEN, ST_CH + fe::ACTOR_SCALARS}, 20));
        fc_b = register_parameter("fc_b", torch::zeros({ST_HIDDEN}));
        out_w = register_parameter("out_w", u({nlogits, ST_HIDDEN}, 20));
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
        auto acc = torch::conv2d(x, qw(stem_w, 7), qw(stem_b, 127));
        rec(acc);
        auto h = act(acc, int(shifts[0]));
        torch::Tensor hb = h;                      // block input (residual source)
        for (int b = 0; b < ST_LAYERS; b++) {
            auto a1 = torch::conv2d(h, qw(pw_w[b], 7), qw(pw_b[b], 127));
            rec(a1);
            auto t = act(a1, int(shifts[1 + 2 * b]));
            auto a2 = torch::conv2d(t, qw(dw_w[b], 7), qw(dw_b[b], 127), 1, 1, 1, ST_CH);
            int qd = int(shifts[2 + 2 * b]);
            rec(a2);
            if (b % 2) { h = act(a2 + hb * double(1 << qd), qd); hb = h; }
            else h = act(a2, qd);
        }
        auto centre = h.index({torch::indexing::Slice(), torch::indexing::Slice(), fe::HALF, fe::HALF});
        auto f = torch::cat({centre, sc}, 1);
        auto a3 = torch::nn::functional::linear(f, qw(fc_w, 127), qw(fc_b, 127));
        rec(a3);
        auto hid = act(a3, int(shifts[ST_FC_SHIFT]));
        return torch::nn::functional::linear(hid, qw(out_w, 127), qw(out_b, 127 * 16));
    }
    // choose each shift so that ~99% of positive accumulators land within 0..15
    void calibrate(torch::Tensor q) {
        torch::NoGradGuard ng;
        for (int pass = 0; pass < ST_NSHIFT; pass++) {      // layer by layer (each depends on earlier shifts)
            std::vector<torch::Tensor> accs;
            forward(q, &accs);
            auto& a = accs[pass];
            auto pos = a.flatten();
            pos = pos.index({pos > 0});
            double p99 = pos.numel() ? std::get<0>(pos.sort()).index({long(pos.numel() * 0.99)}).item<double>() : 15;
            int s = 0;
            while ((p99 / double(1 << s)) > 15 && s < 12) s++;
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
