// Local testing only: the float teacher as a C library, so bots/econ can play
// with it (FLORENT_TEACHER=<model.pt>, FLORENT_TEACHER_LIB=<libteacher_capi.so>)
// instead of the int8 student. CPU libtorch; inputs clamped as in neural_train
// (spatial 0..255, scalars int16).
#include <torch/torch.h>
#include "../train/model.hpp"
using namespace florent_rl;

static Teacher* g_net = nullptr;

extern "C" {
int teacher_load(const char* path) {
    torch::set_num_threads(1);
    g_net = new Teacher();
    torch::load(*g_net, path, torch::Device(torch::kCPU));
    (*g_net)->eval();
    return 0;
}
int teacher_logit_count(int p) { return policy_logit_count(p); }
// raw: ACTOR_SPATIAL + ACTOR_SCALARS encoder values; out: policy_logit_count(p) floats
int teacher_logits(int p, const float* raw, float* out) {
    torch::NoGradGuard ng;
    const int D = fe::ACTOR_SPATIAL + fe::ACTOR_SCALARS;
    std::vector<float> x(raw, raw + D);
    for (int i = 0; i < fe::ACTOR_SPATIAL; i++) x[i] = std::min(255.f, std::max(0.f, x[i]));
    for (int i = fe::ACTOR_SPATIAL; i < D; i++) x[i] = std::min(32767.f, std::max(-32768.f, x[i]));
    auto t = torch::from_blob(x.data(), {1, D}, torch::kFloat32);
    auto l = (*g_net)->logits(p, t).contiguous();
    const int n = int(l.numel());
    std::memcpy(out, l.data_ptr<float>(), sizeof(float) * n);
    return n;
}
}
