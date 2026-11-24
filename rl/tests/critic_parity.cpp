// Critic parity: Teacher::value_split (the fused CUDA critic on CUDA unless
// FUSED_CRITIC=0) against the module's PyTorch ops in fp32 (TF32 off) and a
// float64 CPU reference, for the value and every critic parameter gradient.
// As trunk_parity: ReLU masks and max-pool choices within fp32 rounding of a
// tie can flip in any fp32 computation, so many small batches run and the
// fast path must match torch's error profile (median at fp32 rounding, no
// more batches with a jump than torch + 2; over 300 one-board batches both
// had 2).
//   critic_parity [ROWS_PER_BATCH] [BATCHES]   env VERBOSE=1
#include <iostream>

#include "model.hpp"

using namespace florent_rl;

static double rel(const torch::Tensor& a, const torch::Tensor& ref) {
    auto r = ref.to(torch::kCPU, torch::kDouble), x = a.to(torch::kCPU, torch::kDouble);
    return (x - r).abs().max().item<double>() / (r.abs().max().item<double>() + 1e-30);
}

int main(int argc, char** argv) {
    const long n = argc > 1 ? std::atol(argv[1]) : 2;
    const int K = argc > 2 ? std::atoi(argv[2]) : 40;
    torch::manual_seed(3);
    at::globalContext().setAllowTF32CuDNN(false);
    at::globalContext().setAllowTF32CuBLAS(false);
    torch::Device dev = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    Teacher net, ref;
    {
        torch::NoGradGuard ng;
        for (auto& p : net->parameters()) p.normal_(0, 0.3);
        auto src = net->named_parameters(), dst = ref->named_parameters();
        for (auto& kv : src) dst[kv.key()].copy_(kv.value());
        for (auto& t : ref->parameters()) t.set_data(t.to(torch::kDouble));
        for (auto& t : ref->buffers()) if (t.is_floating_point()) t.set_data(t.to(torch::kDouble));
    }
    net->to(dev);
    auto cr = net->critic, rc = ref->critic;
    std::vector<double> eg_t, eg_f;
    double ev_t = 0, ev_f = 0;
    bool ok = true;
    for (int k = 0; k < K; k++) {
        auto sp = (torch::rand({n, fe::CRITIC_SPATIAL}) < 0.12).to(torch::kUInt8);
        auto sc = torch::randint(-50, 1000, {n, fe::CRITIC_SCALARS}, torch::TensorOptions().dtype(torch::kInt32));
        auto w = torch::randn({n});
        auto xr = sp.to(torch::kDouble).view({n, fe::CRITIC_CH, fe::CRITIC_PAD, fe::CRITIC_PAD});
        auto vr = rc->forward(xr, ref->critic_features(sc.to(torch::kDouble)));
        ref->zero_grad();
        (vr * w.to(torch::kDouble)).sum().backward();
        auto spd = sp.to(dev), scd = sc.to(dev), wd = w.to(dev);
        net->zero_grad();
        auto vt = cr->forward(spd.to(torch::kFloat32).view({n, fe::CRITIC_CH, fe::CRITIC_PAD, fe::CRITIC_PAD}),
                              net->critic_features(scd.to(torch::kFloat32)));
        (vt * wd).sum().backward();
        std::vector<torch::Tensor> gt;
        for (auto& t : cr->parameters()) gt.push_back(t.grad().clone());
        net->zero_grad();
        auto vf = net->value_split(spd, scd);
        (vf * wd).sum().backward();
        if (std::getenv("REPEAT")) {          // determinism: the same batch again
            std::vector<torch::Tensor> g1;
            for (auto& t : cr->parameters()) g1.push_back(t.grad().clone());
            net->zero_grad();
            (net->value_split(spd, scd) * wd).sum().backward();
            size_t j = 0; double d = 0;
            for (auto& t : cr->parameters()) { d = std::max(d, rel(t.grad(), g1[j])); j++; }
            std::cout << "  batch " << k << ": repeat max rel diff " << d << "\n";
        }
        const double a_v = rel(vt, vr), b_v = rel(vf, vr);
        ev_t = std::max(ev_t, a_v); ev_f = std::max(ev_f, b_v);
        double a = 0, b = 0;
        auto rp = rc->named_parameters();
        size_t i = 0;
        for (auto& kv : cr->named_parameters()) {
            const auto& g64 = rp[kv.key()].grad();
            const double ea = rel(gt[i], g64), eb = rel(kv.value().grad(), g64);
            if (std::getenv("VERBOSE") && (ea > 1e-5 || eb > 1e-5)) std::cout << "    " << kv.key() << ": torch " << ea << " fast " << eb << "\n";
            a = std::max(a, ea); b = std::max(b, eb);
            i++;
        }
        if (std::getenv("VERBOSE")) {
            // smallest relative top-2 gap over the level-0 max-pool windows (float64):
            // a gap at fp32 rounding level lets any fp32 computation pick the other max
            torch::NoGradGuard ng;
            auto h = rc->blocks[0]->as<ResBlock>()->forward(rc->stem->forward(xr).relu());
            auto win = h.unfold(2, 2, 2).unfold(3, 2, 2).reshape({n, 32, 24, 24, 4});
            auto top = std::get<0>(win.topk(2, -1));
            auto t1 = top.select(-1, 0), t2 = top.select(-1, 1);
            auto gap = ((t1 - t2) / t1.abs().clamp_min(1e-30)).masked_fill(t1 <= 0, 1e9);
            std::cout << "  batch " << k << ": smallest level-0 pool gap (float64) " << gap.min().item<double>() << "\n";
        }
        if (std::getenv("VERBOSE")) std::cout << "  batch " << k << ": value torch " << a_v << " fast " << b_v << " | grads torch " << a << " fast " << b << "\n";
        eg_t.push_back(a); eg_f.push_back(b);
    }
    auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    auto jumps = [](const std::vector<double>& v) { int c = 0; for (double x : v) c += x > 1e-4; return c; };
    ok = median(eg_f) < 1e-5 && jumps(eg_f) <= jumps(eg_t) + 2 && ev_f <= 3 * ev_t + 1e-6;
    std::cout << "critic: value vs float64 torch " << ev_t << " fast " << ev_f << " | grads median torch " << median(eg_t)
              << " fast " << median(eg_f) << ", batches with a jump torch " << jumps(eg_t) << "/" << K << " fast " << jumps(eg_f) << "/" << K << "\n";
    std::cout << "path: " << (dev.is_cuda() && use_fused_critic() ? "fused CUDA critic" : "PyTorch ops") << "\n";
    std::cout << (ok ? "CRITIC PARITY OK" : "CRITIC PARITY FAILED") << "\n";
    return ok ? 0 : 1;
}
