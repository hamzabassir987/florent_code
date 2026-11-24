// Actor trunk parity. Three computations of the same logits and parameter
// gradients, on random weights and sparse random compact inputs:
//   ref   : the original padded trunk read at its centre, float64 on the CPU
//   torch : Actor::forward_padded in fp32 on the device (TF32 off)
//   fast  : Teacher::logits_split, i.e. the shrinking-window trunk (the fused
//           CUDA kernels on CUDA unless FUSED_TRUNK=0)
// A gradient is a sum over ReLU masks, and once in a while a pre-activation
// lies within fp32 rounding of 0: that mask then differs from float64's in
// ANY fp32 computation (torch's too), and the gradient jumps. So the test
// runs many small batches per policy and passes when fast's median error is
// at fp32 rounding level (< 1e-5) and its batches with a jump (error > 1e-4)
// are no more frequent than torch's (+1). Logits must always match.
//   trunk_parity [ROWS_PER_BATCH] [BATCHES]   env CPU=1, VERBOSE=1 (every batch)
#include <iostream>

#include "model.hpp"

using namespace florent_rl;

static double rel(const torch::Tensor& a, const torch::Tensor& ref) {
    auto r = ref.to(torch::kCPU, torch::kDouble), x = a.to(torch::kCPU, torch::kDouble);
    return (x - r).abs().max().item<double>() / (r.abs().max().item<double>() + 1e-30);
}

int main(int argc, char** argv) {
    const long n = argc > 1 ? std::atol(argv[1]) : 32;
    const int K = argc > 2 ? std::atoi(argv[2]) : 12;
    torch::manual_seed(1);
    at::globalContext().setAllowTF32CuDNN(false);
    at::globalContext().setAllowTF32CuBLAS(false);
    torch::Device dev = torch::cuda::is_available() && !std::getenv("CPU") ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    Teacher net, ref;
    {
        torch::NoGradGuard ng;
        for (auto& p : net->parameters()) p.normal_(0, 0.3);
        auto src = net->named_parameters(), dst = ref->named_parameters();
        for (auto& kv : src) dst[kv.key()].copy_(kv.value());
    }
    {   // float64 parameters and float buffers (the index buffers stay int64)
        torch::NoGradGuard ng;
        for (auto& t : ref->parameters()) t.set_data(t.to(torch::kDouble));
        for (auto& t : ref->buffers()) if (t.is_floating_point()) t.set_data(t.to(torch::kDouble));
    }
    net->to(dev);
    bool ok = true;
    double worst_l = 0;
    for (int p = 0; p < NUM_POLICIES; p++) {
        auto actor = net->actors[p]->as<ActorImpl>();
        auto ractor = ref->actors[p]->as<ActorImpl>();
        std::vector<double> eg_t, eg_f;
        double el = 0;
        for (int k = 0; k < K; k++) {
            auto sp = torch::randint(0, 256, {n, fe::ACTOR_SPATIAL}, torch::TensorOptions().dtype(torch::kUInt8));
            sp.masked_fill_(torch::rand({n, fe::ACTOR_SPATIAL}) < 0.7, 0);
            auto sc = torch::randint(-50, 1000, {n, fe::ACTOR_SCALARS}, torch::TensorOptions().dtype(torch::kInt16));
            auto w = torch::randn({n, policy_logit_count(p)});
            // float64 reference
            auto xr = sp.to(torch::kDouble).view({n, fe::ACTOR_CH, fe::PATCH, fe::PATCH}) * ref->ch_scale;
            auto lr = ractor->forward_padded(xr, ref->actor_features(sc.to(torch::kDouble)));
            ref->zero_grad();
            (lr * w.to(torch::kDouble)).sum().backward();
            // fp32 padded torch ops
            auto spd = sp.to(dev), scd = sc.to(dev), wd = w.to(dev);
            auto xt = spd.to(torch::kFloat32).view({n, fe::ACTOR_CH, fe::PATCH, fe::PATCH}) * net->ch_scale;
            net->zero_grad();
            auto lt = actor->forward_padded(xt, net->actor_features(scd.to(torch::kFloat32)));
            (lt * wd).sum().backward();
            std::vector<torch::Tensor> gt;
            for (auto& t : actor->parameters()) gt.push_back(t.grad().clone());
            // the fast path
            net->zero_grad();
            auto lf = net->logits_split(p, spd, scd);
            (lf * wd).sum().backward();
            const double el_t = rel(lt, lr), el_f = rel(lf, lr);
            if (el_f > 3 * el_t + 1e-6) { ok = false; std::cout << "  policy " << p << " batch " << k << ": logits " << el_f << " vs torch " << el_t << "   FAIL\n"; }
            el = std::max(el, el_f);
            double a = 0, b = 0;
            auto rp = ractor->named_parameters();
            size_t i = 0;
            for (auto& kv : actor->named_parameters()) {
                const auto& g64 = rp[kv.key()].grad();
                a = std::max(a, rel(gt[i], g64)); b = std::max(b, rel(kv.value().grad(), g64));
                i++;
            }
            if (std::getenv("VERBOSE")) std::cout << "  policy " << p << " batch " << k << ": grads torch " << a << " fast " << b << "\n";
            eg_t.push_back(a); eg_f.push_back(b);
        }
        auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        auto jumps = [](const std::vector<double>& v) { int c = 0; for (double x : v) c += x > 1e-4; return c; };
        const bool pok = median(eg_f) < 1e-5 && jumps(eg_f) <= jumps(eg_t) + 1;
        std::cout << "policy " << p << ": logits vs float64 " << el << " | grads median torch " << median(eg_t) << " fast " << median(eg_f)
                  << ", batches with a jump torch " << jumps(eg_t) << "/" << K << " fast " << jumps(eg_f) << "/" << K << (pok ? "" : "   FAIL") << "\n";
        ok &= pok; worst_l = std::max(worst_l, el);
    }
    std::cout << "path: " << (dev.is_cuda() && use_fused_trunk() ? "fused CUDA trunk" : "PyTorch ops") << "\n";
    std::cout << (ok ? "TRUNK PARITY OK" : "TRUNK PARITY FAILED") << " (worst logits vs float64 " << worst_l << ")\n";
    return ok ? 0 : 1;
}
