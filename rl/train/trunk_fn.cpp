// Autograd wrapper of the fused CUDA actor trunk (trunk_kernels.cu).
#include "trunk_fn.hpp"

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <stdexcept>

#include "trunk_kernels.hpp"
#include "../../engine/src/obs.hpp"
static_assert(TRUNK_IN_CH == fe::ACTOR_CH, "trunk_kernels.hpp TRUNK_IN_CH must equal the encoder's ACTOR_CH");

namespace florent_rl {

static void check(const char* what) {
    if (const char* e = trunk_last_error()) throw std::runtime_error(std::string("fused trunk ") + what + ": " + e);
}

struct TrunkFn : torch::autograd::Function<TrunkFn> {
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, const torch::Tensor& x, const torch::Tensor& flat) {
        const long n = x.size(0);
        auto out = torch::empty({n, 32}, flat.options());
        trunk_forward(x.data_ptr<uint8_t>(), n, flat.data_ptr<float>(), out.data_ptr<float>(),
                      c10::cuda::getCurrentCUDAStream().stream());
        check("forward");
        ctx->save_for_backward({x, flat});
        return out;
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list g) {
        auto saved = ctx->get_saved_variables();
        const torch::Tensor& x = saved[0];
        const torch::Tensor& flat = saved[1];
        const long n = x.size(0);
        auto gout = g[0].contiguous();
        const int grid = trunk_grid(n);
        auto scratch = torch::empty({std::max(1, grid), TRUNK_SCRATCH}, flat.options());
        auto gacc = torch::empty({std::max(1, grid), TRUNK_FLAT}, flat.options());
        auto gflat = torch::empty({TRUNK_FLAT}, flat.options());
        trunk_backward(x.data_ptr<uint8_t>(), n, flat.data_ptr<float>(), gout.data_ptr<float>(), scratch.data_ptr<float>(),
                       gacc.data_ptr<float>(), gflat.data_ptr<float>(), c10::cuda::getCurrentCUDAStream().stream());
        check("backward");
        return {torch::Tensor(), gflat};
    }
};

struct CriticFn : torch::autograd::Function<CriticFn> {
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, const torch::Tensor& x, const torch::Tensor& flat) {
        const long n = x.size(0);
        auto p1 = torch::empty({n, 32 * 24 * 24}, flat.options()), p2 = torch::empty({n, 32 * 12 * 12}, flat.options());
        auto out = torch::empty({n, CRITIC_OUT}, flat.options());
        critic_forward(x.data_ptr<uint8_t>(), n, flat.data_ptr<float>(), p1.data_ptr<float>(), p2.data_ptr<float>(),
                       out.data_ptr<float>(), c10::cuda::getCurrentCUDAStream().stream());
        check("critic forward");
        ctx->save_for_backward({x, flat, p1, p2});
        return out;
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list g) {
        auto sv = ctx->get_saved_variables();
        const torch::Tensor &x = sv[0], &flat = sv[1], &p1 = sv[2], &p2 = sv[3];
        const long n = x.size(0);
        auto gout = g[0].contiguous();
        const int grid = std::max(1, critic_grid(n));
        auto scratch = torch::empty({grid, CRITIC_SCRATCH}, flat.options());
        auto gacc = torch::empty({grid, CRITIC_FLAT}, flat.options());
        auto gp1 = torch::empty_like(p1), gp2 = torch::empty_like(p2);
        auto gflat = torch::empty({CRITIC_FLAT}, flat.options());
        critic_backward(x.data_ptr<uint8_t>(), n, flat.data_ptr<float>(), p1.data_ptr<float>(), p2.data_ptr<float>(),
                        gout.data_ptr<float>(), gp1.data_ptr<float>(), gp2.data_ptr<float>(), scratch.data_ptr<float>(),
                        gacc.data_ptr<float>(), gflat.data_ptr<float>(), c10::cuda::getCurrentCUDAStream().stream());
        check("critic backward");
        return {torch::Tensor(), gflat};
    }
};

torch::Tensor fused_critic(const torch::Tensor& x, const torch::Tensor& flat) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kUInt8 && x.is_contiguous() && x.dim() == 2 && x.size(1) == CRITIC_IN_CH * 48 * 48,
                "fused_critic: x must be a contiguous CUDA uint8 [n, 29*48*48]");
    TORCH_CHECK(flat.is_cuda() && flat.scalar_type() == torch::kFloat32 && flat.numel() == CRITIC_FLAT, "fused_critic: flat");
    return CriticFn::apply(x, flat.contiguous());
}

bool fused_trunk_built() { return true; }

torch::Tensor fused_trunk(const torch::Tensor& x, const torch::Tensor& flat) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kUInt8 && x.is_contiguous() && x.dim() == 2 && x.size(1) == TRUNK_IN_CH * 289,
                "fused_trunk: x must be a contiguous CUDA uint8 [n, TRUNK_IN_CH*17*17]");
    TORCH_CHECK(flat.is_cuda() && flat.scalar_type() == torch::kFloat32 && flat.numel() == TRUNK_FLAT, "fused_trunk: flat");
    return TrunkFn::apply(x, flat.contiguous());
}

}  // namespace florent_rl
