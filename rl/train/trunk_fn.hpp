// Fused CUDA actor trunk, as a differentiable torch op (trunk_fn.cpp; a stub
// without CUDA). Input: raw uint8 actor patches [n, fe::ACTOR_CH*17*17] on the GPU and
// the flat parameter vector (ActorImpl::trunk_flat); output: the trunk's
// centre [n, 32] (same values as ActorImpl::trunk_centre).
#pragma once
#include <torch/torch.h>

namespace florent_rl {
bool fused_trunk_built();
torch::Tensor fused_trunk(const torch::Tensor& x, const torch::Tensor& flat);
// The critic's spatial part: raw uint8 boards [n, 29*48*48] on the GPU and
// CriticImpl::spatial_flat -> [n, 288] (the pooled map, PyTorch flatten order).
torch::Tensor fused_critic(const torch::Tensor& x, const torch::Tensor& flat);
}
