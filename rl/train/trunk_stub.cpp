// Built when the CUDA toolkit is not available: the fused trunk is off.
#include "trunk_fn.hpp"
namespace florent_rl {
bool fused_trunk_built() { return false; }
torch::Tensor fused_trunk(const torch::Tensor&, const torch::Tensor&) { TORCH_CHECK(false, "fused trunk not built"); }
torch::Tensor fused_critic(const torch::Tensor&, const torch::Tensor&) { TORCH_CHECK(false, "fused critic not built"); }
}
