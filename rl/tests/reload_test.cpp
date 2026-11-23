// Checkpoint reload test: a reloaded teacher must reproduce every policy's
// logits and the critic value bit for bit.   reload_test [CHECKPOINT]
#include <iostream>
#include "../train/model.hpp"
using namespace florent_rl;
int main(int argc, char** argv) {
    torch::manual_seed(3);
    Teacher a;
    if (argc > 1) torch::load(a, argv[1]);
    a->eval();
    const std::string path = "/tmp/reload_test.pt";
    torch::save(a, path);
    Teacher b; torch::load(b, path); b->eval();
    torch::NoGradGuard ng;
    auto x = torch::randint(0, 3, {8, fe::ACTOR_SPATIAL + fe::ACTOR_SCALARS}).to(torch::kFloat32);
    auto c = torch::randint(0, 2, {4, fe::CRITIC_SPATIAL + fe::CRITIC_SCALARS}).to(torch::kFloat32);
    double worst = 0;
    for (int p = 0; p < NUM_POLICIES; p++)
        worst = std::max(worst, (a->logits(p, x) - b->logits(p, x)).abs().max().item<double>());
    worst = std::max(worst, (a->value(c) - b->value(c)).abs().max().item<double>());
    std::cout << (worst == 0.0 ? "RELOAD OK" : "RELOAD MISMATCH") << " max_abs_diff " << worst << "\n";
    return worst == 0.0 ? 0 : 1;
}
