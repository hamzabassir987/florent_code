// Checkpoint surgery (keeps everything else bit for bit):
//  * builder `builder_build_type`: the gunner and sentinel logits get the
//    spike's output row (weights + bias), so builders want them as much as
//    spikes; the masks (docs/masks.md) decide where and facing what;
//  * the gunner and sentinel policies' output layers are zeroed: every head of
//    theirs is uniform over its legal options (fire is masked to enemies).
//
//   surgery IN.pt OUT.pt
#include <iostream>
#include "../train/model.hpp"
using namespace florent_rl;
int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: surgery IN.pt OUT.pt\n"; return 2; }
    Teacher net; torch::load(net, argv[1], torch::Device(torch::kCPU));
    torch::NoGradGuard ng;
    auto bo = net->actors[P_BUILDER]->as<Actor>()->out;
    const int off = head_offset(P_BUILDER, HB_BTYPE);
    auto idx = [](int type) { for (int i = 0; i < 9; i++) if (BUILD_ORDER[i] == type) return i; return -1; };
    const int sp = off + idx(fe::SPIKE);
    for (int type : {fe::GUNNER, fe::SENTINEL}) {
        const int r = off + idx(type);
        bo->weight[r].copy_(bo->weight[sp]);
        bo->bias[r].copy_(bo->bias[sp]);
        std::cout << "builder_build_type: row " << r << " (" << (type == fe::GUNNER ? "gunner" : "sentinel") << ") <- row " << sp << " (spike)\n";
    }
    for (int p : {P_GUNNER, P_SENTINEL}) {
        auto o = net->actors[p]->as<Actor>()->out;
        o->weight.zero_(); o->bias.zero_();
        std::cout << (p == P_GUNNER ? "gunner" : "sentinel") << " policy: output layer zeroed (uniform heads)\n";
    }
    torch::save(net, argv[2]);
    std::cout << "saved " << argv[2] << "\n";
}
