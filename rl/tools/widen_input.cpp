// Checkpoint converter after new actor input channels were appended (the
// encoder's ACTOR_CH grew): every tensor is copied from the old checkpoint;
// each actor's 1x1 stem gets the old weights for the old channels and ZERO
// weights for the new ones, so the converted net decides exactly as before
// until training uses the new inputs. ch_scale (a fixed buffer) is the new
// default. Any other shape difference is an error.
//
//   widen_input OLD.pt NEW.pt
#include <iostream>
#include "../train/model.hpp"
using namespace florent_rl;

static int changed = 0;
static void load_rec(torch::nn::Module& m, torch::serialize::InputArchive& a, const std::string& path) {
    for (auto& kv : m.named_parameters(/*recurse=*/false)) {
        torch::Tensor old;
        a.read(kv.key(), old, /*is_buffer=*/false);
        torch::NoGradGuard ng;
        torch::Tensor& t = kv.value();
        if (old.sizes() == t.sizes()) { t.copy_(old); continue; }
        if (t.dim() == 4 && old.dim() == 4 && t.size(0) == old.size(0) && t.size(1) > old.size(1) && kv.key() == "weight") {
            t.zero_();
            t.narrow(1, 0, old.size(1)).copy_(old);
            std::cout << "  " << path << kv.key() << ": " << old.sizes() << " -> " << t.sizes() << " (new input channels: zero weights)\n";
            changed++;
            continue;
        }
        std::cerr << "shape mismatch at " << path << kv.key() << ": " << old.sizes() << " vs " << t.sizes() << "\n";
        std::exit(3);
    }
    for (auto& kv : m.named_buffers(/*recurse=*/false)) {
        torch::Tensor old;
        a.read(kv.key(), old, /*is_buffer=*/true);
        torch::NoGradGuard ng;
        if (old.sizes() == kv.value().sizes()) kv.value().copy_(old);
        else std::cout << "  " << path << kv.key() << ": " << old.sizes() << " -> " << kv.value().sizes() << " (fixed buffer: new default)\n";
    }
    for (auto& kv : m.named_children()) {
        torch::serialize::InputArchive sub;
        a.read(kv.key(), sub);
        load_rec(*kv.value(), sub, path + kv.key() + ".");
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: widen_input OLD.pt NEW.pt\n"; return 2; }
    Teacher net;
    torch::serialize::InputArchive a;
    a.load_from(argv[1], torch::Device(torch::kCPU));
    load_rec(*net, a, "");
    std::cout << changed << " stem weight(s) widened to " << fe::ACTOR_CH << " input channels\n";
    torch::save(net, argv[2]);
    std::cout << "saved " << argv[2] << "\n";
}
