// Convert a trained LibTorch SpatialPolicy checkpoint into the portable JSON
// format consumed by the pure-Python production bot (rl/bot/policy.py).
//
// The production interpreter cannot import torch, so this conversion is the
// only supported route from training to deployment. One weight block is
// emitted per unit-policy trunk; the critic is omitted because it is training
// only and never runs in a match.
#include "model.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace florent_rl;

static void mat(std::ostream& os, const torch::Tensor& w, bool& first) {
    os << (first ? "" : ",") << "[";
    auto a = w.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
    auto acc = a.accessor<float, 2>();
    for (int64_t r = 0; r < a.size(0); ++r) {
        os << (r ? "," : "") << "[";
        for (int64_t c = 0; c < a.size(1); ++c) os << (c ? "," : "") << acc[r][c];
        os << "]";
    }
    os << "]";
    first = false;
}

static void vec(std::ostream& os, const torch::Tensor& b, bool& first) {
    os << (first ? "" : ",") << "[";
    auto a = b.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
    auto acc = a.accessor<float, 1>();
    for (int64_t i = 0; i < a.size(0); ++i) os << (i ? "," : "") << acc[i];
    os << "]";
    first = false;
}

// Emit {"weights":[...],"bias":[...]} for one torch::nn::Linear layer.
static void linear(std::ostream& os, torch::nn::LinearImpl* ln) {
    os << "{\"weights\":";
    bool f = true; mat(os, ln->weight, f);
    os << ",\"bias\":";
    f = true; vec(os, ln->bias, f);
    os << "}";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: export_weights CHECKPOINT.pt OUT.json\n";
        return 2;
    }
    SpatialPolicy net;
    try {
        torch::load(net, argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "load failed: " << e.what() << "\n";
        return 3;
    }
    std::ofstream os(argv[2]);
    if (!os) return 4;
    os << "{\"format\":\"florent-policy-json-v1\",\"meta\":{"
       << "\"reward\":[-1,0,1],\"numpy\":false,"
       << "\"actor_input\":" << ACTOR_INPUT << ","
       << "\"max_candidates\":" << MAX_CANDIDATES << "},\"policies\":{";
    for (int p = 0; p < NUM_HEADS - 1; ++p) {
        auto a = net->l1[p]->as<torch::nn::LinearImpl>();
        auto b = net->l2[p]->as<torch::nn::LinearImpl>();
        auto head = net->heads[p]->as<torch::nn::LinearImpl>();
        os << (p ? "," : "") << "\"" << p << "\":[";
        linear(os, a);
        os << ",";
        linear(os, b);
        os << ",";
        linear(os, head);
        os << "]";
    }
    os << "}}\n";
    std::cout << "EXPORT OK " << argv[2] << "\n";
    return 0;
}
