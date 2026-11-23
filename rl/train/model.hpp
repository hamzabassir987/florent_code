#pragma once
#include "florent_layout.hpp"
#include <torch/torch.h>

namespace florent_rl {

struct ActorTrunkImpl : torch::nn::Module {
    torch::nn::Conv2d stem{nullptr};
    torch::nn::ModuleList pw, dw;
    ActorTrunkImpl() : pw(), dw() {
        stem = register_module("stem", torch::nn::Conv2d(torch::nn::Conv2dOptions(fe::ACTOR_CH, 32, 1)));
        for (int b = 0; b < 4; ++b) {
            pw->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 1)));
            dw->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).padding(1).groups(32)));
        }
        register_module("pw", pw);
        register_module("dw", dw);
    }
    torch::Tensor forward(torch::Tensor x) {
        auto h = torch::relu(stem->forward(x));
        for (int b = 0; b < 4; ++b) {
            auto z = torch::relu(pw[b]->as<torch::nn::Conv2d>()->forward(h));
            z = dw[b]->as<torch::nn::Conv2d>()->forward(z);
            h = torch::relu(h + z);
        }
        return h;
    }
};
TORCH_MODULE(ActorTrunk);

struct ActionHeadImpl : torch::nn::Module {
    // Structured logits; the decoder maps each fixed Candidate to exactly one
    // tuple of these entries and sums the tuple score before legal masking.
    torch::nn::ModuleList parts;
    static constexpr int WAIT=1, MOVE=8, SPAWN=49, CONVERT=5, BUILD_TYPE=9,
        BUILD_OFFSET=8, BUILD_DIR=9, TARGET=121, ROTATE=8,
        LAUNCH_BUILDER=8, LAUNCH_DEST=121;
    ActionHeadImpl() : parts() {
        for (int n : {WAIT, MOVE, SPAWN, CONVERT, BUILD_TYPE, BUILD_OFFSET,
                      BUILD_DIR, TARGET, ROTATE, LAUNCH_BUILDER, LAUNCH_DEST})
            parts->push_back(torch::nn::Linear(72, n));
        register_module("parts", parts);
    }
    torch::Tensor forward(torch::Tensor x) {
        std::vector<torch::Tensor> out;
        for (auto& m : *parts) out.push_back(m->as<torch::nn::Linear>()->forward(x));
        return torch::cat(out, 1);
    }
    torch::Tensor scores(torch::Tensor x, const std::vector<Candidate>& cs, int ax, int ay) {
        auto raw = forward(x);
        std::vector<torch::Tensor> vals;
        constexpr int O_WAIT=0, O_MOVE=1, O_SPAWN=9, O_CONVERT=58, O_BUILD_TYPE=63,
            O_BUILD_OFF=72, O_BUILD_DIR=80, O_TARGET=89, O_ROTATE=210,
            O_LAUNCH_B=218;
        for (const auto& c : cs) {
            int idx=O_WAIT;
            switch(c.kind) {
            case ActionKind::Move: idx=O_MOVE+c.a-1; break;
            case ActionKind::Spawn: idx=O_SPAWN+(c.a-ax+2)+5*(c.b-ay+2); break;
            case ActionKind::Convert: idx=O_CONVERT+(c.a==1?0:c.a==2?1:c.a==4?2:c.a==8?3:4); break;
            case ActionKind::Build: idx=O_BUILD_TYPE+(c.a==fe::CONVEYOR?0:c.a==fe::SPLITTER?1:c.a==fe::HARVESTER?2:c.a==fe::GUNNER?3:c.a==fe::SENTINEL?4:c.a==fe::LAUNCHER?5:c.a==fe::SPIKE?6:c.a==fe::BARRIER?7:8); break;
            case ActionKind::Attack: case ActionKind::Heal: case ActionKind::Destroy: case ActionKind::Fire:
                idx=O_TARGET+(c.a-ax+5)+11*(c.b-ay+5); break;
            case ActionKind::Rotate: idx=O_ROTATE+c.a-1; break;
            case ActionKind::Launch: idx=O_LAUNCH_B+(c.a-ax+1)+3*(c.b-ay+1); break;
            default: break;
            }
            const int64_t safe = std::max<int64_t>(0, std::min<int64_t>(idx, raw.size(1)-1));
            vals.push_back(raw.select(1, safe).unsqueeze(1));
        }
        return torch::cat(vals,1);
    }

};
TORCH_MODULE(ActionHead);


struct CriticBlockImpl : torch::nn::Module {
    torch::nn::Conv2d pw{nullptr}, dw{nullptr};
    CriticBlockImpl(int channels) {
        pw = register_module("pw", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 1)));
        dw = register_module("dw", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1).groups(channels)));
    }
    torch::Tensor forward(torch::Tensor x) {
        auto z = torch::relu(pw->forward(x));
        z = dw->forward(z);
        return torch::relu(x + z);
    }
};
TORCH_MODULE(CriticBlock);

struct SpatialPolicyImpl : torch::nn::Module {
    torch::nn::ModuleList trunks, heads;
    torch::nn::Conv2d critic_stem{nullptr};
    torch::nn::ModuleList critic_blocks;
    torch::nn::Linear value{nullptr};

    SpatialPolicyImpl() : trunks(), heads(), critic_blocks() {
        for (int p = 0; p < NUM_HEADS - 1; ++p) {
            trunks->push_back(ActorTrunk());
            heads->push_back(ActionHead());
        }
        register_module("trunks", trunks);
        register_module("heads", heads);

        critic_stem = register_module("critic_stem", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(fe::CRITIC_CH, 32, 1)));
        for (int i = 0; i < 3; ++i) critic_blocks->push_back(CriticBlock(32));
        register_module("critic_blocks", critic_blocks);
        // 48 -> 24 -> 12 -> 6, 32 channels plus 33 scalar features.
        value = register_module("value", torch::nn::Linear(32 * 6 * 6 + fe::CRITIC_SCALARS, 1));
    }

    torch::Tensor features(torch::Tensor x, int p = 0) {
        p = std::max(0, std::min(p, NUM_HEADS - 2));
        auto spatial = x.slice(1, 0, fe::ACTOR_SPATIAL)
                         .view({x.size(0), fe::ACTOR_CH, fe::PATCH, fe::PATCH});
        auto scalars = x.slice(1, fe::ACTOR_SPATIAL, ACTOR_INPUT);
        auto h = trunks[p]->as<ActorTrunk>()->forward(spatial);
        auto centre = h.index({torch::indexing::Slice(), torch::indexing::Slice(), fe::HALF, fe::HALF});
        std::vector<torch::Tensor> parts{centre, scalars};
        return torch::cat(parts, 1);
    }
    torch::Tensor policy_scores(torch::Tensor x, const std::vector<Candidate>& cs, int ax, int ay, int p = 0) {
        p = std::max(0, std::min(p, NUM_HEADS - 2));
        return heads[p]->as<ActionHead>()->scores(features(x, p), cs, ax, ay);
    }
    torch::Tensor critic(torch::Tensor x) {
        const auto spatial_n = fe::CRITIC_SPATIAL;
        auto spatial = x.slice(1, 0, spatial_n)
                         .view({x.size(0), fe::CRITIC_CH, fe::CRITIC_PAD, fe::CRITIC_PAD});
        auto scalars = x.slice(1, spatial_n, CRITIC_INPUT);
        auto h = torch::relu(critic_stem->forward(spatial));
        for (int i = 0; i < 3; ++i) {
            h = critic_blocks[i]->as<CriticBlock>()->forward(h);
            h = torch::max_pool2d(h, {2, 2}, {2, 2});
        }
        h = h.flatten(1);
        return value->forward(torch::cat({h, scalars}, 1));
    }
};
TORCH_MODULE(SpatialPolicy);
}
