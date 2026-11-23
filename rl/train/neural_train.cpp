#include "model.hpp"
#include "trajectory.hpp"
#include "policy_math.hpp"
#include "../sim/florent_game.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace florent_rl;

// Sample from the *masked* categorical over the full fixed candidate list and
// return the log-probability of that choice under the same distribution. The
// loss recomputes log-probs with the identical mask, so the PPO ratio is
// well-defined.
static int sample(const torch::Tensor& logits, const std::vector<uint8_t>& mask,
                  std::mt19937& rng, float& logp) {
    auto l = logits.detach().to(torch::kCPU);
    const int n = int(mask.size());
    float mx = -1e30f;
    for (int i = 0; i < n; ++i) if (mask[i]) mx = std::max(mx, l[i].item<float>());
    std::vector<double> p(n, 0.0); double z = 0;
    for (int i = 0; i < n; ++i) if (mask[i]) { p[i] = std::exp(double(l[i].item<float>()) - mx); z += p[i]; }
    double u = double(rng()) / double(rng.max()), c = 0; int last = 0;
    for (int i = 0; i < n; ++i) {
        if (!mask[i]) continue;
        last = i; c += p[i] / z;
        if (u <= c) { logp = float(std::log(p[i] / z)); return i; }
    }
    logp = float(std::log(p[last] / z));
    return last;
}

static torch::Tensor masked_logp(const torch::Tensor& logits, const torch::Tensor& mask, int label) {
    auto bmask = mask.to(torch::kBool);
    auto neg = torch::full_like(logits, -1e30f).masked_fill(bmask, 0.f);
    auto lse = torch::logsumexp(logits + neg, 0);
    return logits[label] - lse;
}

static torch::Tensor row_tensor(const std::vector<float>& r) {
    return torch::from_blob(const_cast<float*>(r.data()), {1, (long)r.size()}, torch::kFloat32).clone();
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: neural_train MAP ITERATIONS [CHECKPOINT]\n"; return 2; }
    const int iterations = std::atoi(argv[2]);
    const std::string ck = argc > 3 ? argv[3] : "florent.pt";
    const int horizon = 64, epochs = 2;
    torch::manual_seed(7); std::mt19937 rng(7);
    SpatialPolicy net; net->train();
    torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(3e-4));

    for (int it = 0; it < iterations; ++it) {
        Match m; if (!m.load(argv[1], uint64_t(it + 1))) return 3; m.reset();
        Trajectory tr;
        std::vector<torch::Tensor> rows, masks, critics;
        std::vector<int64_t> labels, policies;
        std::vector<float> old_logps;
        std::vector<std::vector<Candidate>> candidate_rows;
        std::vector<std::pair<int,int>> actor_pos_rows;
        std::vector<float> critic_values;
        std::vector<uint8_t> over_after;       // game over after this round
        std::vector<std::pair<int, int>> round_idx; // index of (team0, team1) RoundValue

        for (int r = 0; r < horizon && !m.game.over; ++r) {
            m.round_critic();
            std::vector<float> cr(m.critic_channels.size() + m.critic_scalars.size());
            for (size_t i = 0; i < m.critic_channels.size(); ++i) cr[i] = float(m.critic_channels[i]);
            for (size_t i = 0; i < m.critic_scalars.size(); ++i)
                cr[m.critic_channels.size() + i] = float(m.critic_scalars[i]);
            torch::NoGradGuard ng;
            auto cx = row_tensor(cr);
            float v = net->critic(cx).item<float>();
            critics.push_back(cx);
            critic_values.push_back(v);

            int rnd = m.game.round;
            int i0 = int(tr.rounds.size());
            RoundValue a; a.team = 0; a.round = rnd; a.value = v;
            RoundValue b; b.team = 1; b.round = rnd; b.value = -v;
            tr.rounds.push_back(a); tr.rounds.push_back(b);
            round_idx.emplace_back(i0, i0 + 1);

            for (int e : m.acting_units()) {
                if (e < 0 || e >= int(m.game.ents.size()) || !m.game.ents[e].alive) continue;
                std::vector<Candidate> cs; enumerate_candidates(m.game, e, cs);
                std::vector<uint8_t> mask(cs.size(), 0);
                bool any = false;
                for (int k = 0; k < int(cs.size()) && k < MAX_CANDIDATES; ++k) {
                    if (legal_candidate(m.game, e, cs[k])) { mask[k] = 1; any = true; }
                }
                if (!any) continue;
                bool any_action = false;
                for (size_t k = 1; k < mask.size(); ++k) if (mask[k]) { any_action = true; break; }
                if (!any_action) continue; // implicit wait; no actor forward needed
                // Gathered structured scores are one-to-one with this fixed
                // candidate row; do not pad the legal mask to head capacity.
                const int nmask = int(cs.size());
                mask.resize(nmask, 0);

                std::vector<float> ar; m.actor_observation(e, ar);
                auto x = row_tensor(ar);
                const UnitPolicy up = policy_for_type(m.game.ents[e].type);
                const int pi = std::max(0, std::min(int(up), NUM_HEADS - 2));
                torch::Tensor logits;
                const auto& ae = m.game.ents[e];
                { torch::NoGradGuard ng2; logits = net->policy_scores(x, cs, ae.x, ae.y, pi).squeeze(0).detach(); }
                float lp = 0; int chosen = sample(logits, mask, rng, lp);
                if (m.act(e, cs[chosen]) != fe::OK) return 4;

                rows.push_back(x);
                masks.push_back(torch::from_blob(const_cast<uint8_t*>(mask.data()),
                                                 {(long)mask.size()}, torch::kByte).clone());
                labels.push_back(chosen);
                policies.push_back(pi);
                candidate_rows.push_back(cs);
                actor_pos_rows.emplace_back(ae.x, ae.y);
                old_logps.push_back(lp);
                const int team = m.game.ents[e].team;
                tr.steps.push_back({team, rnd, e, 0, up, lp, team == 0 ? v : -v, 0, 0, 0, false});
            }
            m.game.end_round();
            over_after.push_back(m.game.over ? 1 : 0);
        }

        // Bootstrap every non-terminal round from the *next* round's critic.
        float bootstrap = 0.f;
        if (!m.game.over && !critic_values.empty()) {
            m.round_critic();
            std::vector<float> cr(m.critic_channels.size() + m.critic_scalars.size());
            for (size_t i = 0; i < m.critic_channels.size(); ++i) cr[i] = float(m.critic_channels[i]);
            for (size_t i = 0; i < m.critic_scalars.size(); ++i)
                cr[m.critic_channels.size() + i] = float(m.critic_scalars[i]);
            torch::NoGradGuard ng; bootstrap = net->critic(row_tensor(cr)).item<float>();
        }

        const int nrounds = int(critic_values.size());
        for (int r = 0; r < nrounds; ++r) {
            const bool term = over_after[size_t(r)] != 0;
            const float next0 = term ? 0.f : (r + 1 < nrounds ? critic_values[size_t(r + 1)] : bootstrap);
            RoundValue& t0 = tr.rounds[size_t(round_idx[size_t(r)].first)];
            RoundValue& t1 = tr.rounds[size_t(round_idx[size_t(r)].second)];
            t0.next_value = next0; t1.next_value = -next0;
            t0.terminal = term; t1.terminal = term;
            if (term) {
                const float r0 = float(m.terminal_reward(0));
                t0.reward = r0; t1.reward = -r0;
            }
        }
        tr.finish(bootstrap, true);
        if (rows.empty()) continue;

        // Value targets are team-0 framed; only team-0 rounds feed the critic.
        std::vector<float> targets;
        targets.reserve(round_idx.size());
        for (auto& p : round_idx) targets.push_back(tr.rounds[size_t(p.first)].ret);
        auto target = torch::tensor(targets);

        double last_loss = 0, last_value = 0;
        for (int ep = 0; ep < epochs; ++ep) {
            torch::Tensor policy_loss = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32));
            for (size_t i = 0; i < rows.size(); ++i) {
                auto logits = net->policy_scores(rows[i], candidate_rows[i], actor_pos_rows[i].first, actor_pos_rows[i].second, int(policies[i])).squeeze(0);
                auto lp = masked_logp(logits, masks[i], int(labels[i]));
                const float adv = tr.steps[i].advantage;
                auto ratio = torch::exp(lp - old_logps[i]);
                auto clipped = torch::clamp(ratio, 1.f - CLIP_EPS, 1.f + CLIP_EPS);
                policy_loss += -torch::min(ratio * adv, clipped * adv);
            }
            policy_loss = policy_loss / float(rows.size());
            auto values = net->critic(torch::cat(critics)).squeeze(1);
            auto value_loss = torch::mse_loss(values, target);
            auto loss = policy_loss + 0.5f * value_loss;
            opt.zero_grad(); loss.backward();
            torch::nn::utils::clip_grad_norm_(net->parameters(), 0.5); opt.step();
            last_loss = loss.item<float>(); last_value = value_loss.item<float>();
        }
        std::cout << "iter=" << it + 1 << " loss=" << last_loss
                  << " value=" << last_value << " samples=" << rows.size()
                  << " rounds=" << nrounds << "\n";
    }
    torch::save(net, ck);
    std::cout << "NEURAL TRAIN COMPLETE checkpoint=" << ck << "\n";
}
