// PPO self-play trainer for the float teacher, with factorized heads.
//
// Policy semantics (locked, see SESSION_HANDOFF.md):
//  * every head is its own masked categorical;
//  * one record per participating head: head id, legal mask, choice, old logp;
//  * new logp recomputed per head; ratio and clipping per head; the same
//    team-round advantage applies to every head of that decision;
//  * losses are averaged over head records; log-probs are never summed.
//
// Rollouts: G games persist across iterations; each iteration plays R more
// rounds of each, bootstrapping from the critic at the cut. Terminal reward
// only (+1 / -1 / 0). Self-play: one network plays both teams.
//
//   neural_train ITERATIONS OUT.pt [maps...]   (env: GAMES, ROUNDS, EPOCHS, LR, INIT;
//    curriculum: MAX_TURNS round limit, ORE_RADIUS ore distance from cores)
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../sim/florent_game.hpp"
#include "../sim/florent_heads.hpp"
#include "model.hpp"
#include "trajectory.hpp"

using namespace florent_rl;

static int env_int(const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; }
static float env_float(const char* k, float d) { const char* v = std::getenv(k); return v ? float(std::atof(v)) : d; }

struct Rec {                       // one participating head of one decision
    int head, choice;
    std::vector<uint8_t> mask;
    float old_logp;
};
struct DecisionRow {
    int policy, team, round_key;   // round_key indexes RoundRow
    std::vector<int16_t> obs;      // raw actor encoder output
    std::vector<Rec> recs;
    float adv = 0;
};
struct RoundRow {                  // one critic sample (team-0 frame)
    std::vector<uint8_t> critic;   // channels are one-hot; scalars stored separately
    std::vector<int32_t> cscal;
    float value = 0, ret = 0;
};

struct Game {
    Match m;
    std::mt19937 rng;
    int index = 0;
};

static torch::Tensor critic_tensor(const std::vector<int32_t>& ch, const std::vector<int32_t>& sc) {
    std::vector<float> v(ch.size() + sc.size());
    for (size_t i = 0; i < ch.size(); i++) v[i] = float(ch[i]);
    for (size_t i = 0; i < sc.size(); i++) v[ch.size() + i] = float(sc[i]);
    return torch::from_blob(v.data(), {1, (long)v.size()}, torch::kFloat32).clone();
}

static torch::Tensor actor_tensor(const std::vector<int16_t>& o) {
    std::vector<float> v(o.begin(), o.end());
    return torch::from_blob(v.data(), {1, (long)v.size()}, torch::kFloat32).clone();
}

int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "usage: neural_train ITERATIONS OUT.pt MAP [MAP...]\n"; return 2; }
    const int iterations = std::atoi(argv[1]);
    const std::string out = argv[2];
    std::vector<std::string> maps(argv + 3, argv + argc);
    const int G = env_int("GAMES", 4), R = env_int("ROUNDS", 32), EPOCHS = env_int("EPOCHS", 3);
    const int MB = env_int("MINIBATCH", 256);
    const float LR = env_float("LR", 3e-4f), ENT = env_float("ENT", 0.01f), VF = env_float("VF", 0.5f);
    // curriculum: round limit (official 1000); ore farther than this from both cores is removed (-1: keep all)
    const int MAX_TURNS = env_int("MAX_TURNS", 1000), ORE_RADIUS = env_int("ORE_RADIUS", -1);
    torch::set_num_threads(env_int("THREADS", 4));
    torch::manual_seed(7);

    Teacher net;
    if (const char* init = std::getenv("INIT")) { torch::load(net, init); std::cout << "loaded " << init << "\n"; }
    torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(LR));

    std::mt19937 master(12345);
    std::vector<Game> games(G);
    auto new_game = [&](Game& g) {
        std::string mp = maps[master() % maps.size()];
        g.m = Match();
        if (!g.m.load(mp, master())) { std::cerr << "bad map " << mp << "\n"; std::exit(3); }
        g.m.reset();
        g.m.game.max_turns = MAX_TURNS;
        g.m.game.limit_ore(ORE_RADIUS);
        g.rng.seed(master());
    };
    for (auto& g : games) new_game(g);
    long finished = 0, a_wins = 0;

    for (int it = 1; it <= iterations; it++) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<DecisionRow> rows;
        std::vector<RoundRow> rounds;
        std::vector<RoundValue> rvals;       // per game: team 0 and team 1 rows, for GAE
        std::vector<std::vector<int>> game_rounds(G);
        net->eval();
        for (int gi = 0; gi < G; gi++) {
            Game& g = games[gi];
            Match& m = g.m;
            std::vector<RoundValue> seq0, seq1;
            std::vector<int> keys;
            for (int r = 0; r < R && !m.game.over; r++) {
                m.round_critic();
                RoundRow rr; rr.critic.assign(m.critic_channels.begin(), m.critic_channels.end()); rr.cscal = m.critic_scalars;
                {
                    torch::NoGradGuard ng;
                    rr.value = net->value(critic_tensor(m.critic_channels, m.critic_scalars)).item<float>();
                }
                int key = int(rounds.size());
                rounds.push_back(std::move(rr));
                keys.push_back(key);
                for (int a : m.acting_units()) {
                    if (a >= int(m.game.ents.size()) || !m.game.ents[a].alive) continue;
                    int type = m.game.ents[a].type;
                    if (type == fe::SPIKE) {
                        Candidate c;
                        if (spike_action(m.game, a, c)) m.act(a, c);
                        continue;
                    }
                    int p = policy_of_type(type);
                    if (p < 0) continue;
                    m.observe(a);                            // memory updates every turn, as deployed
                    if (!has_choice(m.game, a)) continue;     // only wait: no forward
                    std::vector<float> obs; m.encode_only(a, obs);
                    DecisionRow row; row.policy = p; row.team = m.game.ents[a].team; row.round_key = key;
                    row.obs.assign(obs.begin(), obs.end());
                    torch::Tensor lt;
                    {
                        torch::NoGradGuard ng;
                        lt = net->logits(p, actor_tensor(row.obs)).squeeze(0).contiguous();
                    }
                    const float* L = lt.data_ptr<float>();
                    Decision d = decide(m.game, a, L, [&](int head, const float* l, const std::vector<uint8_t>& mask, float& lp) {
                        // sample from the masked softmax (double precision)
                        double mx = -1e30;
                        for (size_t i = 0; i < mask.size(); i++) if (mask[i]) mx = std::max(mx, double(l[i]));
                        std::vector<double> w(mask.size(), 0.0); double z = 0;
                        for (size_t i = 0; i < mask.size(); i++) if (mask[i]) { w[i] = std::exp(double(l[i]) - mx); z += w[i]; }
                        double u = std::uniform_real_distribution<double>(0.0, z)(g.rng), c = 0;
                        int pick = -1;
                        for (size_t i = 0; i < mask.size(); i++) if (mask[i]) { pick = int(i); c += w[i]; if (u <= c) break; }
                        lp = masked_logp(l, mask, pick);
                        (void)head;
                        return pick;
                    });
                    for (auto& c : d.actions)
                        if (m.act(a, c) != fe::OK) { std::cerr << "illegal decoded action\n"; return 4; }
                    for (auto& hr : d.records) row.recs.push_back({hr.head, hr.choice, hr.mask, hr.logp});
                    rows.push_back(std::move(row));
                }
                m.game.end_round();
                RoundValue v0; v0.team = 0; v0.round = key; v0.value = rounds[key].value;
                RoundValue v1; v1.team = 1; v1.round = key; v1.value = -rounds[key].value;
                v0.terminal = v1.terminal = m.game.over;
                if (m.game.over) { v0.reward = float(m.terminal_reward(0)); v1.reward = -v0.reward; }
                seq0.push_back(v0); seq1.push_back(v1);
            }
            // bootstrap
            float boot = 0.f;
            if (!m.game.over) {
                m.round_critic();
                torch::NoGradGuard ng;
                boot = net->value(critic_tensor(m.critic_channels, m.critic_scalars)).item<float>();
            }
            for (size_t i = 0; i < seq0.size(); i++) {
                float nxt = i + 1 < seq0.size() ? seq0[i + 1].value : boot;
                seq0[i].next_value = nxt; seq1[i].next_value = -nxt;
            }
            for (auto* seq : {&seq0, &seq1}) {
                std::vector<RoundValue*> ptr; for (auto& v : *seq) ptr.push_back(&v);
                finish_gae(ptr, GAMMA, LAMBDA, false);
            }
            for (size_t i = 0; i < seq0.size(); i++) { rvals.push_back(seq0[i]); rvals.push_back(seq1[i]); rounds[seq0[i].round].ret = seq0[i].ret; }
            if (m.game.over) {
                finished++; if (m.game.winner == 0) a_wins++;
                new_game(g);
            }
        }
        // advantages to decisions, normalized over the batch
        std::vector<float> adv0(rounds.size()), adv1(rounds.size());
        for (auto& v : rvals) (v.team == 0 ? adv0 : adv1)[v.round] = v.advantage;
        double mean = 0, var = 0;
        for (auto& r : rows) { r.adv = r.team == 0 ? adv0[r.round_key] : adv1[r.round_key]; mean += r.adv; }
        if (!rows.empty()) {
            mean /= rows.size();
            for (auto& r : rows) var += (r.adv - mean) * (r.adv - mean);
            float inv = 1.f / std::sqrt(float(var / rows.size()) + 1e-8f);
            for (auto& r : rows) r.adv = float((r.adv - mean) * inv);
        }
        auto t1 = std::chrono::steady_clock::now();

        // ---- PPO update
        net->train();
        double pl_sum = 0, vl_sum = 0, ent_sum = 0, ratio_dev0 = -1; long nrec = 0, clipped = 0;
        std::vector<int> order(rows.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = int(i);
        for (int ep = 0; ep < EPOCHS; ep++) {
            std::shuffle(order.begin(), order.end(), master);
            for (size_t s = 0; s < order.size(); s += MB) {
                size_t e = std::min(order.size(), s + MB);
                torch::Tensor pol_loss = torch::zeros({}), ent_loss = torch::zeros({});
                long n_here = 0;
                for (int p = 0; p < NUM_POLICIES; p++) {
                    std::vector<int> idx;
                    for (size_t k = s; k < e; k++) if (rows[order[k]].policy == p) idx.push_back(order[k]);
                    if (idx.empty()) continue;
                    const long obs_n = long(rows[idx[0]].obs.size());
                    std::vector<float> buf(idx.size() * obs_n);
                    for (size_t b = 0; b < idx.size(); b++)
                        std::copy(rows[idx[b]].obs.begin(), rows[idx[b]].obs.end(), buf.begin() + b * obs_n);
                    auto x = torch::from_blob(buf.data(), {(long)idx.size(), obs_n}, torch::kFloat32).clone();
                    auto logits = net->logits(p, x);
                    for (int head : policy_heads(p)) {
                        std::vector<long> rr, ch; std::vector<uint8_t> mk; std::vector<float> olp, adv;
                        int n = HEAD_SIZE[head];
                        for (size_t b = 0; b < idx.size(); b++)
                            for (auto& rec : rows[idx[b]].recs)
                                if (rec.head == head) {
                                    rr.push_back(long(b)); ch.push_back(rec.choice);
                                    mk.insert(mk.end(), rec.mask.begin(), rec.mask.end());
                                    olp.push_back(rec.old_logp); adv.push_back(rows[idx[b]].adv);
                                }
                        if (rr.empty()) continue;
                        long k = long(rr.size());
                        auto rows_t = torch::tensor(rr, torch::kLong);
                        auto hl = logits.index_select(0, rows_t).narrow(1, head_offset(p, head), n);
                        auto mask = torch::from_blob(mk.data(), {k, n}, torch::kUInt8).clone().to(torch::kBool);
                        auto lsm = torch::log_softmax(hl.masked_fill(~mask, -1e9f), 1);
                        auto new_lp = lsm.gather(1, torch::tensor(ch, torch::kLong).view({k, 1})).squeeze(1);
                        auto old_lp = torch::tensor(olp);
                        auto A = torch::tensor(adv);
                        auto ratio = torch::exp(new_lp - old_lp);
                        auto surr = torch::min(ratio * A, torch::clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * A);
                        pol_loss = pol_loss - surr.sum();
                        auto probs = lsm.exp().masked_fill(~mask, 0.f);
                        ent_loss = ent_loss - (probs * lsm.masked_fill(~mask, 0.f)).sum();
                        n_here += k;
                        if (ep == 0 && s == 0) {
                            if (ratio_dev0 < 0) ratio_dev0 = 0;
                            ratio_dev0 = std::max(ratio_dev0, (ratio - 1).abs().max().item<double>());
                        }
                        clipped += ((ratio - 1).abs() > CLIP_EPS).sum().item<long>();
                    }
                }
                if (!n_here) continue;
                auto pl = pol_loss / double(n_here), ent = ent_loss / double(n_here);
                auto loss = pl - ENT * ent;
                opt.zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(net->parameters(), 0.5);
                opt.step();
                pl_sum += pl.item<double>(); ent_sum += ent.item<double>(); nrec += n_here;
            }
            // critic: all rounds of this iteration, one batch
            if (!rounds.empty()) {
                long cn = long(rounds[0].critic.size() + rounds[0].cscal.size());
                std::vector<float> cb(rounds.size() * cn); std::vector<float> tgt;
                for (size_t i = 0; i < rounds.size(); i++) {
                    float* dst = cb.data() + i * cn;
                    for (size_t j = 0; j < rounds[i].critic.size(); j++) dst[j] = rounds[i].critic[j];
                    for (size_t j = 0; j < rounds[i].cscal.size(); j++) dst[rounds[i].critic.size() + j] = float(rounds[i].cscal[j]);
                    tgt.push_back(rounds[i].ret);
                }
                auto cx = torch::from_blob(cb.data(), {(long)rounds.size(), cn}, torch::kFloat32).clone();
                auto vl = torch::mse_loss(net->value(cx), torch::tensor(tgt));
                opt.zero_grad();
                (VF * vl).backward();
                torch::nn::utils::clip_grad_norm_(net->parameters(), 0.5);
                opt.step();
                vl_sum += vl.item<double>();
            }
        }
        auto t2 = std::chrono::steady_clock::now();
        double ms_roll = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_upd = std::chrono::duration<double, std::milli>(t2 - t1).count();
        long recs = 0; for (auto& r : rows) recs += long(r.recs.size());
        std::cout << "iter " << it << " decisions " << rows.size() << " head_records " << recs
                  << " ratio_dev_first_minibatch " << ratio_dev0
                  << " policy_loss " << pl_sum << " entropy " << (nrec ? ent_sum : 0)
                  << " value_loss " << vl_sum / std::max(1, EPOCHS)
                  << " clipped " << clipped << " games_done " << finished << " A_wins " << a_wins
                  << " rollout_ms " << int(ms_roll) << " update_ms " << int(ms_upd) << std::endl;
        if (it % env_int("SAVE_EVERY", 25) == 0 || it == iterations) torch::save(net, out);
    }
    std::cout << "TRAIN COMPLETE checkpoint=" << out << std::endl;
}
