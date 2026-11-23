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
//   neural_train ITERATIONS OUT.pt [maps...]   (env: GAMES, ROUNDS, INIT, ENTROPY_FILE;
//    live knobs in the entropy file, see entropy_file.hpp)
#include <torch/torch.h>
#include "entropy_file.hpp"
#include "game_log.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <cmath>

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

// ===========================================================================
// LOG FORMATTING
// ===========================================================================
namespace lfmt {
inline std::string num(double x, int w = 9) {
    std::ostringstream o;
    const double a = std::fabs(x);
    if (x != 0.0 && (a < 1e-3 || a >= 1e5)) o << std::scientific << std::setprecision(2);
    else                                    o << std::fixed << std::setprecision(a >= 100 ? 1 : 4);
    o << x;
    std::string s = o.str();
    if ((int)s.size() < w) s.insert(0, w - s.size(), ' ');
    return s;
}
inline std::string ratio(long a, long b) {
    std::ostringstream o;
    o << std::setw(4) << a << "/" << std::left << std::setw(4) << b << std::right
      << " (" << std::fixed << std::setprecision(1) << std::setw(5) << (b ? 100.0 * a / b : 0.0) << "%)";
    return o.str();
}
inline std::string label(const char* l, int w = 12) {
    std::string s(l);
    if ((int)s.size() < w) s.append(w - s.size(), ' ');
    return s;
}
}  // namespace lfmt

// CSV files next to the checkpoint; appended when the run resumes (INIT).
static std::ofstream open_log(const std::filesystem::path& p, const std::string& header, bool append) {
    const bool fresh = !append || !std::filesystem::exists(p);
    std::ofstream f(p, fresh ? std::ios::trunc : std::ios::app);
    if (fresh) f << header << "\n";
    return f;
}

// One ended game, for the "ended" lines and games.csv.
struct EndStat {
    int rounds = 0, winner = -1;
    std::string cond;
    int ti_col[2] = {0, 0}, vo_col[2] = {0, 0}, harv[2] = {0, 0}, units[2] = {0, 0};
};
static EndStat end_stat(const fe::Game& g) {
    EndStat e; e.rounds = g.round; e.winner = g.winner; e.cond = g.win_condition;
    for (int t = 0; t < 2; t++) { e.ti_col[t] = g.teams[t].ti_collected; e.vo_col[t] = g.teams[t].vo_collected; }
    for (auto& en : g.ents) if (en.alive) { e.units[en.team]++; if (en.type == fe::HARVESTER) e.harv[en.team]++; }
    return e;
}

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
    const int G = env_int("GAMES", 4), R = env_int("ROUNDS", 32);
    const int MB = env_int("MINIBATCH", 256);
    const float VF = env_float("VF", 0.5f);
    // Live knobs (entropy_file.hpp): per-head entropy, lr, epochs, game_rounds,
    // ore_radius. Env LR / ENT / EPOCHS only set the starting values.
    const std::string KNOB_PATH = std::getenv("ENTROPY_FILE") ? std::getenv("ENTROPY_FILE") : "entropy_file";
    Knobs K;
    K.lr = env_float("LR", 3e-4f); K.epochs = env_int("EPOCHS", 3);
    for (auto& e : K.head_entropy) e = env_float("ENT", 0.01f);
    if (!read_entropy_file(KNOB_PATH, K)) std::cout << "no " << KNOB_PATH << ": defaults\n";
    if (std::getenv("THREADS")) torch::set_num_threads(env_int("THREADS", 0));   // default: torch picks (all cores)
    torch::manual_seed(7);

    Teacher net;
    if (const char* init = std::getenv("INIT")) { torch::load(net, init); std::cout << "loaded " << init << "\n"; }
    torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(K.lr));

    std::mt19937 master(12345);
    std::vector<Game> games(G);
    auto new_game = [&](Game& g) {
        std::string mp = maps[master() % maps.size()];
        g.m = Match();
        if (!g.m.load(mp, master())) { std::cerr << "bad map " << mp << "\n"; std::exit(3); }
        g.m.reset();
        g.m.game.max_turns = K.game_rounds;
        g.m.game.limit_ore(K.ore_radius, master());
        g.rng.seed(master());
    };
    for (auto& g : games) new_game(g);
    long finished = 0, a_wins = 0;

    namespace fs = std::filesystem;
    const fs::path run_dir = fs::path(out).parent_path().empty() ? fs::path(".") : fs::path(out).parent_path();
    const bool resumed = std::getenv("INIT") != nullptr;
    std::string ppo_header = "iter,decisions,head_records,ended,a_wins,mean_rounds,policy_loss,value_loss,entropy,"
                             "explained_var,adv_mean,adv_std,adv_min,adv_max,ratio_dev,clipped_pct,lr,epochs,game_rounds,ore_radius,rollout_s,update_s";
    for (int h = 0; h < NUM_HEAD_IDS; h++) ppo_header += std::string(",H_") + HEAD_NAMES[h];
    std::ofstream ppo_log = open_log(run_dir / "ppo_logs.csv", ppo_header, resumed);
    std::ofstream games_log = open_log(run_dir / "games.csv",
        "iter,map_kind,rounds,winner,condition,ti_collected_a,ti_collected_b,vo_collected_a,vo_collected_b,harvesters_a,harvesters_b,units_a,units_b,greedy", resumed);
    {
        long np = 0; for (auto& t : net->parameters()) np += t.numel();
        std::cout << "train: " << maps.size() << " maps, " << G << " games x " << R << " rounds an iteration, minibatch " << MB
                  << ", params " << np << ", threads " << torch::get_num_threads() << "\n"
                  << "train: knobs from " << KNOB_PATH << ", logs in " << run_dir.string() << " (ppo_logs.csv, games.csv)" << std::endl;
    }

    for (int it = 1; it <= iterations; it++) {
        read_entropy_file(KNOB_PATH, K);
        for (auto& pg : opt.param_groups()) static_cast<torch::optim::AdamOptions&>(pg.options()).lr(K.lr);
        for (auto& g : games) g.m.game.max_turns = K.game_rounds;   // running games too
        const int EPOCHS = K.epochs;
        auto t0 = std::chrono::steady_clock::now();
        std::vector<DecisionRow> rows;
        std::vector<RoundRow> rounds;
        std::vector<RoundValue> rvals;       // per game: team 0 and team 1 rows, for GAE
        std::vector<std::vector<int>> game_rounds(G);
        std::vector<EndStat> ended;
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
                ended.push_back(end_stat(m.game));
                new_game(g);
            }
        }
        // advantages to decisions, normalized over the batch
        std::vector<float> adv0(rounds.size()), adv1(rounds.size());
        for (auto& v : rvals) (v.team == 0 ? adv0 : adv1)[v.round] = v.advantage;
        double mean = 0, var = 0, adv_min = 0, adv_max = 0;
        for (auto& r : rows) {
            r.adv = r.team == 0 ? adv0[r.round_key] : adv1[r.round_key]; mean += r.adv;
            adv_min = std::min(adv_min, double(r.adv)); adv_max = std::max(adv_max, double(r.adv));
        }
        double adv_mean = 0, adv_std = 0;
        if (!rows.empty()) {
            mean /= rows.size();
            for (auto& r : rows) var += (r.adv - mean) * (r.adv - mean);
            adv_mean = mean; adv_std = std::sqrt(var / rows.size());
            float inv = 1.f / std::sqrt(float(var / rows.size()) + 1e-8f);
            for (auto& r : rows) r.adv = float((r.adv - mean) * inv);
        }
        auto t1 = std::chrono::steady_clock::now();

        // ---- PPO update
        net->train();
        double pl_sum = 0, vl_sum = 0, ent_sum = 0, ratio_dev0 = -1; long nrec = 0, clipped = 0;
        double head_ent[NUM_HEAD_IDS] = {}; long head_n[NUM_HEAD_IDS] = {};   // first epoch, per head
        int n_steps = 0;                                                         // policy minibatch steps
        std::vector<int> order(rows.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = int(i);
        for (int ep = 0; ep < EPOCHS; ep++) {
            std::shuffle(order.begin(), order.end(), master);
            for (size_t s = 0; s < order.size(); s += MB) {
                size_t e = std::min(order.size(), s + MB);
                torch::Tensor pol_loss = torch::zeros({}), ent_loss = torch::zeros({}), ent_bonus = torch::zeros({});
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
                        auto h_sum = -(probs * lsm.masked_fill(~mask, 0.f)).sum();   // this head's entropy, summed
                        ent_loss = ent_loss + h_sum;
                        ent_bonus = ent_bonus + K.head_entropy[head] * h_sum;
                        if (ep == 0) { head_ent[head] += h_sum.item<double>(); head_n[head] += k; }
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
                auto loss = pl - ent_bonus / double(n_here);
                opt.zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(net->parameters(), 0.5);
                opt.step();
                pl_sum += pl.item<double>(); ent_sum += ent.item<double>(); nrec += n_here; n_steps++;
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
        // critic: explained variance of the round values against the returns
        double ev = 0;
        if (rounds.size() > 1) {
            double sy = 0, sy2 = 0, se = 0, se2 = 0; const double n = double(rounds.size());
            for (auto& rr : rounds) { sy += rr.ret; sy2 += rr.ret * rr.ret; double e = rr.ret - rr.value; se += e; se2 += e * e; }
            const double var_y = sy2 / n - (sy / n) * (sy / n), var_e = se2 / n - (se / n) * (se / n);
            ev = 1.0 - var_e / (var_y + 1e-8);
        }
        const long upd = std::max(1L, nrec);
        const double pl_mean = pl_sum / std::max(1, n_steps), ent_mean = ent_sum / std::max(1, n_steps), vl_mean = vl_sum / std::max(1, EPOCHS);
        const long clip_total = std::max(1L, recs * EPOCHS);
        double mean_rounds = 0; int ended_a = 0;
        std::map<std::string, int> by_cond;
        for (auto& e : ended) { mean_rounds += e.rounds; ended_a += e.winner == 0; by_cond[e.cond]++; }
        if (!ended.empty()) mean_rounds /= ended.size();
        (void)upd;

        std::ostringstream o;
        o << "---- iter " << it << "   lr " << K.lr << "   epochs " << EPOCHS << "   game_rounds " << K.game_rounds
          << "   ore_radius " << K.ore_radius << "\n";
        o << "  " << lfmt::label("rollout") << G << " games x " << R << " rounds   " << rows.size() << " decisions   "
          << recs << " head records   " << std::fixed << std::setprecision(1) << ms_roll / 1000 << " s\n" << std::defaultfloat;
        o << "  " << lfmt::label("ppo") << "policy loss" << lfmt::num(pl_mean, 10) << "    value loss" << lfmt::num(vl_mean, 10)
          << "    mean entropy" << lfmt::num(ent_mean, 8) << "    " << std::fixed << std::setprecision(1) << ms_upd / 1000 << " s\n" << std::defaultfloat;
        o << "  " << lfmt::label("") << "ratio dev (1st minibatch)" << lfmt::num(ratio_dev0, 10)
          << "    clipped " << lfmt::ratio(clipped, clip_total) << "\n";
        o << "  " << lfmt::label("advantage") << "mean" << lfmt::num(adv_mean, 9) << "    std" << lfmt::num(adv_std, 9)
          << "    min" << lfmt::num(adv_min, 9) << "    max" << lfmt::num(adv_max, 9) << "   (raw, before normalizing)\n";
        o << "  " << lfmt::label("critic") << "explained variance " << lfmt::num(ev, 6) << "\n";
        if (!ended.empty()) {
            double ti[2] = {0, 0}, hv[2] = {0, 0};
            for (auto& e : ended) for (int t = 0; t < 2; t++) { ti[t] += e.ti_col[t]; hv[t] += e.harv[t]; }
            o << "  " << lfmt::label("ended") << ended.size() << " games   rounds " << std::fixed << std::setprecision(0) << mean_rounds
              << "   A wins " << lfmt::ratio(ended_a, long(ended.size()))
              << "   Ti collected A " << ti[0] / ended.size() << "  B " << ti[1] / ended.size()
              << "   harvesters A " << std::setprecision(1) << hv[0] / ended.size() << "  B " << hv[1] / ended.size() << std::defaultfloat << "\n";
            o << "  " << lfmt::label("");
            for (auto& [c, n] : by_cond) o << c << " " << n << "   ";
            o << "(all runs: " << finished << " games, A " << a_wins << ")\n";
        }
        if (it == 1 || it % 10 == 0) {
            o << "  " << lfmt::label("heads") << lfmt::label("head", 20) << "  entropy     coef   records\n";
            for (int h = 0; h < NUM_HEAD_IDS; h++) {
                o << "  " << lfmt::label("") << lfmt::label(HEAD_NAMES[h], 20);
                if (head_n[h]) o << lfmt::num(head_ent[h] / head_n[h], 9) << lfmt::num(K.head_entropy[h], 9) << std::setw(10) << head_n[h] << "\n";
                else o << "  (no records this iteration)\n";
            }
        }
        std::cout << o.str() << std::flush;

        ppo_log << it << "," << rows.size() << "," << recs << "," << ended.size() << "," << ended_a << "," << mean_rounds
                << "," << pl_mean << "," << vl_mean << "," << ent_mean << "," << ev << "," << adv_mean << "," << adv_std
                << "," << adv_min << "," << adv_max << "," << ratio_dev0 << "," << 100.0 * clipped / clip_total
                << "," << K.lr << "," << EPOCHS << "," << K.game_rounds << "," << K.ore_radius
                << "," << ms_roll / 1000 << "," << ms_upd / 1000;
        for (int h = 0; h < NUM_HEAD_IDS; h++) { ppo_log << ","; if (head_n[h]) ppo_log << head_ent[h] / head_n[h]; }
        ppo_log << std::endl;
        auto game_row = [&](const EndStat& e, int greedy) {
            games_log << it << "," << (K.ore_radius >= 0 ? "ore_r" + std::to_string(K.ore_radius) : std::string("map")) << "," << e.rounds << "," << e.winner << "," << e.cond
                      << "," << e.ti_col[0] << "," << e.ti_col[1] << "," << e.vo_col[0] << "," << e.vo_col[1]
                      << "," << e.harv[0] << "," << e.harv[1] << "," << e.units[0] << "," << e.units[1] << "," << greedy << "\n";
        };
        for (auto& e : ended) game_row(e, 0);
        games_log.flush();
        if (it % env_int("SAVE_EVERY", 25) == 0 || it == iterations) torch::save(net, out);
        if (K.replay_every > 0 && it % K.replay_every == 0) {
            // one greedy self-play game, logged for rl/tools/make_replay.py
            auto tg = std::chrono::steady_clock::now();
            fs::path dir = run_dir / "replays";
            fs::create_directories(dir);
            char name[32]; std::snprintf(name, sizeof name, "iter_%06d", it);
            Match gm;
            const std::string mp = maps[master() % maps.size()];
            const uint64_t seed = master();
            gm.load(mp, seed); gm.reset();
            gm.game.max_turns = K.game_rounds;
            gm.game.limit_ore(K.ore_radius, master());
            std::vector<ActionLine> glog;
            net->eval();
            play_greedy(gm, [&](int p, const std::vector<float>& obs) {
                torch::NoGradGuard ng;
                std::vector<int16_t> o16(obs.begin(), obs.end());
                torch::Tensor lt = net->logits(p, actor_tensor(o16)).squeeze(0).contiguous();
                const float* lp = lt.data_ptr<float>();
                return std::vector<float>(lp, lp + lt.numel());
            }, glog);
            const std::string mapf = (dir / (std::string(name) + ".map26")).string();
            const std::string gamef = (dir / (std::string(name) + ".game")).string();
            write_map(gm.game, mp, mapf);
            write_game(gamef, mapf, seed, gm.game, glog);
            const EndStat ge = end_stat(gm.game);
            game_row(ge, 1); games_log.flush();
            std::cout << "  " << lfmt::label("greedy") << fs::path(mp).filename().string() << "   rounds " << ge.rounds
                      << "   winner " << (ge.winner == 0 ? "A" : ge.winner == 1 ? "B" : "-") << " (" << ge.cond << ")"
                      << "   Ti collected A " << ge.ti_col[0] << "  B " << ge.ti_col[1]
                      << "   harvesters A " << ge.harv[0] << "  B " << ge.harv[1] << "   "
                      << std::fixed << std::setprecision(1)
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count() << " s\n"
                      << std::defaultfloat
                      << "  " << lfmt::label("") << "log " << gamef << "  (rl/tools/make_replay.py -> .replay26)" << std::endl;
        }
    }
    std::cout << "TRAIN COMPLETE checkpoint=" << out << std::endl;
}
