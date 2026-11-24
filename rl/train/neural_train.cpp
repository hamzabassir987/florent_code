// PPO self-play trainer for the float teacher, with factorized heads.
//
// Policy semantics (locked, see SESSION_HANDOFF.md):
//  * every head is its own masked categorical;
//  * one record per participating head: head id, legal mask, choice, old logp;
//  * new logp recomputed per head; ratio and clipping per head; the same
//    team-round advantage applies to every head of that decision;
//  * losses are averaged over head records; log-probs are never summed.
//
// Rollouts (batched): the GAMES training games live ACROSS iterations; each
// iteration plays ROUNDS more rounds of every one (restarting games that end),
// then bootstraps from the critic at the cut. All games advance together: a
// step encodes the next deciding unit of every game (OpenMP, game i on thread
// i % T), runs one forward per policy over all of them, and every game plays
// its unit's decision. Units act in order inside a round, so a step takes one
// unit per game. The critic is deferred: its rows are queued and evaluated in
// big batches (the weights do not change during a rollout). Terminal reward
// only (+1 / -1 / 0). Self-play: one network plays both teams.
//
// PPO: advantages normalized over the batch then clipped to +-ADV_CLIP;
// per-head TARGET entropy (coef * H / H.detach(): each head's push has the
// same size whatever its number of options); separate gradient clipping for
// the policy (actors) and the value (critic); with epochs > 1 a minibatch
// whose KL(old || new) is over KL_STOP ends the iteration's updates.
//
// Eval: every eval_every iterations, eval_games games of the live net against
// the frozen one (half on each side), sampled then greedy; >= PROMOTE greedy
// wins promotes the live net. Greedy game 0 is logged to replays/
// (rl/tools/make_replay.py turns it into an official .replay26).
//
//   neural_train ITERATIONS OUT.pt MAP [MAP...]
//   env: GAMES (950) ROUNDS (150) N_MINI (15 minibatches a pass) VALUE_KEEP (0.1)
//        MICRO (512 rows per forward/backward) CRITIC_BATCH (1024 rows per critic inference)
//        RECORD_P (0.05: share of decisions kept for PPO; ~17 KB each) INIT (checkpoint) START_ITER (0) FROZEN (checkpoint)
//        THREADS (all) DEVICE (cuda if available | cpu) SNAP_EVERY (400)
//        SAVE_EVERY (10) ENTROPY_FILE (./entropy_file)
// Live knobs in the entropy file, see entropy_file.hpp.
#include <omp.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../sim/florent_game.hpp"
#include "../sim/florent_heads.hpp"
#include "entropy_file.hpp"
#include "game_log.hpp"
#include "model.hpp"
#include "trajectory.hpp"

using namespace florent_rl;
namespace fs = std::filesystem;

static int env_int(const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; }
static float env_float(const char* k, float d) { const char* v = std::getenv(k); return v ? float(std::atof(v)) : d; }

constexpr double INITIAL_LR = 3e-5, FINAL_LR = 1e-5;
constexpr double ADV_CLIP = 6.0;
constexpr double KL_STOP = 0.03;
constexpr double CLIP_POL = 1.0, CLIP_VAL = 1.0;    // gradient-norm clip per model
constexpr float VALUE_COEF = 0.5f;
constexpr float PROMOTE = 0.85f;
static int CRITIC_BATCH = 4000;                      // rows per critic inference forward (env CRITIC_BATCH)
static int MICRO = 4000;                              // rows per forward/backward when training (env MICRO)
constexpr int CRITIC_QUEUE = 40000;                 // queued critic rows before an early flush

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

// CSV files next to the checkpoint; appended when the run resumes (START_ITER > 0).
static std::ofstream open_log(const fs::path& p, const std::string& header, bool append) {
    const bool fresh = !append || !fs::exists(p);
    std::ofstream f(p, fresh ? std::ios::trunc : std::ios::app);
    if (fresh) f << header << "\n";
    return f;
}

static const char* KIND_NAMES[] = {"wait", "move", "spawn", "convert", "build", "attack", "heal", "destroy", "fire", "rotate", "launch"};
constexpr int NUM_KINDS = 11;

// ===========================================================================
// GAMES
// ===========================================================================
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

struct Rec { int head, choice; std::vector<uint8_t> mask; float old_logp; };

// One recorded decision. The spatial input is stored as uint8 (every actor
// channel is 0..255), the scalars as int16.
struct Sample {
    int policy, team;
    int seg, rnd;                  // where its round lives in its slot (for the advantage)
    std::vector<uint8_t> sp;
    std::array<int16_t, fe::ACTOR_SCALARS> sc;
    std::vector<Rec> recs;
    float adv = 0;
};

// One team-round of a game segment (team-0 frame for the value).
struct RoundE {
    float value = 0, reward = 0, adv[2] = {0, 0}, ret0 = 0;
    bool terminal = false;
    int keep = -1;                 // -2: keep at the flush; >= 0: index into the kept value rows
};
struct Segment {                    // one game's rounds inside this stretch
    std::vector<RoundE> rounds;
    bool terminal = false;
    float boot = 0;                // value after the last round if not terminal
};

// A critic row waiting for its value.
struct CriticRow { std::vector<uint8_t> sp; std::vector<int32_t> sc; int seg, rnd; bool boot; };

struct Slot {
    Match m;
    std::mt19937_64 rng;
    bool started = false, done = false, paused = false;
    std::vector<int> order; size_t pos = 0; bool in_round = false;
    int start_round = 0;
    int played = 0;                                  // rounds this stretch
    // the unit waiting for a forward
    bool need = false; int ent = -1, policy = -1, net = 0; std::vector<float> obs, logits;
    // play setup
    int net_of[2] = {0, 0}; bool greedy = false, record = true;
    std::vector<ActionLine>* log = nullptr;
    std::string map_path; uint64_t seed = 0;
    // outputs
    std::vector<Segment> segs;
    std::vector<Sample> samples;
    std::vector<CriticRow> crit;
    std::vector<EndStat> ended;
    long decisions = 0, oob = 0;
    long tried[NUM_KINDS] = {}, refused[NUM_KINDS] = {};
};

struct Setup {
    std::vector<std::string> maps;
    Knobs* K = nullptr;
    float record_p = 1.f, value_keep = 0.25f;
};

static void advance(Slot& s, const Setup& S, int stretch, bool restart, bool train);

static void start_game(Slot& s, const Setup& S) {
    s.m = Match();
    s.map_path = S.maps[s.rng() % S.maps.size()];
    s.seed = s.rng() & 0xFFFFFFFFull;
    if (!s.m.load(s.map_path, s.seed)) { std::cerr << "bad map " << s.map_path << "\n"; std::exit(3); }
    s.m.reset();
    s.m.game.round = s.start_round;
    s.start_round = 0;              // staggered once (the games of iteration 0); every restart starts at round 0
    s.m.game.max_turns = S.K->game_rounds;
    s.m.game.limit_ore(S.K->ore_radius, s.rng());
    s.started = true; s.in_round = false; s.done = false;
    s.segs.push_back(Segment());
}

// Advance one game until its next unit needs a forward, the stretch ends
// (paused) or the game ends without restart (done).
static void advance(Slot& s, const Setup& S, int stretch, bool restart, bool train) {
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    for (;;) {
        if (!s.in_round) {
            if (s.m.game.over) {
                s.segs.back().terminal = true;
                s.ended.push_back(end_stat(s.m.game));
                if (!restart) { s.done = true; return; }
                start_game(s, S);
            }
            if (stretch > 0 && s.played >= stretch) { s.paused = true; return; }
            if (train) {                              // round start: the critic row of this team-round
                s.m.round_critic();
                Segment& sg = s.segs.back();
                CriticRow cr;
                cr.sp.assign(s.m.critic_channels.begin(), s.m.critic_channels.end());
                cr.sc = s.m.critic_scalars;
                cr.seg = int(s.segs.size()) - 1; cr.rnd = int(sg.rounds.size()); cr.boot = false;
                sg.rounds.push_back(RoundE());
                if (u01(s.rng) < S.value_keep) sg.rounds.back().keep = -2;
                s.crit.push_back(std::move(cr));
            }
            s.order = s.m.acting_units(); s.pos = 0; s.in_round = true;
        }
        Match& m = s.m;
        while (s.pos < s.order.size()) {
            const int a = s.order[s.pos];
            if (a >= int(m.game.ents.size()) || !m.game.ents[a].alive) { s.pos++; continue; }
            const int type = m.game.ents[a].type;
            if (type == fe::SPIKE) {                   // deterministic, as deployed
                Candidate c;
                if (spike_action(m.game, a, c) && m.act(a, c) == fe::OK && s.log) s.log->push_back({m.game.round, m.game.ents[a].id, c});
                s.pos++; continue;
            }
            const int p = policy_of_type(type);
            if (p < 0) { s.pos++; continue; }
            m.observe(a);                              // memory updates every turn, as deployed
            if (!has_choice(m.game, a)) { s.pos++; continue; }   // only wait: no forward
            m.encode_only(a, s.obs);
            s.need = true; s.ent = a; s.policy = p; s.net = s.net_of[m.game.ents[a].team];
            return;
        }
        m.game.end_round();
        s.played++;
        s.in_round = false;
        if (train) {
            RoundE& re = s.segs.back().rounds.back();
            re.terminal = m.game.over;
            // a coin-flip finish is a draw in training: reward 0 for both teams
            if (m.game.over && m.game.win_condition != "coinflip") re.reward = float(m.terminal_reward(0));
        }
    }
}

// Compact uploads: the critic's one-hot board as uint8, its scalars as int32;
// the float conversion happens on the device (Teacher::value_split).
static void critic_upload(const std::vector<const std::vector<uint8_t>*>& sps, const std::vector<const std::vector<int32_t>*>& scs,
                          torch::Device dev, torch::Tensor& tsp, torch::Tensor& tsc) {
    const long n = long(sps.size());
    tsp = torch::empty({n, fe::CRITIC_SPATIAL}, torch::kUInt8);
    tsc = torch::empty({n, fe::CRITIC_SCALARS}, torch::kInt32);
    uint8_t* a = tsp.data_ptr<uint8_t>(); int32_t* b = tsc.data_ptr<int32_t>();
    for (long i = 0; i < n; i++) {
        std::memcpy(a + i * fe::CRITIC_SPATIAL, sps[i]->data(), fe::CRITIC_SPATIAL);
        std::memcpy(b + i * fe::CRITIC_SCALARS, scs[i]->data(), sizeof(int32_t) * fe::CRITIC_SCALARS);
    }
    tsp = tsp.to(dev); tsc = tsc.to(dev);
}
// The actor's input as stored and as played: spatial clamped to 0..255 (uint8),
// scalars to int16. The rollout plays on exactly what PPO later trains on.
static void actor_compact(const std::vector<float>& obs, uint8_t* sp, int16_t* sc, long& oob) {
    for (int i = 0; i < fe::ACTOR_SPATIAL; i++) {
        float v = obs[i];
        if (v < 0 || v > 255) { oob++; v = std::clamp(v, 0.f, 255.f); }
        sp[i] = uint8_t(v);
    }
    for (int i = 0; i < fe::ACTOR_SCALARS; i++) sc[i] = int16_t(std::clamp(obs[fe::ACTOR_SPATIAL + i], -32768.f, 32767.f));
}

static int sample_masked(const float* l, const std::vector<uint8_t>& mask, std::mt19937_64& rng, float& lp) {
    double mx = -1e30;
    for (size_t i = 0; i < mask.size(); i++) if (mask[i]) mx = std::max(mx, double(l[i]));
    std::vector<double> w(mask.size(), 0.0); double z = 0;
    for (size_t i = 0; i < mask.size(); i++) if (mask[i]) { w[i] = std::exp(double(l[i]) - mx); z += w[i]; }
    double u = std::uniform_real_distribution<double>(0.0, z)(rng), c = 0;
    int pick = -1;
    for (size_t i = 0; i < mask.size(); i++) if (mask[i]) { pick = int(i); c += w[i]; if (u <= c) break; }
    lp = masked_logp(l, mask, pick);
    return pick;
}

// Play the forward's result for the waiting unit.
static void play_decision(Slot& s, const Setup& S, bool train) {
    Match& m = s.m;
    const int a = s.ent;
    Decision d = decide(m.game, a, s.logits.data(), [&](int, const float* l, const std::vector<uint8_t>& mk, float& lp) {
        return s.greedy ? argmax_choice(l, mk, lp) : sample_masked(l, mk, s.rng, lp);
    });
    for (auto& c : d.actions) {
        const int k = int(c.kind);
        s.tried[k]++;
        if (m.act(a, c) != fe::OK) { s.refused[k]++; continue; }
        if (s.log) s.log->push_back({m.game.round, m.game.ents[a].id, c});
    }
    s.decisions++;
    if (train && s.record && std::uniform_real_distribution<float>(0.f, 1.f)(s.rng) < S.record_p) {
        Sample smp;
        smp.policy = s.policy; smp.team = m.game.ents[a].team;
        smp.seg = int(s.segs.size()) - 1; smp.rnd = int(s.segs.back().rounds.size()) - 1;
        smp.sp.resize(fe::ACTOR_SPATIAL);
        long dummy = 0;
        actor_compact(s.obs, smp.sp.data(), smp.sc.data(), dummy);
        for (auto& hr : d.records) smp.recs.push_back({hr.head, hr.choice, hr.mask, hr.logp});
        s.samples.push_back(std::move(smp));
    }
    s.need = false; s.pos++;
}

struct PlayStats { long steps = 0; double t_encode = 0, t_forward = 0, t_act = 0, t_critic = 0; long critic_rows = 0; };

// Kept value rows (for the value loss): the critic input and the return.
struct ValueRow { std::vector<uint8_t> sp; std::vector<int32_t> sc; float ret = 0; int slot, seg, rnd; };

// Evaluate every queued critic row and write the values back; rows marked to
// keep move to `kept`.
static void flush_critic(std::vector<Slot>& slots, Teacher& net, torch::Device dev, std::vector<ValueRow>& kept, PlayStats& st) {
    auto c0 = std::chrono::steady_clock::now();
    std::vector<std::pair<int, CriticRow*>> all;
    for (int i = 0; i < int(slots.size()); i++) for (auto& cr : slots[i].crit) all.push_back({i, &cr});
    c10::InferenceMode im;
    for (size_t lo = 0; lo < all.size(); lo += CRITIC_BATCH) {
        const size_t hi = std::min(all.size(), lo + CRITIC_BATCH);
        std::vector<const std::vector<uint8_t>*> sps; std::vector<const std::vector<int32_t>*> scs;
        for (size_t k = lo; k < hi; k++) { sps.push_back(&all[k].second->sp); scs.push_back(&all[k].second->sc); }
        torch::Tensor tsp, tsc; critic_upload(sps, scs, dev, tsp, tsc);
        auto v = net->value_split(tsp, tsc).to(torch::kCPU).contiguous();
        const float* vp = v.data_ptr<float>();
        for (size_t k = lo; k < hi; k++) {
            CriticRow& cr = *all[k].second;
            Slot& s = slots[all[k].first];
            if (cr.boot) { s.segs[cr.seg].boot = vp[k - lo]; continue; }
            RoundE& re = s.segs[cr.seg].rounds[cr.rnd];
            re.value = vp[k - lo];
            if (re.keep == -2) {
                re.keep = int(kept.size());
                kept.push_back({std::move(cr.sp), std::move(cr.sc), 0.f, all[k].first, cr.seg, cr.rnd});
            }
        }
        st.critic_rows += long(hi - lo);
    }
    for (auto& s : slots) { s.crit.clear(); s.crit.shrink_to_fit(); }
    st.t_critic += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
}

// Run all slots until each is paused or done. nets[k] plays net index k.
static PlayStats play(std::vector<Slot>& slots, const Setup& S, const std::vector<Teacher*>& nets, torch::Device dev,
                      int stretch, bool restart, bool train, std::vector<ValueRow>* kept) {
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    PlayStats st;
    const int G = int(slots.size());
    for (auto& s : slots) { s.paused = false; s.played = 0; }
    c10::InferenceMode im;
    for (;;) {
        auto t0 = clk::now();
        #pragma omp parallel for schedule(static, 1)
        for (int i = 0; i < G; i++) {
            Slot& s = slots[i];
            if (!s.need && !s.done && !s.paused) advance(s, S, stretch, restart, train);
        }
        auto t1 = clk::now();
        st.t_encode += secs(t0, t1);
        // one forward per (net, policy) over every waiting game
        bool any = false;
        for (int k = 0; k < int(nets.size()); k++)
            for (int p = 0; p < NUM_POLICIES; p++) {
                std::vector<int> idx;
                for (int i = 0; i < G; i++) if (slots[i].need && slots[i].net == k && slots[i].policy == p) idx.push_back(i);
                if (idx.empty()) continue;
                any = true;
                const long n = long(idx.size());
                auto tsp = torch::empty({n, fe::ACTOR_SPATIAL}, torch::kUInt8);
                auto tsc = torch::empty({n, fe::ACTOR_SCALARS}, torch::kInt16);
                for (long j = 0; j < n; j++)
                    actor_compact(slots[idx[j]].obs, tsp.data_ptr<uint8_t>() + j * fe::ACTOR_SPATIAL,
                                  tsc.data_ptr<int16_t>() + j * fe::ACTOR_SCALARS, slots[idx[j]].oob);
                auto l = (*nets[k])->logits_split(p, tsp.to(dev), tsc.to(dev)).to(torch::kCPU).contiguous();
                const long L = l.size(1);
                const float* lp = l.data_ptr<float>();
                for (long j = 0; j < n; j++) slots[idx[j]].logits.assign(lp + j * L, lp + (j + 1) * L);
                st.steps++;
            }
        auto t2 = clk::now();
        st.t_forward += secs(t1, t2);
        if (!any) break;
        #pragma omp parallel for schedule(static, 1)
        for (int i = 0; i < G; i++) if (slots[i].need) play_decision(slots[i], S, train);
        st.t_act += secs(t2, clk::now());
        if (train && kept) {
            size_t q = 0; for (auto& s : slots) q += s.crit.size();
            if (q >= size_t(CRITIC_QUEUE)) flush_critic(slots, *nets[0], dev, *kept, st);
        }
    }
    if (train && kept) {
        // bootstrap rows for the segments cut by the stretch
        for (int i = 0; i < G; i++) {
            Slot& s = slots[i];
            if (s.done || s.segs.empty() || s.segs.back().terminal || s.segs.back().rounds.empty()) continue;
            s.m.round_critic();
            CriticRow cr; cr.sp.assign(s.m.critic_channels.begin(), s.m.critic_channels.end()); cr.sc = s.m.critic_scalars;
            cr.seg = int(s.segs.size()) - 1; cr.rnd = -1; cr.boot = true;
            s.crit.push_back(std::move(cr));
        }
        flush_critic(slots, *nets[0], dev, *kept, st);
    }
    return st;
}

static void copy_model(Teacher& from, Teacher& to) {
    torch::NoGradGuard ng;
    auto src = from->named_parameters(), dst = to->named_parameters();
    for (auto& kv : src) dst[kv.key()].copy_(kv.value());
    auto sb = from->named_buffers(), db = to->named_buffers();
    for (auto& kv : sb) db[kv.key()].copy_(kv.value());
}

// ===========================================================================
// MAIN
// ===========================================================================
int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "usage: neural_train ITERATIONS OUT.pt MAP [MAP...]\n"; return 2; }
    const int iterations = std::atoi(argv[1]);
    const std::string out = argv[2];
    Setup S;
    S.maps.assign(argv + 3, argv + argc);
    const int G = env_int("GAMES", 100), R = env_int("ROUNDS", 100);
    const int N_MINI = env_int("N_MINI", 10);      // minibatches per pass (reference pipeline: 15)
    // Full-batch learner: do not split PPO minibatches into MICRO chunks.
    MICRO = env_int("MICRO", std::numeric_limits<int>::max());   // default: no split
    CRITIC_BATCH = std::max(1, env_int("CRITIC_BATCH", 4000));
    S.value_keep = env_float("VALUE_KEEP", 1.0f);
    S.record_p = env_float("RECORD_P", 1.0f);
    const int start_iter = env_int("START_ITER", 0);
    if (std::getenv("THREADS")) { omp_set_num_threads(env_int("THREADS", 1)); torch::set_num_threads(env_int("THREADS", 1)); }
    const int T = omp_get_max_threads();

    torch::Device dev(torch::kCPU);
    {
        const char* d = std::getenv("DEVICE");
        if ((!d || std::string(d) == "cuda") && torch::cuda::is_available()) dev = torch::Device(torch::kCUDA);
    }
    torch::manual_seed(7);

    const std::string KNOB_PATH = std::getenv("ENTROPY_FILE") ? std::getenv("ENTROPY_FILE") : "entropy_file";
    Knobs K;
    K.lr = INITIAL_LR; K.epochs = 1;
    for (auto& e : K.head_entropy) e = 0.008f;
    if (!read_entropy_file(KNOB_PATH, K)) std::cout << "train: no " << KNOB_PATH << ", defaults" << std::endl;
    S.K = &K;

    const fs::path run_dir = fs::path(out).parent_path().empty() ? fs::path(".") : fs::path(out).parent_path();
    Teacher net, frozen;
    if (const char* init = std::getenv("INIT")) { torch::load(net, init); std::cout << "train: loaded " << init << std::endl; }
    copy_model(net, frozen);
    {
        const std::string fp = std::getenv("FROZEN") ? std::getenv("FROZEN") : (run_dir / "frozen.pt").string();
        if ((std::getenv("FROZEN") || start_iter > 0) && fs::exists(fp)) { torch::load(frozen, fp); std::cout << "train: frozen net from " << fp << std::endl; }
    }
    net->to(dev); frozen->to(dev); frozen->eval();
    torch::save(frozen, (run_dir / "frozen.pt").string());

    std::vector<torch::Tensor> pol_params, val_params;
    for (auto& kv : net->named_parameters()) (kv.key().rfind("critic", 0) == 0 ? val_params : pol_params).push_back(kv.value());
    torch::optim::Adam opt_pol(pol_params, torch::optim::AdamOptions(K.lr));
    torch::optim::Adam opt_val(val_params, torch::optim::AdamOptions(K.lr));

    // logs
    const bool resumed = start_iter > 0;
    std::string ppo_header = "iter,samples,head_records,ended,a_wins,mean_rounds,policy_loss,value_loss,entropy,kl,kl_stop,"
                             "explained_var,adv_mean,adv_std,adv_min,adv_max,clipped_pct,lr,epochs,game_rounds,ore_radius,rollout_s,update_s";
    for (int h = 0; h < NUM_HEAD_IDS; h++) ppo_header += std::string(",H_") + HEAD_NAMES[h];
    std::ofstream ppo_log = open_log(run_dir / "ppo_logs.csv", ppo_header, resumed);
    std::ofstream games_log = open_log(run_dir / "games.csv",
        "iter,map_kind,rounds,winner,condition,ti_collected_a,ti_collected_b,vo_collected_a,vo_collected_b,harvesters_a,harvesters_b,units_a,units_b", resumed);
    std::ofstream grad_log = open_log(run_dir / "grad_norms.csv", "iter,pol_steps,pol_mean,pol_max,pol_clipped,val_steps,val_mean,val_max,val_clipped", resumed);
    std::string ref_header = "iter";
    for (int k = 1; k < NUM_KINDS; k++) ref_header += std::string(",") + KIND_NAMES[k] + "_tried," + KIND_NAMES[k] + "_refused";
    std::ofstream refusal_log = open_log(run_dir / "refusal_logs.csv", ref_header, resumed);
    std::ofstream winrate_log = open_log(run_dir / "winrate_logs.csv", "iter,mode,live_wins,games,winrate,mean_rounds,promoted", resumed);

    {
        long np = 0, nv = 0;
        for (auto& t : pol_params) np += t.numel();
        for (auto& t : val_params) nv += t.numel();
        std::cout << "train: " << S.maps.size() << " maps, " << G << " games x " << R << " rounds an iteration, " << N_MINI << " minibatches a pass, micro-batch " << MICRO << ", critic batch " << CRITIC_BATCH
                  << ", device " << (dev.is_cuda() ? "cuda" : "cpu") << ", " << T << " threads\n"
                  << "train: params policy " << np << ", value " << nv << ", value rows kept " << S.value_keep
                  << ", decisions recorded " << S.record_p << "\n"
                  << "train: lr " << INITIAL_LR << " -> " << FINAL_LR << " over " << iterations << " iterations (an lr line in "
                  << KNOB_PATH << " overrides), gamma " << GAMMA << ", lambda " << LAMBDA << ", clip " << CLIP_EPS
                  << ", adv clip " << ADV_CLIP << ", kl stop " << KL_STOP << "\n"
                  << "train: logs in " << run_dir.string() << " (ppo_logs, games, grad_norms, refusal_logs, winrate_logs .csv)" << std::endl;
        if (start_iter > 0) std::cout << "train: continuing from iteration " << start_iter << std::endl;
    }

    std::vector<Slot> slots(G);
    for (int i = 0; i < G; i++) slots[i].rng.seed(0x5EED0000ull + uint64_t(i) * 7919ull + uint64_t(start_iter) * 104729ull);
    long finished = 0, a_wins_all = 0;

    for (int it = start_iter + 1; it <= iterations; it++) {
        const auto t_start = std::chrono::steady_clock::now();
        const double progress = double(it) / std::max(1, iterations);
        K.lr = INITIAL_LR + (FINAL_LR - INITIAL_LR) * progress;
        read_entropy_file(KNOB_PATH, K);
        for (auto* o : {&opt_pol, &opt_val})
            for (auto& pg : o->param_groups()) static_cast<torch::optim::AdamOptions&>(pg.options()).lr(K.lr);
        for (auto& s : slots) if (s.started) s.m.game.max_turns = K.game_rounds;   // running games too
        const int EPOCHS = K.epochs;

        // ---- rollout
        for (int i = 0; i < G; i++) {
            Slot& s = slots[i];
            s.segs.clear(); s.samples.clear(); s.crit.clear(); s.ended.clear();
            s.decisions = 0; s.oob = 0;
            std::fill(std::begin(s.tried), std::end(s.tried), 0L); std::fill(std::begin(s.refused), std::end(s.refused), 0L);
            s.greedy = false; s.record = true; s.log = nullptr; s.net_of[0] = s.net_of[1] = 0;
            if (!s.started) {
                const int game_len = std::max(1, S.K->game_rounds);
                const int step = env_int("START_TURN_STEP", 10);
                s.start_round = ((i % game_len) * (step % game_len)) % game_len;
                start_game(s, S);
            }
            else s.segs.push_back(Segment());      // the running game continues in a new segment
        }
        net->eval();
        std::vector<ValueRow> kept;
        const PlayStats ps = play(slots, S, {&net}, dev, R, /*restart=*/true, /*train=*/true, &kept);

        // ---- GAE per segment and team (team 1 sees -value)
        for (auto& s : slots)
            for (auto& sg : s.segs) {
                const size_t n = sg.rounds.size();
                if (!n) continue;
                for (int team = 0; team < 2; team++) {
                    const float sgn = team == 0 ? 1.f : -1.f;
                    std::vector<RoundValue> seq(n);
                    for (size_t i = 0; i < n; i++) {
                        seq[i].value = sgn * sg.rounds[i].value;
                        seq[i].reward = sgn * sg.rounds[i].reward;
                        seq[i].terminal = sg.rounds[i].terminal;
                        seq[i].next_value = i + 1 < n ? sgn * sg.rounds[i + 1].value : (sg.terminal ? 0.f : sgn * sg.boot);
                    }
                    std::vector<RoundValue*> ptr; for (auto& v : seq) ptr.push_back(&v);
                    finish_gae(ptr, GAMMA, LAMBDA, false);
                    for (size_t i = 0; i < n; i++) {
                        sg.rounds[i].adv[team] = seq[i].advantage;
                        if (team == 0) sg.rounds[i].ret0 = seq[i].ret;
                    }
                }
            }
        for (auto& v : kept) v.ret = slots[v.slot].segs[v.seg].rounds[v.rnd].ret0;
        // ---- gather the samples; advantages normalized, then clipped
        std::vector<Sample> samples;
        std::vector<EndStat> ended;
        long decisions = 0, oob = 0, tried[NUM_KINDS] = {}, refused[NUM_KINDS] = {};
        for (int i = 0; i < G; i++) {
            Slot& s = slots[i];
            for (auto& smp : s.samples) { smp.adv = s.segs[smp.seg].rounds[smp.rnd].adv[smp.team]; samples.push_back(std::move(smp)); }
            s.samples.clear();
            for (auto& e : s.ended) ended.push_back(e);
            decisions += s.decisions; oob += s.oob;
            for (int k = 0; k < NUM_KINDS; k++) { tried[k] += s.tried[k]; refused[k] += s.refused[k]; }
        }
        double adv_mean = 0, adv_std = 0, adv_min = 0, adv_max = 0; long adv_clipped = 0;
        if (!samples.empty()) {
            adv_min = adv_max = samples[0].adv;
            for (auto& x : samples) { adv_mean += x.adv; adv_min = std::min(adv_min, double(x.adv)); adv_max = std::max(adv_max, double(x.adv)); }
            adv_mean /= samples.size();
            for (auto& x : samples) adv_std += (x.adv - adv_mean) * (x.adv - adv_mean);
            adv_std = std::sqrt(adv_std / samples.size());
            for (auto& x : samples) {
                double a = (x.adv - adv_mean) / (adv_std + 1e-8);
                if (std::fabs(a) > ADV_CLIP) { adv_clipped++; a = std::clamp(a, -ADV_CLIP, ADV_CLIP); }
                x.adv = float(a);
            }
        }
        const auto t_roll = std::chrono::steady_clock::now();

        // ---- PPO update
        // Memory: every forward/backward runs on at most MICRO rows; the
        // gradients of a minibatch's chunks add up before its one optimizer
        // step (the loss is a sum over records divided by the minibatch's
        // record count, so the step is the same as one big batch).
        net->train();
        double pl_sum = 0, vl_sum = 0, ent_sum = 0, kl_sum = 0; int n_steps = 0, v_steps = 0;
        long nrec = 0, clipped = 0;
        bool kl_stop = false; int stop_ep = -1; double stop_kl = 0;
        double head_ent[NUM_HEAD_IDS] = {}; long head_n[NUM_HEAD_IDS] = {};
        double gn_sum[2] = {0, 0}, gn_max[2] = {0, 0}; long gn_clip[2] = {0, 0}; int gn_steps[2] = {0, 0};
        auto clip_norm = [&](std::vector<torch::Tensor>& ps, double lim, int g) {
            const double n = torch::nn::utils::clip_grad_norm_(ps, lim);
            gn_sum[g] += n; gn_max[g] = std::max(gn_max[g], n); gn_steps[g]++; if (n > lim) gn_clip[g]++;
        };
        std::vector<int> order(samples.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 shuf(uint32_t(it) * 2654435761u);
        // As the reference pipeline: each pass splits the rollout into N_MINI
        // minibatches; each policy step is followed by one value step on the
        // matching share of the kept team-rounds.
        const size_t MB = std::max<size_t>(1, (samples.size() + N_MINI - 1) / N_MINI);
        const size_t VB = std::max<size_t>(1, (kept.size() + N_MINI - 1) / N_MINI);
        std::vector<int> vo(kept.size()); std::iota(vo.begin(), vo.end(), 0);

        // Per-minibatch sums, kept on the device and read back once per
        // minibatch (no .item() per head).
        struct Acc {
            torch::Tensor surr, kl, clip, H;      // H: [NUM_HEAD_IDS]
            long cnt[NUM_HEAD_IDS] = {}; long n = 0;
            explicit Acc(torch::Device d) {
                surr = torch::zeros({}, d); kl = torch::zeros({}, d); clip = torch::zeros({}, d);
                H = torch::zeros({NUM_HEAD_IDS}, d);
            }
        };
        // One chunk (rows of one policy). Always adds its sums to `acc`. With
        // `grad`, also returns the chunk's loss term:
        //   -surr / n_total  -  (1/present) * sum_h coef_h * Hbar_h / Hbar_h.detach()
        // Hbar_h is head h's mean entropy over the minibatch. If the policy is in
        // one chunk (h_mean == nullptr) it is computed here, exactly; otherwise
        // h_mean comes from a no-grad pass over the whole minibatch, and the
        // chunk's share is H_chunk / (cnt_h * h_mean_h).
        auto chunk = [&](int p, const int* rows, long nr, bool grad, Acc& acc, long n_total,
                         const long* cnt_total, const double* h_mean, int present) -> torch::Tensor {
            auto tsp = torch::empty({nr, fe::ACTOR_SPATIAL}, torch::kUInt8);
            auto tsc = torch::empty({nr, fe::ACTOR_SCALARS}, torch::kInt16);
            for (long b = 0; b < nr; b++) {
                const Sample& sm = samples[rows[b]];
                std::memcpy(tsp.data_ptr<uint8_t>() + b * fe::ACTOR_SPATIAL, sm.sp.data(), fe::ACTOR_SPATIAL);
                std::memcpy(tsc.data_ptr<int16_t>() + b * fe::ACTOR_SCALARS, sm.sc.data(), sizeof(int16_t) * fe::ACTOR_SCALARS);
            }
            auto logits = net->logits_split(p, tsp.to(dev), tsc.to(dev));
            torch::Tensor loss = torch::zeros({}, dev);
            for (int head : policy_heads(p)) {
                const int n = HEAD_SIZE[head];
                std::vector<long> rr, ch; std::vector<uint8_t> mk; std::vector<float> olp, adv;
                for (long b = 0; b < nr; b++)
                    for (auto& rec : samples[rows[b]].recs)
                        if (rec.head == head) {
                            rr.push_back(b); ch.push_back(rec.choice);
                            mk.insert(mk.end(), rec.mask.begin(), rec.mask.end());
                            olp.push_back(rec.old_logp); adv.push_back(samples[rows[b]].adv);
                        }
                if (rr.empty()) continue;
                const long k = long(rr.size());
                auto hl = logits.index_select(0, torch::tensor(rr, torch::kLong).to(dev)).narrow(1, head_offset(p, head), n);
                auto mask = torch::from_blob(mk.data(), {k, n}, torch::kUInt8).to(dev).to(torch::kBool);
                auto lsm = torch::log_softmax(hl.masked_fill(~mask, -1e9f), 1);
                auto new_lp = lsm.gather(1, torch::tensor(ch, torch::kLong).view({k, 1}).to(dev)).squeeze(1);
                auto log_ratio = new_lp - torch::tensor(olp).to(dev);
                auto A = torch::tensor(adv).to(dev);
                auto ratio = torch::exp(log_ratio);
                auto surr = torch::min(ratio * A, torch::clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * A).sum();
                auto H = -(lsm.exp().masked_fill(~mask, 0.f) * lsm.masked_fill(~mask, 0.f)).sum();   // summed over k records
                acc.surr += surr.detach();
                acc.kl += ((ratio - 1) - log_ratio).detach().sum();                 // KL(old || new) estimate
                acc.clip += ((ratio - 1).abs() > CLIP_EPS).sum().to(torch::kFloat32);
                acc.H[head] += H.detach();
                acc.cnt[head] += k; acc.n += k;
                if (grad) {
                    const double w = K.head_entropy[head] / double(std::max(1, present));
                    torch::Tensor ent;
                    if (!h_mean) { auto hm = H / double(k); ent = hm / (hm.detach() + 1e-8); }       // whole head in this chunk
                    else ent = H / (double(cnt_total[head]) * (h_mean[head] + 1e-8));
                    loss = loss - surr / double(n_total) - w * ent;
                }
            }
            return loss;
        };
        auto value_step = [&](size_t j) {
            const size_t v0 = j * VB;
            if (v0 >= vo.size()) return;
            const size_t v1 = std::min(vo.size(), v0 + VB);
            const double n_all = double(v1 - v0);
            opt_val.zero_grad();
            double vl_tot = 0;
            for (size_t c0 = v0; c0 < v1; c0 += MICRO) {
                const size_t c1 = std::min(v1, c0 + MICRO);
                std::vector<const std::vector<uint8_t>*> sps; std::vector<const std::vector<int32_t>*> scs; std::vector<float> tgt;
                for (size_t b = c0; b < c1; b++) { const ValueRow& vr = kept[vo[b]]; sps.push_back(&vr.sp); scs.push_back(&vr.sc); tgt.push_back(vr.ret); }
                torch::Tensor tsp, tsc; critic_upload(sps, scs, dev, tsp, tsc);
                auto se = (net->value_split(tsp, tsc) - torch::tensor(tgt).to(dev)).pow(2).sum();
                (VALUE_COEF * se / n_all).backward();                          // chunk share of the mean
                vl_tot += se.item<double>();
            }
            clip_norm(val_params, CLIP_VAL, 1);
            opt_val.step();
            vl_sum += vl_tot / n_all; v_steps++;
        };
        for (int ep = 0; ep < EPOCHS && !kl_stop; ep++) {
            std::shuffle(order.begin(), order.end(), shuf);
            std::shuffle(vo.begin(), vo.end(), shuf);
            for (size_t s0 = 0, j = 0; s0 < order.size() && !kl_stop; s0 += MB, j++) {
                const size_t s1 = std::min(order.size(), s0 + MB);
                std::vector<int> by_pol[NUM_POLICIES];
                for (size_t k = s0; k < s1; k++) by_pol[samples[order[k]].policy].push_back(order[k]);
                // head record counts and the heads present, from the records (CPU, no forward)
                long cnt_total[NUM_HEAD_IDS] = {}; long n_total = 0; int present = 0;
                for (size_t k = s0; k < s1; k++)
                    for (auto& rec : samples[order[k]].recs) { cnt_total[rec.head]++; n_total++; }
                for (int h = 0; h < NUM_HEAD_IDS; h++) present += cnt_total[h] > 0;
                if (!n_total) continue;
                bool split = false;
                for (int p = 0; p < NUM_POLICIES; p++) split |= by_pol[p].size() > size_t(MICRO);
                // no-grad pass only when needed: the KL check (pass 2+) or a policy split into chunks
                double h_mean[NUM_HEAD_IDS] = {};
                if (ep > 0 || split) {
                    torch::NoGradGuard ng;
                    Acc pre(dev);
                    for (int p = 0; p < NUM_POLICIES; p++)
                        for (size_t c = 0; c < by_pol[p].size(); c += MICRO)
                            chunk(p, by_pol[p].data() + c, long(std::min(by_pol[p].size() - c, size_t(MICRO))), false, pre, 0, nullptr, nullptr, 0);
                    auto r = torch::cat({pre.kl.view({1}), pre.H}).to(torch::kCPU);
                    const float* rp = r.data_ptr<float>();
                    const double kl = rp[0] / double(pre.n);
                    if (ep > 0 && kl > KL_STOP) { kl_stop = true; stop_ep = ep; stop_kl = kl; break; }
                    for (int h = 0; h < NUM_HEAD_IDS; h++) if (pre.cnt[h]) h_mean[h] = rp[1 + h] / double(pre.cnt[h]);
                }
                // the gradient pass; its sums are the logged ones
                Acc acc(dev);
                opt_pol.zero_grad();
                for (int p = 0; p < NUM_POLICIES; p++)
                    for (size_t c = 0; c < by_pol[p].size(); c += MICRO) {
                        auto loss = chunk(p, by_pol[p].data() + c, long(std::min(by_pol[p].size() - c, size_t(MICRO))), true, acc,
                                          n_total, cnt_total, split ? h_mean : nullptr, present);
                        if (loss.requires_grad()) loss.backward();
                    }
                clip_norm(pol_params, CLIP_POL, 0);
                opt_pol.step();
                // one read-back per minibatch
                auto r = torch::cat({acc.surr.view({1}), acc.kl.view({1}), acc.clip.view({1}), acc.H}).to(torch::kCPU);
                const float* rp = r.data_ptr<float>();
                double ent_mean = 0;
                for (int h = 0; h < NUM_HEAD_IDS; h++) {
                    if (!acc.cnt[h]) continue;
                    ent_mean += rp[3 + h] / double(acc.cnt[h]);
                    if (ep == 0) { head_ent[h] += rp[3 + h]; head_n[h] += acc.cnt[h]; }
                }
                pl_sum += -rp[0] / double(acc.n); ent_sum += ent_mean / std::max(1, present);
                kl_sum += rp[1] / double(acc.n); nrec += acc.n; clipped += long(rp[2]); n_steps++;
                value_step(j);
            }
        }
        const auto t_upd = std::chrono::steady_clock::now();
        const double s_roll = std::chrono::duration<double>(t_roll - t_start).count();
        const double s_upd = std::chrono::duration<double>(t_upd - t_roll).count();

        // ---- critic quality: explained variance over the kept rows (values from the rollout)
        double ev = 0;
        if (kept.size() > 1) {
            double sy = 0, sy2 = 0, se = 0, se2 = 0; const double n = double(kept.size());
            for (auto& v : kept) {
                const RoundE& re = slots[v.slot].segs[v.seg].rounds[v.rnd];
                sy += v.ret; sy2 += v.ret * v.ret; const double e = v.ret - re.value; se += e; se2 += e * e;
            }
            ev = 1.0 - (se2 / n - (se / n) * (se / n)) / (sy2 / n - (sy / n) * (sy / n) + 1e-8);
        }

        // ---- log
        long recs = 0; for (auto& x : samples) recs += long(x.recs.size());
        const double pl_mean = pl_sum / std::max(1, n_steps), ent_mean = ent_sum / std::max(1, n_steps);
        const double vl_mean = vl_sum / std::max(1, v_steps), kl_mean = kl_sum / std::max(1, n_steps);
        const long clip_total = std::max(1L, nrec);
        double mean_rounds = 0; int ended_a = 0, draws = 0;
        std::map<std::string, int> by_cond;
        for (auto& e : ended) {
            mean_rounds += e.rounds; by_cond[e.cond]++;
            if (e.cond == "coinflip") draws++; else ended_a += e.winner == 0;
        }
        if (!ended.empty()) mean_rounds /= ended.size();
        finished += long(ended.size()); a_wins_all += ended_a;
        const double mem_gb = double(samples.size()) * (fe::ACTOR_SPATIAL + 2 * fe::ACTOR_SCALARS) / 1e9
                            + double(kept.size()) * (fe::CRITIC_SPATIAL + 4 * fe::CRITIC_SCALARS) / 1e9;

        std::ostringstream o;
        o << "---- iter " << it << "   lr " << K.lr << "   epochs " << EPOCHS << "   game_rounds " << K.game_rounds
          << "   ore_radius " << K.ore_radius << "\n";
        o << "  " << lfmt::label("rollout") << G << " games x " << R << " rounds   " << decisions << " decisions, "
          << samples.size() << " recorded (" << recs << " head records)   " << kept.size() << " value rows   ~"
          << std::fixed << std::setprecision(2) << mem_gb << " GB\n" << std::defaultfloat;
        o << "  " << lfmt::label("profile") << std::fixed << std::setprecision(2) << ps.steps << " forwards   encode+engine "
          << ps.t_encode << " s   forward " << ps.t_forward << " s   decide+act " << ps.t_act << " s   critic " << ps.t_critic
          << " s (" << ps.critic_rows << " rows)\n" << std::defaultfloat;
        o << "  " << lfmt::label("ppo") << "policy loss" << lfmt::num(pl_mean, 10) << "    value loss" << lfmt::num(vl_mean, 10)
          << "    mean entropy" << lfmt::num(ent_mean, 8) << "    kl" << lfmt::num(kl_mean, 10) << "\n";
        o << "  " << lfmt::label("") << "ratio clipped " << lfmt::ratio(clipped, clip_total) << "    "
          << n_steps << " policy steps, " << v_steps << " value steps\n";
        if (kl_stop)
            o << "  " << lfmt::label("kl stop") << "pass " << stop_ep + 1 << ": kl " << lfmt::num(stop_kl, 9) << " > " << KL_STOP
              << ", the rest of the updates skipped\n";
        o << "  " << lfmt::label("advantage") << "mean" << lfmt::num(adv_mean, 9) << "    std" << lfmt::num(adv_std, 9)
          << "    min" << lfmt::num(adv_min, 9) << "    max" << lfmt::num(adv_max, 9)
          << "   (raw)   clipped at +-" << ADV_CLIP << " std: " << adv_clipped << "\n";
        o << "  " << lfmt::label("critic") << "explained variance " << lfmt::num(ev, 6) << "\n";
        o << "  " << lfmt::label("grad") << "policy mean" << lfmt::num(gn_sum[0] / std::max(1, gn_steps[0]), 9)
          << " max" << lfmt::num(gn_max[0], 9) << " clipped " << gn_clip[0] << "/" << gn_steps[0]
          << "    value mean" << lfmt::num(gn_sum[1] / std::max(1, gn_steps[1]), 9) << " max" << lfmt::num(gn_max[1], 9)
          << " clipped " << gn_clip[1] << "/" << gn_steps[1] << "\n";
        {
            long tr = 0, rf = 0;
            for (int k = 1; k < NUM_KINDS; k++) { tr += tried[k]; rf += refused[k]; }
            o << "  " << lfmt::label("refused") << rf << " of " << tr << " engine calls refused";
            for (int k = 1; k < NUM_KINDS; k++) if (refused[k]) o << "   " << KIND_NAMES[k] << " " << refused[k] << "/" << tried[k];
            if (oob) o << "   (" << oob << " actor inputs outside 0..255, clamped)";
            o << "\n";
        }
        if (!ended.empty()) {
            double ti[2] = {0, 0}, hv[2] = {0, 0};
            for (auto& e : ended) for (int t = 0; t < 2; t++) { ti[t] += e.ti_col[t]; hv[t] += e.harv[t]; }
            o << "  " << lfmt::label("ended") << ended.size() << " games   rounds " << std::fixed << std::setprecision(0) << mean_rounds
              << "   A wins " << lfmt::ratio(ended_a, long(ended.size())) << "   draws (coin flip) " << draws
              << "   Ti collected A " << ti[0] / ended.size() << "  B " << ti[1] / ended.size()
              << "   harvesters A " << std::setprecision(1) << hv[0] / ended.size() << "  B " << hv[1] / ended.size() << std::defaultfloat << "\n";
            o << "  " << lfmt::label("");
            for (auto& [c, n] : by_cond) o << c << " " << n << "   ";
            o << "(all runs: " << finished << " games, A " << a_wins_all << ")\n";
        }
        if (it == start_iter + 1 || it % 10 == 0) {
            o << "  " << lfmt::label("heads") << lfmt::label("head", 20) << "  entropy     coef   records\n";
            for (int h = 0; h < NUM_HEAD_IDS; h++) {
                o << "  " << lfmt::label("") << lfmt::label(HEAD_NAMES[h], 20);
                if (head_n[h]) o << lfmt::num(head_ent[h] / head_n[h], 9) << lfmt::num(K.head_entropy[h], 9) << std::setw(10) << head_n[h] << "\n";
                else o << "  (no records this iteration)\n";
            }
        }
        o << "  " << lfmt::label("time") << std::fixed << std::setprecision(1) << s_roll + s_upd << " s (rollout " << s_roll
          << " s, update " << s_upd << " s)\n" << std::defaultfloat;
        std::cout << o.str() << std::flush;

        ppo_log << it << "," << samples.size() << "," << recs << "," << ended.size() << "," << ended_a << "," << mean_rounds
                << "," << pl_mean << "," << vl_mean << "," << ent_mean << "," << kl_mean << "," << int(kl_stop) << "," << ev
                << "," << adv_mean << "," << adv_std << "," << adv_min << "," << adv_max << "," << 100.0 * clipped / clip_total
                << "," << K.lr << "," << EPOCHS << "," << K.game_rounds << "," << K.ore_radius << "," << s_roll << "," << s_upd;
        for (int h = 0; h < NUM_HEAD_IDS; h++) { ppo_log << ","; if (head_n[h]) ppo_log << head_ent[h] / head_n[h]; }
        ppo_log << std::endl;
        const std::string map_kind = K.ore_radius >= 0 ? "ore_r" + std::to_string(K.ore_radius) : std::string("map");
        for (auto& e : ended)
            games_log << it << "," << map_kind << "," << e.rounds << "," << e.winner << "," << e.cond
                      << "," << e.ti_col[0] << "," << e.ti_col[1] << "," << e.vo_col[0] << "," << e.vo_col[1]
                      << "," << e.harv[0] << "," << e.harv[1] << "," << e.units[0] << "," << e.units[1] << "\n";
        games_log.flush();
        grad_log << it << "," << gn_steps[0] << "," << gn_sum[0] / std::max(1, gn_steps[0]) << "," << gn_max[0] << "," << gn_clip[0]
                 << "," << gn_steps[1] << "," << gn_sum[1] / std::max(1, gn_steps[1]) << "," << gn_max[1] << "," << gn_clip[1] << std::endl;
        refusal_log << it;
        for (int k = 1; k < NUM_KINDS; k++) refusal_log << "," << tried[k] << "," << refused[k];
        refusal_log << std::endl;

        // ---- checkpoints
        if (it % env_int("SAVE_EVERY", 10) == 0 || it == iterations) torch::save(net, out);
        if (it % env_int("SNAP_EVERY", 400) == 0) torch::save(net, (run_dir / ("model_" + std::to_string(it) + ".pt")).string());

        // ---- eval vs frozen: sampled, then greedy (greedy decides promotion)
        if (K.eval_every > 0 && it % K.eval_every == 0) {
            net->eval();
            for (int mode = 0; mode < 2; mode++) {
                const bool greedy = mode == 1;
                const auto te = std::chrono::steady_clock::now();
                std::vector<Slot> ev_slots(K.eval_games);
                std::vector<ActionLine> log0;
                for (int i = 0; i < K.eval_games; i++) {
                    Slot& s = ev_slots[i];
                    s.rng.seed(0xE7A1ull * uint64_t(it) + uint64_t(i) * 31337ull + uint64_t(mode));
                    s.net_of[0] = i & 1; s.net_of[1] = 1 - (i & 1);   // live (net 0) is A on even games
                    s.greedy = greedy; s.record = false;
                    if (i == 0 && greedy) s.log = &log0;
                    start_game(s, S);
                }
                play(ev_slots, S, {&net, &frozen}, dev, /*stretch=*/0, /*restart=*/false, /*train=*/false, nullptr);
                long wins = 0, games = 0, ev_draws = 0; double rounds = 0;
                for (int i = 0; i < K.eval_games; i++) {
                    const fe::Game& g = ev_slots[i].m.game;
                    games++; rounds += g.round;
                    if (g.win_condition == "coinflip") ev_draws++;            // a draw, not a win
                    else if (g.winner == ((i & 1) ? 1 : 0)) wins++;
                }
                const float wr = games ? float(wins) / games : 0.f;
                bool promoted = false;
                std::cout << "  " << lfmt::label(greedy ? "eval greedy" : "eval sample") << "vs frozen " << lfmt::ratio(wins, games) << "   draws " << ev_draws
                          << "   rounds " << std::fixed << std::setprecision(0) << rounds / std::max(1L, games) << "   "
                          << std::setprecision(1) << std::chrono::duration<double>(std::chrono::steady_clock::now() - te).count()
                          << " s" << std::defaultfloat << std::endl;
                if (greedy) {
                    const fs::path dir = run_dir / "replays";
                    fs::create_directories(dir);
                    const std::string name = "eval_" + std::to_string(it);
                    const std::string mapf = (dir / (name + ".map26")).string(), gamef = (dir / (name + ".game")).string();
                    write_map(ev_slots[0].m.game, ev_slots[0].map_path, mapf);
                    write_game(gamef, mapf, ev_slots[0].seed, ev_slots[0].m.game, log0);
                    std::cout << "  " << lfmt::label("") << "replay log " << gamef << " (" << fs::path(ev_slots[0].map_path).filename().string()
                              << ", live is team A) -> rl/tools/make_replay.py" << std::endl;
                    if (wr >= PROMOTE) {
                        copy_model(net, frozen);
                        torch::save(frozen, (run_dir / "frozen.pt").string());
                        promoted = true;
                        std::cout << "  promoting frozen opponent" << std::endl;
                    }
                }
                winrate_log << it << "," << (greedy ? "greedy" : "sample") << "," << wins << "," << games << "," << wr
                            << "," << rounds / std::max(1L, games) << "," << int(promoted) << std::endl;
            }
        }
    }
    torch::save(net, out);
    std::cout << "TRAIN COMPLETE checkpoint=" << out << std::endl;
}
