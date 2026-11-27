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
//   env: GAMES (200) ROUNDS (150) N_MINI (15 minibatches a pass) VALUE_KEEP (0.1)
//        MICRO (512 rows per forward/backward) CRITIC_BATCH (1024 rows per critic inference)
//        RECORD_P (0.05: share of decisions kept for PPO; ~17 KB each) INIT (checkpoint) START_ITER (0) FROZEN (checkpoint)
//        THREADS (all) DEVICE (cuda if available | cpu) SNAP_EVERY (400)
//        SAVE_EVERY (10) ENTROPY_FILE (./entropy_file)
// Live knobs in the entropy file, see entropy_file.hpp.
#include <omp.h>
#include <torch/torch.h>
#ifdef FLORENT_CUDA_GRAPHS
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

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
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../sim/florent_game.hpp"
#include "../sim/florent_heads.hpp"
#include "entropy_file.hpp"
#include "game_log.hpp"
#include "../sim/map_gen.hpp"
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
    std::string kind;              // map: file stem, or gen_g<gap>[_pit]; + _ore<r> with ore_radius
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

// One recorded decision. Its input (spatial uint8: every actor channel is
// 0..255; scalars int16) lives in the device-side sample store, at `row`.
struct Sample {
    int policy, team;
    int seg, rnd;                  // where its round lives in its slot (for the advantage)
    long row = -1;                 // row in the sample store
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
    bool need = false; int ent = -1, policy = -1, net = 0;
    int8_t plan4[4] = {0, 0, 0, 0};                  // the waiting unit's conveyor plan (adjacent tiles N E S W)
    int own_launchers = 0;                           // own launchers it knows of (launcher cap)
    Targets tg;                                      // its target knowledge (turret / convert masks)
    uint8_t* in_sp = nullptr; int16_t* in_sc = nullptr;   // this slot's row of the (pinned) input stage
    const float* lg = nullptr;                            // its logits row after the forward
    bool recorded = false;                                // the last decision was recorded
    bool gate = false; Candidate gate_build;              // the waiting builder asks the sentinel gate first
    // play setup
    int net_of[2] = {0, 0}; bool greedy = false, record = true;
    std::vector<ActionLine>* log = nullptr;
    std::string map_path; uint64_t seed = 0;
    std::string kind;                                // for games.csv (see EndStat::kind)
    bool real_maps_only = false;                     // eval: the real maps only
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
    const bool gen = !s.real_maps_only && S.K->map_gen > 0.f && std::uniform_real_distribution<float>(0.f, 1.f)(s.rng) < S.K->map_gen;
    if (gen) {
        // a generated map (rl/sim/map_gen.hpp): core distance and pits under the entropy file's control
        MapGenParams P; P.core_gap = S.K->core_distance; P.core_gap_min = S.K->core_distance_min; P.pit_share = S.K->pit_share;
        MapGenInfo info;
        const std::string bytes = generate_map(P, s.rng, &info);
        s.map_path = "(generated)";
        s.seed = s.rng() & 0xFFFFFFFFull;
        if (!s.m.load_bytes(bytes, s.seed)) { std::cerr << "generated map failed to load: " << info.line << "\n"; std::exit(3); }
        s.kind = "gen_g" + std::to_string(info.gap) + (info.pit ? "_pit" : "");
    } else {
        s.map_path = S.maps[s.rng() % S.maps.size()];
        s.seed = s.rng() & 0xFFFFFFFFull;
        if (!s.m.load(s.map_path, s.seed)) { std::cerr << "bad map " << s.map_path << "\n"; std::exit(3); }
        s.kind = fs::path(s.map_path).stem().string();
    }
    const uint64_t ore_seed = s.rng();
    const int ore_r = pick_ore_radius(*S.K, ore_seed);
    if (ore_r >= 0) s.kind += "_ore" + std::to_string(ore_r);
    s.m.reset();
    s.m.game.round = s.start_round;
    s.start_round = 0;              // staggered once (the games of iteration 0); every restart starts at round 0
    s.m.game.max_turns = S.K->game_rounds;
    s.m.game.limit_ore(ore_r, ore_seed);
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
                s.ended.back().kind = s.kind;
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
            if (type == fe::SPIKE || type == fe::SENTINEL) {                   // deterministic, as deployed
                const bool first = !m.memories.count(a);   // its first turn (no memory yet): fires once
                m.observe(a);                          // its memory + the team report bus, as every unit
                Candidate c;
                if (spike_action(m.game, a, c, first) && m.act(a, c) == fe::OK && s.log) s.log->push_back({m.game.round, m.game.ents[a].id, c});
                s.pos++; continue;
            }
            const int p = policy_of_type(type);
            if (p < 0) { s.pos++; continue; }
            m.observe(a);                              // memory updates every turn, as deployed
            // encode first: the conveyor planner mask (path channel) is part of has_choice, as in the bot
            const long oob = m.encode_compact(a, s.in_sp, s.in_sc);
            plan4_from_obs(s.in_sp, s.plan4);
            s.own_launchers = m.memories.at(a).own_count(fe::LAUNCHER);
            targets_from_memory(m.memories.at(a), s.tg);
            if (!has_choice(m.game, a, s.plan4, s.own_launchers, &s.tg)) { s.pos++; continue; }   // only wait: no forward
            s.oob += oob;
            s.need = true; s.ent = a; s.policy = p; s.net = s.net_of[m.game.ents[a].team];
            // sentinel gate: a builder that can build an aimed sentinel asks the gate
            // network (P_SENTINEL, on this builder's input) first; "no": a second
            // forward, the builder network, on the same input (play_decision)
            s.gate = p == P_BUILDER && sentinel_gate(m.game, a, s.plan4, s.own_launchers, &s.tg, s.gate_build);
            if (s.gate) s.policy = P_SENTINEL;
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

// Pinned host memory when the device is CUDA (fast, asynchronous uploads).
static torch::TensorOptions host_opts(torch::ScalarType t, bool cuda) { return torch::TensorOptions().dtype(t).pinned_memory(cuda); }

// Compact uploads: the critic's one-hot board as uint8, its scalars as int32;
// the float conversion happens on the device (Teacher::value_split). The rows
// are packed in parallel into pinned memory and uploaded asynchronously (the
// caller keeps the host tensors alive until the copy has run).
static void critic_pack(const std::vector<const std::vector<uint8_t>*>& sps, const std::vector<const std::vector<int32_t>*>& scs,
                        bool cuda, torch::Tensor& hsp, torch::Tensor& hsc) {
    const long n = long(sps.size());
    hsp = torch::empty({n, fe::CRITIC_SPATIAL}, host_opts(torch::kUInt8, cuda));
    hsc = torch::empty({n, fe::CRITIC_SCALARS}, host_opts(torch::kInt32, cuda));
    uint8_t* a = hsp.data_ptr<uint8_t>(); int32_t* b = hsc.data_ptr<int32_t>();
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        std::memcpy(a + i * fe::CRITIC_SPATIAL, sps[i]->data(), fe::CRITIC_SPATIAL);
        std::memcpy(b + i * fe::CRITIC_SCALARS, scs[i]->data(), sizeof(int32_t) * fe::CRITIC_SCALARS);
    }
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

// Play the forward's result for the waiting unit. A recorded decision's input
// is copied to the sample store afterwards (store_recorded).
static void play_decision(Slot& s, const Setup& S, bool train) {
    Match& m = s.m;
    const int a = s.ent;
    const bool gate = s.gate;
    auto chooser = [&](int, const float* l, const std::vector<uint8_t>& mk, float& lp) {
        return s.greedy ? argmax_choice(l, mk, lp) : sample_masked(l, mk, s.rng, lp);
    };
    Decision d = gate ? decide_gate(s.lg + head_offset(P_SENTINEL, HS_MODE), chooser, s.gate_build)
                      : decide(m.game, a, s.lg, chooser, s.plan4, s.own_launchers, &s.tg);
    for (auto& c : d.actions) {
        const int k = int(c.kind);
        s.tried[k]++;
        if (m.act(a, c) != fe::OK) { s.refused[k]++; continue; }
        if (s.log) s.log->push_back({m.game.round, m.game.ents[a].id, c});
    }
    s.decisions++;
    s.recorded = false;
    // only heads with a real choice (2+ legal options) are recorded: a single
    // legal option has log-prob 0 and entropy 0 whatever the weights, so it
    // gives no gradient and would only dilute the loss and the logs. A
    // decision with no such head is not recorded.
    int real = 0;
    for (auto& hr : d.records) { int n = 0; for (uint8_t v : hr.mask) n += v != 0; real += n >= 2; }
    if (train && s.record && real > 0 && std::uniform_real_distribution<float>(0.f, 1.f)(s.rng) < S.record_p) {
        Sample smp;
        smp.policy = s.policy; smp.team = m.game.ents[a].team;
        smp.seg = int(s.segs.size()) - 1; smp.rnd = int(s.segs.back().rounds.size()) - 1;
        for (auto& hr : d.records) {
            int n = 0; for (uint8_t v : hr.mask) n += v != 0;
            if (n >= 2) smp.recs.push_back({hr.head, hr.choice, std::move(hr.mask), hr.logp});
        }
        s.samples.push_back(std::move(smp));
        s.recorded = true;
    }
    if (gate && d.actions.empty()) {                 // gate said no: the builder network decides, same input
        s.gate = false; s.policy = P_BUILDER;
        return;                                      // still waiting (need stays true)
    }
    s.gate = false;
    s.need = false; s.pos++;
}

// ===========================================================================
// ROLLOUT I/O
// ===========================================================================
// Every step of a rollout: each waiting game writes its unit's compact input
// straight into its row of a pinned stage (in the parallel advance), the
// stage goes up in ONE copy, each (net, policy) group gathers its rows on the
// device, runs its forward and scatters its logits into one device logits
// table, which comes back in ONE copy. Recorded decisions' inputs are then
// copied device-to-device into the sample store, which PPO trains from: the
// rollout's inputs never go back through host memory.
static int max_logits() { int m = 0; for (int p = 0; p < NUM_POLICIES; p++) m = std::max(m, policy_logit_count(p)); return m; }

struct RollIO {
    long cap = 0;
    torch::Tensor h_sp, h_sc, d_sp, d_sc;          // [cap, ACTOR_SPATIAL] uint8, [cap, ACTOR_SCALARS] int16
    torch::Tensor h_lg, d_lg;                      // [cap, max_logits()] float
    torch::Tensor h_idx;                           // pinned int64 scratch for index uploads
    long idx_cap = 0;
    void ensure(long G, torch::Device dev) {
        const bool cuda = dev.is_cuda();
        if (G > cap) {
            cap = G;
            h_sp = torch::zeros({G, fe::ACTOR_SPATIAL}, host_opts(torch::kUInt8, cuda));
            h_sc = torch::zeros({G, fe::ACTOR_SCALARS}, host_opts(torch::kInt16, cuda));
            h_lg = torch::zeros({G, max_logits()}, host_opts(torch::kFloat32, cuda));
            if (cuda) {
                auto o = torch::TensorOptions().device(dev);
                d_sp = torch::zeros({G, fe::ACTOR_SPATIAL}, o.dtype(torch::kUInt8));
                d_sc = torch::zeros({G, fe::ACTOR_SCALARS}, o.dtype(torch::kInt16));
                d_lg = torch::zeros({G, max_logits()}, o.dtype(torch::kFloat32));
            } else { d_sp = h_sp; d_sc = h_sc; d_lg = h_lg; }
        }
        // one region per (net, policy) group + one for the recorded rows (x2: slots and store rows)
        const long need = (2 * NUM_POLICIES + 2) * G;
        if (need > idx_cap) { idx_cap = need; h_idx = torch::zeros({need}, host_opts(torch::kLong, cuda)); }
    }
    // upload v (int64) through the pinned scratch at offset `off` (asynchronous on CUDA)
    torch::Tensor upload_idx(const std::vector<long>& v, long off, torch::Device dev) {
        long* p = h_idx.data_ptr<long>() + off;
        std::memcpy(p, v.data(), sizeof(long) * v.size());
        auto t = h_idx.narrow(0, off, long(v.size()));
        return dev.is_cuda() ? t.to(dev, /*non_blocking=*/true) : t.clone();
    }
};
static RollIO g_io;
#ifdef FLORENT_CUDA_GRAPHS
static std::vector<c10::cuda::CUDAStream> g_group_streams;   // one per (net, policy) group of a step
#endif

// Decision inputs kept for PPO, on the device, reused across iterations.
// Memory: sized before each rollout (prepare: the previous rollout's count +
// 10%, or an estimate), with the old buffer released first, so a full store is
// never copied into a bigger one in the middle of a rollout. If it must still
// grow and the device is out of memory (or past STORE_GB), the store stays
// full: further decisions of that rollout are played but not recorded.
struct SampleStore {
    torch::Tensor sp, sc;
    long n = 0, cap = 0, limit = -1;   // limit: max rows (STORE_GB), -1 none
    bool full = false;
    long skipped = 0;
    static constexpr long ROW = fe::ACTOR_SPATIAL + 2 * fe::ACTOR_SCALARS;
    bool alloc(long nc, torch::Device dev, bool keep) {
        if (limit >= 0) nc = std::min(nc, limit);
        if (nc <= cap) return false;
        auto o = torch::TensorOptions().device(dev);
        try {
            if (!keep) { sp = torch::Tensor(); sc = torch::Tensor(); cap = 0;
#ifdef FLORENT_CUDA_GRAPHS
                if (dev.is_cuda()) c10::cuda::CUDACachingAllocator::emptyCache();
#endif
            }
            auto nsp = torch::empty({nc, fe::ACTOR_SPATIAL}, o.dtype(torch::kUInt8));
            auto nsc = torch::empty({nc, fe::ACTOR_SCALARS}, o.dtype(torch::kInt16));
            if (keep && n > 0) { nsp.narrow(0, 0, n).copy_(sp.narrow(0, 0, n)); nsc.narrow(0, 0, n).copy_(sc.narrow(0, 0, n)); }
            sp = nsp; sc = nsc; cap = nc;
            return true;
        } catch (const c10::Error&) {
            return false;                  // out of memory: keep what we have
        }
    }
    // before a rollout: empty, sized for `expect` rows
    void prepare(long expect, torch::Device dev) {
        n = 0; full = false; skipped = 0;
        if (expect > cap) alloc(expect, dev, /*keep=*/false);
    }
    // room for `need` rows? (grows if it can; else the store is full)
    bool reserve(long need, torch::Device dev) {
        if (need <= cap) return true;
        if (full) return false;
        if (!alloc(std::max(need, cap + cap / 4 + 4096), dev, /*keep=*/true)) { full = true; return need <= cap; }
        return true;
    }
};
static SampleStore g_store;
static long g_store_last = 0;          // rows the previous rollout wanted (recorded + skipped)

// ===========================================================================
// ROLLOUT FORWARD, with CUDA graphs
// ===========================================================================
// A rollout runs thousands of small forwards (one per policy per step), so on
// a GPU the time goes to launching kernels, not to computing. A CUDA graph
// records one forward's kernels once and replays them with a single launch.
// One graph per (net, policy, batch bucket): the batch is padded up to a
// power of two (>= 8); the padded rows' outputs are ignored. The weights are
// updated in place (optimizer, copy_model), so a graph stays valid all run.
// Each new graph is checked once against the plain forward; a mismatch turns
// graphs off (CUDA_GRAPHS=0 turns them off from the start).
struct FwdGraphs {
    bool enabled = false;
#ifdef FLORENT_CUDA_GRAPHS
    struct Entry { std::unique_ptr<at::cuda::CUDAGraph> g; torch::Tensor sp, sc, out; };
    std::map<std::tuple<const void*, int, long>, Entry> m;
#endif
};
static FwdGraphs g_graphs;

// Forward of policy p on rows `idx` (device int64, n rows) of the device
// inputs sp / sc. Returns the device logits [n, L] (valid until the next
// forward of the same graph).
static torch::Tensor policy_forward(Teacher& net, int p, const torch::Tensor& sp, const torch::Tensor& sc,
                                    const torch::Tensor& idx, long n, torch::Device dev) {
#ifdef FLORENT_CUDA_GRAPHS
    if (g_graphs.enabled && dev.is_cuda()) {
        long B = 8; while (B < n) B *= 2;
        const auto key = std::make_tuple((const void*)net.get(), p, B);
        auto it = g_graphs.m.find(key);
        if (it == g_graphs.m.end()) {
            FwdGraphs::Entry e;
            auto opt = torch::TensorOptions().device(dev);
            e.sp = torch::zeros({B, fe::ACTOR_SPATIAL}, opt.dtype(torch::kUInt8));
            e.sc = torch::zeros({B, fe::ACTOR_SCALARS}, opt.dtype(torch::kInt16));
            {
                auto st = c10::cuda::getStreamFromPool();
                c10::cuda::getCurrentCUDAStream().synchronize();
                c10::cuda::CUDAStreamGuard guard(st);
                for (int w = 0; w < 3; w++) net->logits_split(p, e.sp, e.sc);     // warm-up before capture
                st.synchronize();
                e.g = std::make_unique<at::cuda::CUDAGraph>();
                e.g->capture_begin();
                e.out = net->logits_split(p, e.sp, e.sc);
                e.g->capture_end();
                st.synchronize();
            }
            // check the graph against the plain forward on this batch
            auto xs = sp.index_select(0, idx), xc = sc.index_select(0, idx);
            e.sp.narrow(0, 0, n).copy_(xs); e.sc.narrow(0, 0, n).copy_(xc);
            e.g->replay();
            auto ref = net->logits_split(p, xs, xc);
            const double diff = (e.out.narrow(0, 0, n) - ref).abs().max().item<double>();
            std::cout << "  cuda graph  captured: policy " << p << " batch " << B << ", check max diff " << diff << std::endl;
            if (!(diff < 1e-3)) {
                std::cout << "  cuda graph  mismatch: graphs off for the rest of the run" << std::endl;
                g_graphs.enabled = false;
                return ref;
            }
            it = g_graphs.m.emplace(key, std::move(e)).first;
            return it->second.out.narrow(0, 0, n);
        }
        FwdGraphs::Entry& e = it->second;
        net->inference_flat(p);                          // weights changed since the capture: refresh in place
        auto osp = e.sp.narrow(0, 0, n), osc = e.sc.narrow(0, 0, n);
        torch::index_select_out(osp, sp, 0, idx);
        torch::index_select_out(osc, sc, 0, idx);
        e.g->replay();
        return e.out.narrow(0, 0, n);
    }
#endif
    return net->logits_split(p, sp.index_select(0, idx), sc.index_select(0, idx));
}

struct PlayStats { long steps = 0; double t_encode = 0, t_forward = 0, t_act = 0, t_critic = 0; long critic_rows = 0; };

// Kept value rows (for the value loss): the row of the device critic store
// and the return.
struct ValueRow { long row = -1; float ret = 0; int slot, seg, rnd; };
struct CriticStore {
    std::vector<torch::Tensor> sp_parts, sc_parts;   // kept rows, flush by flush
    torch::Tensor sp, sc;                            // concatenated at the end of the rollout
    long n = 0;
};

// Critic inference runs on its own CUDA stream, overlapped with the rollout:
// queued rows are evaluated in batches as soon as CRITIC_FLUSH of them are
// waiting (the weights do not change during a rollout); the values are read
// back once at the end.
struct CriticPending {
    torch::Tensor hsp, hsc, v, keep_h;               // pinned inputs (alive until uploaded), device values
    std::vector<std::pair<int, CriticRow>> rows;     // (slot, row metadata; inputs already freed)
};
struct CriticRun {
    std::vector<CriticPending> pend;
#ifdef FLORENT_CUDA_GRAPHS
    std::optional<c10::cuda::CUDAStream> stream;
#endif
};
static int CRITIC_FLUSH = 2000;

// Launch the critic on every queued row (asynchronously on CUDA); rows marked
// to keep are also copied into the critic store.
static void launch_critic(std::vector<Slot>& slots, Teacher& net, torch::Device dev, CriticStore& cs, CriticRun& run, PlayStats& st) {
    auto c0 = std::chrono::steady_clock::now();
    std::vector<std::pair<int, CriticRow*>> all;
    for (int i = 0; i < int(slots.size()); i++) for (auto& cr : slots[i].crit) all.push_back({i, &cr});
    const bool cuda = dev.is_cuda();
#ifdef FLORENT_CUDA_GRAPHS
    std::optional<c10::cuda::CUDAStreamGuard> sg;
    if (cuda) {
        if (!run.stream) run.stream = c10::cuda::getStreamFromPool();
        at::cuda::CUDAEvent ready;                      // the critic stream starts after the main stream's work
        ready.record(c10::cuda::getCurrentCUDAStream());
        ready.block(*run.stream);
        sg.emplace(*run.stream);
    }
#endif
    for (size_t lo = 0; lo < all.size(); lo += CRITIC_BATCH) {
        const size_t hi = std::min(all.size(), lo + CRITIC_BATCH);
        std::vector<const std::vector<uint8_t>*> sps; std::vector<const std::vector<int32_t>*> scs;
        for (size_t k = lo; k < hi; k++) { sps.push_back(&all[k].second->sp); scs.push_back(&all[k].second->sc); }
        CriticPending pd;
        critic_pack(sps, scs, cuda, pd.hsp, pd.hsc);
        auto tsp = cuda ? pd.hsp.to(dev, true) : pd.hsp, tsc = cuda ? pd.hsc.to(dev, true) : pd.hsc;
        {
            c10::InferenceMode im;
            pd.v = net->value_split(tsp, tsc);
        }
        std::vector<long> keep;
        for (size_t k = lo; k < hi; k++) {
            CriticRow& cr = *all[k].second;
            Slot& s = slots[all[k].first];
            if (!cr.boot && s.segs[cr.seg].rounds[cr.rnd].keep == -2) {
                s.segs[cr.seg].rounds[cr.rnd].keep = int(cs.n + long(keep.size()));
                keep.push_back(long(k - lo));
            }
        }
        if (!keep.empty()) {
            c10::InferenceMode off(false);           // normal tensors: PPO trains on them later
            torch::NoGradGuard ng;
            // pinned + asynchronous: a pageable upload would wait for this batch's forward
            auto kh = torch::empty({long(keep.size())}, host_opts(torch::kLong, cuda));
            std::memcpy(kh.data_ptr<long>(), keep.data(), sizeof(long) * keep.size());
            auto ki = cuda ? kh.to(dev, /*non_blocking=*/true) : kh;
            pd.keep_h = kh;
            cs.sp_parts.push_back(tsp.index_select(0, ki));
            cs.sc_parts.push_back(tsc.index_select(0, ki));
            cs.n += long(keep.size());
        }
        for (size_t k = lo; k < hi; k++) {
            CriticRow& cr = *all[k].second;
            std::vector<uint8_t>().swap(cr.sp); std::vector<int32_t>().swap(cr.sc);
            pd.rows.push_back({all[k].first, std::move(cr)});
        }
        st.critic_rows += long(hi - lo);
        run.pend.push_back(std::move(pd));
    }
    for (auto& s : slots) s.crit.clear();
    st.t_critic += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
}

// Wait for every launched critic batch and write its values back.
static void finish_critic(std::vector<Slot>& slots, std::vector<ValueRow>& kept, CriticStore& cs, CriticRun& run, PlayStats& st) {
    auto c0 = std::chrono::steady_clock::now();
#ifdef FLORENT_CUDA_GRAPHS
    if (run.stream) run.stream->synchronize();               // every batch's values are computed
#endif
    for (auto& pd : run.pend) {
        auto v = pd.v.to(torch::kCPU).contiguous();
        const float* vp = v.data_ptr<float>();
        for (size_t k = 0; k < pd.rows.size(); k++) {
            const CriticRow& cr = pd.rows[k].second;
            Slot& s = slots[pd.rows[k].first];
            if (cr.boot) { s.segs[cr.seg].boot = vp[k]; continue; }
            RoundE& re = s.segs[cr.seg].rounds[cr.rnd];
            re.value = vp[k];
            if (re.keep >= 0) kept.push_back({long(re.keep), 0.f, pd.rows[k].first, cr.seg, cr.rnd});
        }
    }
    run.pend.clear();
    if (!cs.sp_parts.empty()) {
        c10::InferenceMode off(false);
        torch::NoGradGuard ng;
        cs.sp = cs.sp_parts.size() == 1 ? cs.sp_parts[0] : torch::cat(cs.sp_parts, 0);
        cs.sc = cs.sc_parts.size() == 1 ? cs.sc_parts[0] : torch::cat(cs.sc_parts, 0);
        cs.sp_parts.clear(); cs.sc_parts.clear();
    }
    // kept rows in critic-store order
    std::sort(kept.begin(), kept.end(), [](const ValueRow& a, const ValueRow& b) { return a.row < b.row; });
    st.t_critic += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
}

// Run all slots until each is paused or done. nets[k] plays net index k.
static PlayStats play(std::vector<Slot>& slots, const Setup& S, const std::vector<Teacher*>& nets, torch::Device dev,
                      int stretch, bool restart, bool train, std::vector<ValueRow>* kept, CriticStore* cstore) {
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    PlayStats st;
    const int G = int(slots.size());
    const bool cuda = dev.is_cuda();
    g_io.ensure(G, dev);
    const long LW = g_io.h_lg.size(1);
    for (int i = 0; i < G; i++) {
        Slot& s = slots[i];
        s.paused = false; s.played = 0;
        s.in_sp = g_io.h_sp.data_ptr<uint8_t>() + long(i) * fe::ACTOR_SPATIAL;
        s.in_sc = g_io.h_sc.data_ptr<int16_t>() + long(i) * fe::ACTOR_SCALARS;
        s.lg = g_io.h_lg.data_ptr<float>() + long(i) * LW;
    }
    const bool store = train && kept;
    if (store) {
        // the previous rollout's count + 10%, or ~16 recorded decisions per game-round
        const long expect = g_store_last > 0 ? g_store_last + g_store_last / 10
                                             : long(double(G) * std::max(1, stretch) * 16 * S.record_p);
        g_store.prepare(expect, dev);
    }
    CriticRun crun;
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
        // one upload of every waiting input, then one forward per (net, policy)
        int hi_row = -1;
        for (int i = 0; i < G; i++) if (slots[i].need) hi_row = i;
        if (hi_row < 0) break;
        if (cuda) {
            g_io.d_sp.narrow(0, 0, hi_row + 1).copy_(g_io.h_sp.narrow(0, 0, hi_row + 1), true);
            g_io.d_sc.narrow(0, 0, hi_row + 1).copy_(g_io.h_sc.narrow(0, 0, hi_row + 1), true);
        }
        long region = 0;
#ifdef FLORENT_CUDA_GRAPHS
        // each (net, policy) group on its own stream: the groups' forwards overlap
        std::optional<at::cuda::CUDAEvent> up;
        if (cuda) { up.emplace(); up->record(c10::cuda::getCurrentCUDAStream()); }
        std::vector<at::cuda::CUDAEvent> done;
        done.reserve(nets.size() * NUM_POLICIES);
#endif
        for (int k = 0; k < int(nets.size()); k++)
            for (int p = 0; p < NUM_POLICIES; p++) {
                std::vector<long> idx;
                for (int i = 0; i < G; i++) if (slots[i].need && slots[i].net == k && slots[i].policy == p) idx.push_back(i);
                if (idx.empty()) continue;
                const long n = long(idx.size());
#ifdef FLORENT_CUDA_GRAPHS
                std::optional<c10::cuda::CUDAStreamGuard> sg;
                if (cuda) {
                    if (g_group_streams.size() < size_t(region + 1)) g_group_streams.push_back(c10::cuda::getStreamFromPool(true));
                    sg.emplace(g_group_streams[region]);
                    up->block(g_group_streams[region]);
                }
#endif
                auto d_idx = g_io.upload_idx(idx, region * G, dev);
                region++;
                auto l = policy_forward(*nets[k], p, g_io.d_sp, g_io.d_sc, d_idx, n, dev);
                g_io.d_lg.narrow(1, 0, l.size(1)).index_copy_(0, d_idx, l);
#ifdef FLORENT_CUDA_GRAPHS
                if (cuda) { done.emplace_back(); done.back().record(c10::cuda::getCurrentCUDAStream()); }
#endif
                st.steps++;
            }
#ifdef FLORENT_CUDA_GRAPHS
        for (auto& e : done) e.block(c10::cuda::getCurrentCUDAStream());
#endif
        if (cuda) g_io.h_lg.copy_(g_io.d_lg);                // one read-back (waits for the forwards)
        auto t2 = clk::now();
        st.t_forward += secs(t1, t2);
        #pragma omp parallel for schedule(static, 1)
        for (int i = 0; i < G; i++) if (slots[i].need) play_decision(slots[i], S, train);
        if (store) {                                          // recorded inputs -> sample store, on the device
            std::vector<long> from, to;
            for (int i = 0; i < G; i++)
                if (slots[i].recorded) { slots[i].recorded = false; slots[i].samples.back().row = g_store.n + long(from.size()); from.push_back(i); }
            if (!from.empty() && !g_store.reserve(g_store.n + long(from.size()), dev)) {
                // store full: these decisions stay played but are not recorded
                for (int i : from) slots[i].samples.pop_back();
                g_store.skipped += long(from.size());
                from.clear();
            }
            if (!from.empty()) {
                for (size_t j = 0; j < from.size(); j++) to.push_back(g_store.n + long(j));
                auto df = g_io.upload_idx(from, 2 * NUM_POLICIES * G, dev), dt = g_io.upload_idx(to, (2 * NUM_POLICIES + 1) * G, dev);
                g_store.sp.index_copy_(0, dt, g_io.d_sp.index_select(0, df));
                g_store.sc.index_copy_(0, dt, g_io.d_sc.index_select(0, df));
                g_store.n += long(from.size());
            }
        }
        st.t_act += secs(t2, clk::now());
        if (train && kept) {
            size_t q = 0; for (auto& s : slots) q += s.crit.size();
            if (q >= size_t(CRITIC_FLUSH)) launch_critic(slots, *nets[0], dev, *cstore, crun, st);
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
        launch_critic(slots, *nets[0], dev, *cstore, crun, st);
        finish_critic(slots, *kept, *cstore, crun, st);
    }
    if (store) g_store_last = g_store.n + g_store.skipped;
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
    const int G = env_int("GAMES", 200), R = env_int("ROUNDS", 150);
    const int N_MINI = env_int("N_MINI", 10);      // minibatches per pass (reference pipeline: 15)
    // Full-batch learner: do not split PPO minibatches into MICRO chunks.
    MICRO = env_int("MICRO", 65536);  // default: no split (the fused trunk keeps no activations, so memory allows it)
    // STORE_GB: cap on the device sample store (GB); default: as much as fits
    if (const char* sg = std::getenv("STORE_GB")) g_store.limit = long(std::atof(sg) * 1e9 / double(SampleStore::ROW));
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
#ifdef FLORENT_CUDA_GRAPHS
    g_graphs.enabled = dev.is_cuda() && env_int("CUDA_GRAPHS", 1) != 0;
#endif

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
                  << ", device " << (dev.is_cuda() ? "cuda" : "cpu") << (g_graphs.enabled ? " (rollout cuda graphs)" : "") << ", " << T << " threads\n"
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
            if (K.full_games) {                     // a fresh game from round 0, played to its end
                s.start_round = 0;
                start_game(s, S);
            } else if (!s.started) {
                const int game_len = std::max(1, S.K->game_rounds);
                const int step = env_int("START_TURN_STEP", 10);
                s.start_round = ((i % game_len) * (step % game_len)) % game_len;
                start_game(s, S);
            }
            else s.segs.push_back(Segment());      // the running game continues in a new segment
        }
        net->eval();
        std::vector<ValueRow> kept;
        CriticStore cstore;
        // full games: no stretch limit, no restart (every segment ends terminal);
        // otherwise R more rounds of every running game, cut and bootstrapped
        const PlayStats ps = K.full_games
            ? play(slots, S, {&net}, dev, /*stretch=*/0, /*restart=*/false, /*train=*/true, &kept, &cstore)
            : play(slots, S, {&net}, dev, R, /*restart=*/true, /*train=*/true, &kept, &cstore);
        if (K.full_games) for (auto& s : slots) s.started = false;   // switching back: start staggered games
        if (g_store.skipped > 0) {
            // the store filled up: record a smaller share from the next rollout on, so
            // what is recorded is spread over the whole rollout (not only its start)
            const double fit = double(g_store.cap) / double(g_store.n + g_store.skipped);
            S.record_p = float(std::max(0.01, S.record_p * fit * 0.95));
            std::cout << "  " << lfmt::label("store") << "full at " << g_store.cap << " rows: RECORD_P -> " << S.record_p << std::endl;
        }

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
        // ---- sentinel gate (P_SENTINEL): its decisions are rare, so its samples are
        // carried over iterations (inputs copied to the host) until GATE_MB (800) of
        // them, then the gate gets its OWN PPO update (after the main one) in
        // minibatches of GATE_MB. The gate network changes only in that update, so
        // carried samples stay exactly on-policy however long they wait.
        std::vector<Sample> gate_batch; torch::Tensor gate_sp, gate_sc;
        static const long GATE_MB = env_int("GATE_MB", 800);
        {
            static std::vector<Sample> carry; static torch::Tensor csp, csc; static int age = 0;
            std::vector<long> rows; std::vector<Sample> rest;
            for (auto& x : samples) {
                if (x.policy == P_SENTINEL && x.row >= 0) { rows.push_back(x.row); carry.push_back(std::move(x)); }
                else rest.push_back(std::move(x));
            }
            samples.swap(rest);
            if (!rows.empty()) {
                auto idx = torch::tensor(rows, torch::kLong).to(g_store.sp.device());
                auto nsp = g_store.sp.index_select(0, idx).to(torch::kCPU), nsc = g_store.sc.index_select(0, idx).to(torch::kCPU);
                csp = csp.defined() ? torch::cat({csp, nsp}) : nsp; csc = csc.defined() ? torch::cat({csc, nsc}) : nsc;
            }
            age++;
            if (long(carry.size()) >= GATE_MB) {
                gate_batch.swap(carry); gate_sp = csp; gate_sc = csc;
                std::cout << "  gate: training " << gate_batch.size() << " sentinel-gate samples (collected over " << age << " iterations)" << std::endl;
                carry.clear(); csp = torch::Tensor(); csc = torch::Tensor(); age = 0;
            } else if (!carry.empty()) {
                std::cout << "  gate: " << carry.size() << "/" << GATE_MB << " sentinel-gate samples collected (" << age << " iterations)" << std::endl;
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

        // ---- the iteration's samples are already on the device (g_store);
        //  ROW[sample] = its store row. Per head h: its records as device
        //  tensors (choice, mask, old logp, advantage), and rec_of[h][sample] =
        //  the sample's record in head h, -1 none.
        const long NS = long(samples.size());
        const bool cuda = dev.is_cuda();
        using uclk = std::chrono::steady_clock;
        // timing waits for this thread's stream only (the value steps run on their own)
        auto sync_now = [&] {
#ifdef FLORENT_CUDA_GRAPHS
            if (cuda) c10::cuda::getCurrentCUDAStream().synchronize();
#else
            if (cuda) torch::cuda::synchronize();
#endif
            return uclk::now();
        };
        double t_pack = 0, t_pol = 0, t_val = 0;
        const auto tp0 = uclk::now();
        torch::Tensor ROW;
        {
            std::vector<long> r(NS);
            for (long i = 0; i < NS; i++) r[i] = samples[i].row;
            ROW = torch::tensor(r, torch::kLong).to(dev);
        }
        const torch::Tensor& SP = g_store.sp;
        const torch::Tensor& SC = g_store.sc;
        std::vector<std::vector<int>> rec_of(NUM_HEAD_IDS, std::vector<int>(NS, -1));
        torch::Tensor HCH[NUM_HEAD_IDS], HMK[NUM_HEAD_IDS], HOLP[NUM_HEAD_IDS], HADV[NUM_HEAD_IDS];
        {
            std::vector<long> ch[NUM_HEAD_IDS]; std::vector<uint8_t> mk[NUM_HEAD_IDS];
            std::vector<float> olp[NUM_HEAD_IDS], adv[NUM_HEAD_IDS];
            for (long i = 0; i < NS; i++)
                for (auto& rec : samples[i].recs) {
                    const int h = rec.head;
                    rec_of[h][i] = int(ch[h].size());
                    ch[h].push_back(rec.choice);
                    mk[h].insert(mk[h].end(), rec.mask.begin(), rec.mask.end());
                    olp[h].push_back(rec.old_logp); adv[h].push_back(samples[i].adv);
                }
            for (int h = 0; h < NUM_HEAD_IDS; h++) {
                const long k = long(ch[h].size());
                if (!k) continue;
                HCH[h] = torch::tensor(ch[h], torch::kLong).to(dev);
                HMK[h] = torch::from_blob(mk[h].data(), {k, HEAD_SIZE[h]}, torch::kUInt8).to(dev).to(torch::kBool);
                HOLP[h] = torch::tensor(olp[h]).to(dev);
                HADV[h] = torch::tensor(adv[h]).to(dev);
            }
        }
        // value targets, in critic-store order
        torch::Tensor VRET;
        {
            std::vector<float> r(kept.size());
            for (size_t i = 0; i < kept.size(); i++) r[i] = kept[i].ret;
            VRET = torch::tensor(r).to(dev);
        }

        t_pack = std::chrono::duration<double>(sync_now() - tp0).count();
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
        // One chunk (rows of one policy) with every index it needs, built once
        // per minibatch on the host and uploaded in one copy: its store rows,
        // and per head its records' (row in the chunk, record index).
        struct ChunkHead { int head; long k; torch::Tensor rr, ri; };
        struct Chunk { int p; long nr; torch::Tensor rows; std::vector<ChunkHead> heads; };
        auto make_chunks = [&](const std::vector<int>* by_pol) {
            std::vector<Chunk> cs;
            std::vector<long> buf;
            struct Off { long rows; std::vector<std::tuple<int, long, long, long>> h; };   // head, k, rr off, ri off
            std::vector<Off> offs;
            for (int p = 0; p < NUM_POLICIES; p++)
                for (size_t c = 0; c < by_pol[p].size(); c += MICRO) {
                    const int* rows = by_pol[p].data() + c;
                    const long nr = long(std::min(by_pol[p].size() - c, size_t(MICRO)));
                    Off o; o.rows = long(buf.size());
                    for (long b = 0; b < nr; b++) buf.push_back(samples[rows[b]].row);
                    for (int head : policy_heads(p)) {
                        const std::vector<int>& ro = rec_of[head];
                        std::vector<long> rr, ri;
                        for (long b = 0; b < nr; b++) { const int x = ro[rows[b]]; if (x >= 0) { rr.push_back(b); ri.push_back(x); } }
                        if (rr.empty()) continue;
                        const long k = long(rr.size()), o1 = long(buf.size());
                        buf.insert(buf.end(), rr.begin(), rr.end());
                        buf.insert(buf.end(), ri.begin(), ri.end());
                        o.h.push_back({head, k, o1, o1 + k});
                    }
                    cs.push_back({p, nr, {}, {}});
                    offs.push_back(std::move(o));
                }
            auto host = torch::empty({long(buf.size())}, host_opts(torch::kLong, cuda));
            std::memcpy(host.data_ptr<long>(), buf.data(), sizeof(long) * buf.size());
            auto d = cuda ? host.to(dev, /*non_blocking=*/true) : host;
            for (size_t c = 0; c < cs.size(); c++) {
                cs[c].rows = d.narrow(0, offs[c].rows, cs[c].nr);
                for (auto& [head, k, o1, o2] : offs[c].h) cs[c].heads.push_back({head, k, d.narrow(0, o1, k), d.narrow(0, o2, k)});
            }
            return cs;
        };
        // One chunk. Always adds its sums to `acc`. With `grad`, also returns
        // the chunk's loss term:
        //   -surr / n_total  -  (1/present) * sum_h coef_h * Hbar_h / Hbar_h.detach()
        // Hbar_h is head h's mean entropy over the minibatch. If the policy is in
        // one chunk (h_mean == nullptr) it is computed here, exactly; otherwise
        // h_mean comes from a no-grad pass over the whole minibatch, and the
        // chunk's share is H_chunk / (cnt_h * h_mean_h).
        auto chunk = [&](const Chunk& C, bool grad, Acc& acc, long n_total,
                         const long* cnt_total, const double* h_mean, int present) -> torch::Tensor {
            const int p = C.p;
            auto logits = net->logits_split(p, SP.index_select(0, C.rows), SC.index_select(0, C.rows));
            torch::Tensor loss = torch::zeros({}, dev);
            for (const ChunkHead& ch : C.heads) {
                const int head = ch.head;
                const long k = ch.k;
                auto hl = logits.index_select(0, ch.rr).narrow(1, head_offset(p, head), HEAD_SIZE[head]);
                auto mask = HMK[head].index_select(0, ch.ri);
                auto lsm = torch::log_softmax(hl.masked_fill(~mask, -1e9f), 1);
                auto new_lp = lsm.gather(1, HCH[head].index_select(0, ch.ri).view({k, 1})).squeeze(1);
                auto log_ratio = new_lp - HOLP[head].index_select(0, ch.ri);
                auto A = HADV[head].index_select(0, ch.ri);
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
        auto value_step = [&](const std::vector<int>& vo, size_t j) {
            const size_t v0 = j * VB;
            if (v0 >= vo.size()) return;
            const size_t v1 = std::min(vo.size(), v0 + VB);
            const double n_all = double(v1 - v0);
            opt_val.zero_grad();
            torch::Tensor vl_tot = torch::zeros({}, dev);
            for (size_t c0 = v0; c0 < v1; c0 += MICRO) {
                const size_t c1 = std::min(v1, c0 + MICRO);
                std::vector<long> rows(vo.begin() + c0, vo.begin() + c1);     // kept index == critic-store row
                auto ri = torch::tensor(rows, torch::kLong).to(dev);
                auto se = (net->value_split(cstore.sp.index_select(0, ri), cstore.sc.index_select(0, ri)) - VRET.index_select(0, ri)).pow(2).sum();
                (VALUE_COEF * se / n_all).backward();                          // chunk share of the mean
                vl_tot += se.detach();
            }
            clip_norm(val_params, CLIP_VAL, 1);
            opt_val.step();
            vl_sum += vl_tot.item<double>() / n_all; v_steps++;
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
                    for (auto& rec : samples[order[k]].recs) { cnt_total[rec.head]++; n_total++; }   // recs kept (small): counts only
                for (int h = 0; h < NUM_HEAD_IDS; h++) present += cnt_total[h] > 0;
                if (!n_total) continue;
                bool split = false;
                for (int p = 0; p < NUM_POLICIES; p++) split |= by_pol[p].size() > size_t(MICRO);
                const auto tq0 = sync_now();
                const std::vector<Chunk> chunks = make_chunks(by_pol);
                // no-grad pass only when needed: the KL check (pass 2+) or a policy split into chunks
                double h_mean[NUM_HEAD_IDS] = {};
                if (ep > 0 || split) {
                    torch::NoGradGuard ng;
                    Acc pre(dev);
                    for (const Chunk& C : chunks) chunk(C, false, pre, 0, nullptr, nullptr, 0);
                    auto r = torch::cat({pre.kl.view({1}), pre.H}).to(torch::kCPU);
                    const float* rp = r.data_ptr<float>();
                    const double kl = rp[0] / double(pre.n);
                    if (ep > 0 && kl > KL_STOP) { kl_stop = true; stop_ep = ep; stop_kl = kl; break; }
                    for (int h = 0; h < NUM_HEAD_IDS; h++) if (pre.cnt[h]) h_mean[h] = rp[1 + h] / double(pre.cnt[h]);
                }
                // the gradient pass; its sums are the logged ones
                Acc acc(dev);
                opt_pol.zero_grad();
                for (const Chunk& C : chunks) {
                    auto loss = chunk(C, true, acc, n_total, cnt_total, split ? h_mean : nullptr, present);
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
                const auto tq1 = sync_now();
                value_step(vo, j);
                const auto tq2 = sync_now();
                t_pol += std::chrono::duration<double>(tq1 - tq0).count();
                t_val += std::chrono::duration<double>(tq2 - tq1).count();
            }
        }
        // ---- the sentinel gate's own PPO update: minibatches of GATE_MB, the same
        // clipped surrogate and epochs, entropy weight from the entropy file
        // (sentinel_mode), its own Adam on the gate network only
        if (!gate_batch.empty()) {
            static std::unique_ptr<torch::optim::Adam> gopt;
            auto gate_params = net->actors[P_SENTINEL]->parameters();
            if (!gopt) gopt = std::make_unique<torch::optim::Adam>(gate_params, torch::optim::AdamOptions(K.lr));
            for (auto& g : gopt->param_groups()) static_cast<torch::optim::AdamOptions&>(g.options()).lr(K.lr);
            const long n = long(gate_batch.size());
            std::vector<long> ch(n); std::vector<float> olp(n), adv(n);
            for (long i = 0; i < n; i++) {
                const Rec& rc = gate_batch[i].recs[0];
                ch[i] = rc.choice; olp[i] = rc.old_logp; adv[i] = gate_batch[i].adv;
            }
            auto GSP = gate_sp.to(dev), GSC = gate_sc.to(dev);
            auto GCH = torch::tensor(ch, torch::kLong).to(dev), GOLP = torch::tensor(olp).to(dev), GADV = torch::tensor(adv).to(dev);
            std::vector<long> ord(n); std::iota(ord.begin(), ord.end(), 0);
            std::mt19937 gshuf(uint32_t(it) * 40503u + 7u);
            double gsurr = 0, gent = 0, gkl = 0; long gsteps = 0;
            const double coef = K.head_entropy[HS_MODE];
            for (int ep = 0; ep < EPOCHS; ep++) {
                std::shuffle(ord.begin(), ord.end(), gshuf);
                for (long b0 = 0; b0 < n; b0 += GATE_MB) {
                    const long b1 = std::min(n, b0 + GATE_MB);
                    auto ix = torch::tensor(std::vector<long>(ord.begin() + b0, ord.begin() + b1), torch::kLong).to(dev);
                    auto lg = net->logits_split(P_SENTINEL, GSP.index_select(0, ix), GSC.index_select(0, ix))
                                  .narrow(1, head_offset(P_SENTINEL, HS_MODE), 2);
                    auto lsm = torch::log_softmax(lg, 1);
                    auto new_lp = lsm.gather(1, GCH.index_select(0, ix).view({-1, 1})).squeeze(1);
                    auto log_ratio = new_lp - GOLP.index_select(0, ix);
                    auto ratio = torch::exp(log_ratio);
                    auto A = GADV.index_select(0, ix);
                    auto surr = torch::min(ratio * A, torch::clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * A).mean();
                    auto H = -(lsm.exp() * lsm).sum(1).mean();
                    auto loss = -surr - coef * H;
                    gopt->zero_grad();
                    loss.backward();
                    torch::nn::utils::clip_grad_norm_(gate_params, CLIP_POL);
                    gopt->step();
                    gsurr += surr.item<double>(); gent += H.item<double>();
                    gkl += ((ratio - 1) - log_ratio).mean().item<double>(); gsteps++;
                }
            }
            torch::NoGradGuard ng;
            auto pl = torch::softmax(net->logits_split(P_SENTINEL, GSP, GSC).narrow(1, head_offset(P_SENTINEL, HS_MODE), 2), 1);
            std::cout << std::defaultfloat << std::setprecision(4) << "  gate: update " << gsteps << " steps x " << GATE_MB << ": surrogate " << gsurr / std::max(1L, gsteps)
                      << "  entropy " << gent / std::max(1L, gsteps) << "  kl " << gkl / std::max(1L, gsteps)
                      << "  P(place) now " << pl.select(1, 1).mean().item<double>() << " (placed in the batch: "
                      << std::count(ch.begin(), ch.end(), 1L) << "/" << n << ")" << std::endl;
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
        o << "  " << lfmt::label("rollout") << G << " games x " << (K.full_games ? std::string("full games (") + std::to_string(K.game_rounds) + " rounds max)" : std::to_string(R) + " rounds") << "   " << decisions << " decisions, "
          << samples.size() << " recorded (" << recs << " head records)"
          << (g_store.skipped ? "   STORE FULL: " + std::to_string(g_store.skipped) + " not recorded (lower RECORD_P or set STORE_GB)" : std::string())
          << "   " << kept.size() << " value rows   ~"
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
          << " s, update " << s_upd << " s: pack " << std::setprecision(2) << t_pack << ", policy " << t_pol
          << ", value " << t_val << ")\n" << std::defaultfloat;
        std::cout << o.str() << std::flush;

        ppo_log << it << "," << samples.size() << "," << recs << "," << ended.size() << "," << ended_a << "," << mean_rounds
                << "," << pl_mean << "," << vl_mean << "," << ent_mean << "," << kl_mean << "," << int(kl_stop) << "," << ev
                << "," << adv_mean << "," << adv_std << "," << adv_min << "," << adv_max << "," << 100.0 * clipped / clip_total
                << "," << K.lr << "," << EPOCHS << "," << K.game_rounds << "," << K.ore_radius << "," << s_roll << "," << s_upd;
        for (int h = 0; h < NUM_HEAD_IDS; h++) { ppo_log << ","; if (head_n[h]) ppo_log << head_ent[h] / head_n[h]; }
        ppo_log << std::endl;
        for (auto& e : ended)
            games_log << it << "," << e.kind << "," << e.rounds << "," << e.winner << "," << e.cond
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
                    s.greedy = greedy; s.record = false; s.real_maps_only = true;   // eval: the real maps
                    if (i == 0 && greedy) s.log = &log0;
                    start_game(s, S);
                }
                play(ev_slots, S, {&net, &frozen}, dev, /*stretch=*/0, /*restart=*/false, /*train=*/false, nullptr, nullptr);
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
