// Distil a trained teacher into a FRESH float teacher (policy and critic), with
// smoothed policy labels so the new teacher starts uncollapsed.
//
//   distill_teacher OLD_TEACHER.pt NEW_TEACHER.pt ITERATIONS MAP [MAP...]
//
// OLD_TEACHER.pt must have the current input size: convert an older checkpoint
// first (widen_input OLD.pt OLD62.pt: zero weights for new channels, same decisions).
//
// Labels: per head, the old teacher's masked distribution with its top action
// capped at SMOOTH_MAX (0.9): the mass above the cap goes to the other legal
// actions in proportion to their old probabilities (uniformly when they had
// none), so the alternatives keep at least 1 - SMOOTH_MAX together. A head whose
// top action is already below the cap is unchanged. SMOOTH_MAX=1 disables it.
// Critic: the new critic regresses the old critic's value (MSE) on the critic
// observation of the states the rollout visits (one per game per round).
//
//   env: GAMES (64) ROUNDS (10) STEPS (400) K_PER_HEAD (32) BUF_PER_HEAD (16384)
//        CRITIC_BUF (2048) CRITIC_BATCH (64) CRITIC_W (1) LR (1e-3) SMOOTH_MAX (0.9)
//        TEMP_MIN / TEMP_MAX (0.5 / 1.5) DEVICE INIT (resume the new teacher)
//        EVAL_EVERY (40) EVAL_GAMES (100): new vs old, greedy, sides alternating
// Map mix, game_rounds and ore_radius from the entropy file, as neural_train.
// Iteration 1: the old teacher plays; afterwards the new one (DAgger), sampled
// at each game's temperature.
#include <omp.h>
#include <torch/torch.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>

#include "../sim/florent_game.hpp"
#include "../sim/florent_heads.hpp"
#include "../sim/map_gen.hpp"
#include "entropy_file.hpp"
#include "model.hpp"

using namespace florent_rl;
static int env_int(const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; }
static float env_float(const char* k, float d) { const char* v = std::getenv(k); return v ? float(std::atof(v)) : d; }
static torch::Device g_dev(torch::kCPU);

// one decision's input as the teacher sees it: spatial clamped to 0..255 (uint8),
// scalars clamped to int16 (as neural_train's compact encoding)
struct Obs { std::vector<uint8_t> sp; std::vector<int16_t> sc; };
struct HeadRec { int policy = 0; std::shared_ptr<const Obs> obs; std::vector<uint8_t> mask; std::vector<float> tlog; };
struct CriticRec { std::vector<uint8_t> sp; std::vector<int32_t> sc; float v = 0; };
struct GameStep { std::vector<int> order; size_t pos = 0; bool in_round = false; int rounds = 0; bool active = true; };
struct Pend { int a = -1, p = -1; std::vector<float> obs; int8_t plan4[4] = {0, 0, 0, 0}; int own_l = 0; Targets tg;
             bool gate = false; Candidate gb; };   // gate: a builder that can build an aimed sentinel (sentinel_gate)

static std::shared_ptr<Obs> compact(const std::vector<float>& v) {
    auto o = std::make_shared<Obs>();
    o->sp.resize(fe::ACTOR_SPATIAL); o->sc.resize(fe::ACTOR_SCALARS);
    for (int i = 0; i < fe::ACTOR_SPATIAL; i++) o->sp[i] = uint8_t(std::min(255.f, std::max(0.f, v[i])));
    for (int i = 0; i < fe::ACTOR_SCALARS; i++) o->sc[i] = int16_t(std::min(32767.f, std::max(-32768.f, v[fe::ACTOR_SPATIAL + i])));
    return o;
}
static torch::Tensor batch_raw(const std::vector<const Obs*>& v) {
    const long n = long(v.size()), D = fe::ACTOR_SPATIAL + fe::ACTOR_SCALARS;
    std::vector<float> b(n * D);
    for (long r = 0; r < n; r++) {
        float* x = b.data() + r * D;
        for (int i = 0; i < fe::ACTOR_SPATIAL; i++) x[i] = v[r]->sp[i];
        for (int i = 0; i < fe::ACTOR_SCALARS; i++) x[fe::ACTOR_SPATIAL + i] = v[r]->sc[i];
    }
    return torch::from_blob(b.data(), {n, D}, torch::kFloat32).clone().to(g_dev);
}
static Chooser greedy() {
    return [](int, const float* l, const std::vector<uint8_t>& m, float& lp) { return argmax_choice(l, m, lp); };
}
// smoothed label (log-probabilities over the mask; -1e9 off the mask)
static std::vector<float> smoothed_label(const float* l, const std::vector<uint8_t>& mk, double cap) {
    const int n = int(mk.size());
    std::vector<double> p(n, 0.0);
    double mx = -1e30; int legal = 0;
    for (int i = 0; i < n; i++) if (mk[i]) { mx = std::max(mx, double(l[i])); legal++; }
    double z = 0;
    for (int i = 0; i < n; i++) if (mk[i]) { p[i] = std::exp(double(l[i]) - mx); z += p[i]; }
    for (int i = 0; i < n; i++) p[i] /= z;
    int top = -1; for (int i = 0; i < n; i++) if (mk[i] && (top < 0 || p[i] > p[top])) top = i;
    if (legal >= 2 && cap < 1.0 && p[top] > cap) {
        const double rest = 1.0 - p[top];          // old mass of the alternatives
        for (int i = 0; i < n; i++) {
            if (!mk[i] || i == top) continue;
            p[i] = rest > 1e-12 ? p[i] / rest * (1.0 - cap) : (1.0 - cap) / (legal - 1);
        }
        p[top] = cap;
    }
    std::vector<float> out(n, -1e9f);
    for (int i = 0; i < n; i++) if (mk[i]) out[i] = float(std::log(std::max(p[i], 1e-30)));
    return out;
}

int main(int argc, char** argv) {
    if (argc < 5) { std::cerr << "usage: distill_teacher OLD_TEACHER.pt NEW_TEACHER.pt ITERATIONS MAP...\n"; return 2; }
    const std::string opath = argv[1], npath = argv[2];
    const int iters = std::atoi(argv[3]);
    std::vector<std::string> maps(argv + 4, argv + argc);
    const int G = env_int("GAMES", 64), R = env_int("ROUNDS", 10);
    const double CAP = env_float("SMOOTH_MAX", 0.9f);
    torch::manual_seed(17);
    {
        const char* d = std::getenv("DEVICE");
        if ((!d || std::string(d) == "cuda") && torch::cuda::is_available()) g_dev = torch::Device(torch::kCUDA);
        std::cout << "distill_teacher: device " << (g_dev.is_cuda() ? "cuda" : "cpu") << ", label cap " << CAP << std::endl;
    }
    Teacher old_t;
    try { torch::load(old_t, opath, torch::Device(torch::kCPU)); }
    catch (const std::exception& e) {
        std::cerr << "cannot load " << opath << " (" << e.what() << ")\nAn older checkpoint needs its input widened first: widen_input OLD.pt OLD62.pt\n";
        return 3;
    }
    old_t->to(g_dev); old_t->eval();
    Teacher net;
    if (const char* init = std::getenv("INIT")) torch::load(net, init, torch::Device(torch::kCPU));
    net->to(g_dev);
    torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(env_float("LR", 1e-3f)));

    const std::string KNOB_PATH = std::getenv("ENTROPY_FILE") ? std::getenv("ENTROPY_FILE") : "entropy_file";
    Knobs K{}; K.lr = 0; K.epochs = 1;
    read_entropy_file(KNOB_PATH, K);
    std::mt19937 master(99);
    std::vector<Match> games(G);
    std::vector<std::mt19937> grng(G);
    const float TMIN = env_float("TEMP_MIN", 0.5f), TMAX = env_float("TEMP_MAX", 1.5f);
    std::vector<float> gtemp(G, 1.f);
    auto new_game = [&](int i, int start_round) {
        games[i] = Match();
        if (K.map_gen > 0.f && std::uniform_real_distribution<float>(0.f, 1.f)(master) < K.map_gen) {
            MapGenParams P; P.core_gap = K.core_distance; P.core_gap_min = K.core_distance_min; P.pit_share = K.pit_share;
            std::mt19937_64 mr(master());
            if (!games[i].load_bytes(generate_map(P, mr, nullptr), master())) { std::cerr << "generated map failed to load\n"; std::exit(3); }
        } else {
            games[i].load(maps[master() % maps.size()], master());
        }
        games[i].reset();
        games[i].game.round = start_round;
        games[i].game.max_turns = K.game_rounds;
        { const uint64_t os = master(); games[i].game.limit_ore(pick_ore_radius(K, os), os); }
        grng[i].seed(master());
        gtemp[i] = std::uniform_real_distribution<float>(TMIN, std::max(TMIN, TMAX))(grng[i]);
    };
    for (int i = 0; i < G; i++) new_game(i, int(long(i) * std::max(1, K.game_rounds) / G));

    // plays until a unit needs a forward (true) or the game is over / `limit` rounds done
    // (false); scripted turrets act here; one critic observation per round (crit != nullptr)
    auto advance = [&](Match& m, GameStep& gs, Pend& pd, int limit, std::vector<CriticRec>* crit) -> bool {
        for (;;) {
            if (!gs.in_round) {
                if (m.game.over || gs.rounds >= limit) return false;
                gs.order = m.acting_units(); gs.pos = 0; gs.in_round = true;
                if (crit && m.round_critic()) {
                    CriticRec c;
                    c.sp.resize(m.critic_channels.size());
                    for (size_t i = 0; i < c.sp.size(); i++) c.sp[i] = uint8_t(std::min(255, std::max(0, m.critic_channels[i])));
                    c.sc = m.critic_scalars;
                    crit->push_back(std::move(c));
                }
            }
            while (gs.pos < gs.order.size()) {
                const int a = gs.order[gs.pos];
                if (a >= int(m.game.ents.size()) || !m.game.ents[a].alive) { gs.pos++; continue; }
                const int type = m.game.ents[a].type;
                if (type == fe::SPIKE || type == fe::SENTINEL) {   // scripted turrets
                    const bool first = !m.memories.count(a); m.observe(a); Candidate c;
                    if (spike_action(m.game, a, c, first)) m.act(a, c);
                    gs.pos++; continue;
                }
                const int p = policy_of_type(type);
                if (p < 0) { gs.pos++; continue; }
                m.observe(a);
                m.encode_only(a, pd.obs);
                plan4_from_obs(pd.obs.data(), pd.plan4);
                pd.own_l = m.memories.at(a).own_count(fe::LAUNCHER);
                targets_from_memory(m.memories.at(a), pd.tg);
                if (!has_choice(m.game, a, pd.plan4, pd.own_l, &pd.tg)) { gs.pos++; continue; }
                pd.a = a; pd.p = p;
                pd.gate = p == P_BUILDER && sentinel_gate(m.game, a, pd.plan4, pd.own_l, &pd.tg, pd.gb);
                return true;
            }
            m.game.end_round(); gs.in_round = false; gs.rounds++;
        }
    };
    // batched forwards per policy: old (labels) and new logits on the same compact inputs
    auto forward_batch = [&](const std::vector<int>& idx, std::vector<Pend>& pend, std::vector<std::shared_ptr<Obs>>& ob,
                             std::vector<std::vector<float>>& OL, std::vector<std::vector<float>>& NL,
                             std::vector<std::vector<float>>& OG, std::vector<std::vector<float>>& NG) {
        torch::NoGradGuard ng;
        for (int p = 0; p < NUM_POLICIES; p++) {
            std::vector<int> sub; std::vector<const Obs*> rows;
            for (int g : idx) if (pend[g].p == p) { sub.push_back(g); rows.push_back(ob[g].get()); }
            if (sub.empty()) continue;
            auto rt = batch_raw(rows);
            auto ol = old_t->logits(p, rt).to(torch::kCPU).contiguous();
            auto nl = net->logits(p, rt).to(torch::kCPU).contiguous();
            const long k = ol.size(1);
            for (size_t b = 0; b < sub.size(); b++) {
                OL[sub[b]].assign(ol.data_ptr<float>() + b * k, ol.data_ptr<float>() + (b + 1) * k);
                NL[sub[b]].assign(nl.data_ptr<float>() + b * k, nl.data_ptr<float>() + (b + 1) * k);
            }
            if (p == P_BUILDER) {              // gate rows: both gate networks (P_SENTINEL) on the builder's input
                std::vector<long> gi;
                for (size_t b = 0; b < sub.size(); b++) if (pend[sub[b]].gate) gi.push_back(long(b));
                if (!gi.empty()) {
                    auto gr = rt.index_select(0, torch::tensor(gi, torch::kLong).to(g_dev));
                    auto og = old_t->logits(P_SENTINEL, gr).to(torch::kCPU).contiguous();
                    auto ng = net->logits(P_SENTINEL, gr).to(torch::kCPU).contiguous();
                    const long w = og.size(1);
                    for (size_t j = 0; j < gi.size(); j++) {
                        OG[sub[gi[j]]].assign(og.data_ptr<float>() + j * w, og.data_ptr<float>() + (j + 1) * w);
                        NG[sub[gi[j]]].assign(ng.data_ptr<float>() + j * w, ng.data_ptr<float>() + (j + 1) * w);
                    }
                }
            }
        }
    };

    const long BUF_CAP = env_int("BUF_PER_HEAD", 16384), CBUF = env_int("CRITIC_BUF", 2048);
    const int KH = env_int("K_PER_HEAD", 32), STEPS = env_int("STEPS", 400), CB = env_int("CRITIC_BATCH", 64);
    const double CW = env_float("CRITIC_W", 1.f);
    std::vector<std::vector<HeadRec>> buf(NUM_HEAD_IDS);
    std::vector<long> bpos(NUM_HEAD_IDS, 0);
    std::vector<CriticRec> cbuf; long cpos = 0;
    std::mt19937 brng(1234);
    const bool resumed = std::getenv("INIT") != nullptr;

    for (int it = 1; it <= iters; it++) {
        read_entropy_file(KNOB_PATH, K);
        for (auto& m : games) m.game.max_turns = K.game_rounds;
        const bool old_plays = it == 1 && !resumed;
        long agree = 0, compared = 0, nsamples = 0;
        std::vector<long> hagree(NUM_HEAD_IDS, 0), hcomp(NUM_HEAD_IDS, 0);
        net->eval();
        {
            std::vector<GameStep> gs(G);
            std::vector<Pend> pend(G);
            std::vector<std::shared_ptr<Obs>> ob(G);
            std::vector<std::vector<std::pair<int, HeadRec>>> ghr(G);
            std::vector<std::vector<CriticRec>> gcrit(G);
            std::vector<long> gagree(G, 0), gcomp(G, 0), nsamp(G, 0);
            std::vector<std::vector<long>> gha(G, std::vector<long>(NUM_HEAD_IDS, 0)), ghc(G, std::vector<long>(NUM_HEAD_IDS, 0));
            std::vector<char> need(G, 0);
            std::vector<long> gplaced(G, 0);
            for (;;) {
                #pragma omp parallel for schedule(dynamic, 1)
                for (int gi = 0; gi < G; gi++) {
                    need[gi] = gs[gi].active && advance(games[gi], gs[gi], pend[gi], R, &gcrit[gi]);
                    if (need[gi]) ob[gi] = compact(pend[gi].obs);
                }
                std::vector<int> idx;
                for (int gi = 0; gi < G; gi++) { if (need[gi]) idx.push_back(gi); else gs[gi].active = false; }
                if (idx.empty()) break;
                std::vector<std::vector<float>> OL(G), NL(G), OG(G), NG(G);
                forward_batch(idx, pend, ob, OL, NL, OG, NG);
                #pragma omp parallel for schedule(dynamic, 1)
                for (size_t k = 0; k < idx.size(); k++) {
                    const int gi = idx[k];
                    Match& m = games[gi];
                    Pend& pd = pend[gi];
                    const float* ol = OL[gi].data();
                    const float* nl = NL[gi].data();
                    if (pd.gate) {
                        // sentinel gate: label = the old gate network's (smoothed) distribution, for the new gate network
                        const int off = head_offset(P_SENTINEL, HS_MODE);
                        const float* og = OG[gi].data() + off;
                        const float* ng = NG[gi].data() + off;
                        const std::vector<uint8_t> gm = {1, 1};
                        ghc[gi][HS_MODE]++; gha[gi][HS_MODE] += (og[1] > og[0]) == (ng[1] > ng[0]);
                        const float* GL = old_plays ? og : ng;
                        const double T = std::max(1e-3, double(gtemp[gi]));
                        const double w1 = 1.0 / (1.0 + std::exp((double(GL[0]) - double(GL[1])) / T));
                        const bool place = std::uniform_real_distribution<double>(0, 1)(grng[gi]) < w1;
                        HeadRec hr; hr.policy = P_SENTINEL; hr.obs = ob[gi]; hr.mask = gm;
                        hr.tlog = smoothed_label(og, gm, CAP);
                        ghr[gi].push_back({HS_MODE, std::move(hr)});
                        if (place) { m.act(pd.a, pd.gb); gplaced[gi]++; nsamp[gi]++; gs[gi].pos++; continue; }
                    }
                    Decision dt = decide(m.game, pd.a, ol, greedy(), pd.plan4, pd.own_l, &pd.tg);
                    Decision ds = decide(m.game, pd.a, nl, greedy(), pd.plan4, pd.own_l, &pd.tg);
                    gcomp[gi]++;
                    bool same = dt.actions.size() == ds.actions.size();
                    for (size_t j = 0; same && j < dt.actions.size(); j++) {
                        auto &x = dt.actions[j], &y = ds.actions[j];
                        same = x.kind == y.kind && x.a == y.a && x.b == y.b && x.c == y.c && x.d == y.d;
                    }
                    gagree[gi] += same;
                    for (auto& rt : dt.records)
                        for (auto& rs : ds.records)
                            if (rs.head == rt.head) { ghc[gi][rt.head]++; gha[gi][rt.head] += rs.choice == rt.choice; break; }
                    const float* L = old_plays ? ol : nl;
                    std::mt19937& rng = grng[gi];
                    const double T = std::max(1e-3, double(gtemp[gi]));
                    Decision d = decide(m.game, pd.a, L, [&](int, const float* l, const std::vector<uint8_t>& mk, float& lp) {
                        double mx = -1e30; for (size_t i = 0; i < mk.size(); i++) if (mk[i]) mx = std::max(mx, double(l[i]));
                        std::vector<double> w(mk.size(), 0.0); double z = 0;
                        for (size_t i = 0; i < mk.size(); i++) if (mk[i]) { w[i] = std::exp((double(l[i]) - mx) / T); z += w[i]; }
                        double u = std::uniform_real_distribution<double>(0, z)(rng), c = 0; int pick = -1;
                        for (size_t i = 0; i < mk.size(); i++) if (mk[i]) { pick = int(i); c += w[i]; if (u <= c) break; }
                        lp = 0; return pick;
                    }, pd.plan4, pd.own_l, &pd.tg);
                    // labels: every head either decode visited (the old teacher's smoothed distribution)
                    std::vector<std::pair<int, std::vector<uint8_t>>> heads;
                    for (auto& hr : d.records) heads.push_back({hr.head, hr.mask});
                    for (auto& hr : dt.records) heads.push_back({hr.head, hr.mask});
                    for (auto& c : d.actions) m.act(pd.a, c);
                    std::shared_ptr<const Obs> o = ob[gi];
                    for (auto& hm : heads) {
                        HeadRec hr; hr.policy = pd.p; hr.obs = o; hr.mask = hm.second;
                        hr.tlog = smoothed_label(ol + head_offset(pd.p, hm.first), hm.second, CAP);
                        ghr[gi].push_back({hm.first, std::move(hr)});
                    }
                    nsamp[gi]++;
                    gs[gi].pos++;
                }
            }
            // critic targets: the old critic's value of every collected critic observation
            std::vector<CriticRec> fresh;
            for (int gi = 0; gi < G; gi++) for (auto& c : gcrit[gi]) fresh.push_back(std::move(c));
            {
                torch::NoGradGuard ng;
                for (size_t s = 0; s < fresh.size(); s += 64) {
                    const size_t e = std::min(fresh.size(), s + 64), n = e - s;
                    std::vector<uint8_t> sp(n * fe::CRITIC_SPATIAL); std::vector<int32_t> sc(n * fe::CRITIC_SCALARS);
                    for (size_t r = 0; r < n; r++) {
                        std::copy(fresh[s + r].sp.begin(), fresh[s + r].sp.end(), sp.begin() + r * fe::CRITIC_SPATIAL);
                        std::copy(fresh[s + r].sc.begin(), fresh[s + r].sc.end(), sc.begin() + r * fe::CRITIC_SCALARS);
                    }
                    auto spt = torch::from_blob(sp.data(), {long(n), fe::CRITIC_SPATIAL}, torch::kUInt8).to(g_dev);
                    auto sct = torch::from_blob(sc.data(), {long(n), fe::CRITIC_SCALARS}, torch::kInt32).to(g_dev);
                    auto v = old_t->value_split(spt, sct).to(torch::kCPU).contiguous();
                    for (size_t r = 0; r < n; r++) fresh[s + r].v = v.data_ptr<float>()[r];
                }
            }
            for (auto& c : fresh) {
                if (long(cbuf.size()) < CBUF) cbuf.push_back(std::move(c));
                else { cbuf[cpos % CBUF] = std::move(c); cpos++; }
            }
            for (int gi = 0; gi < G; gi++) {
                agree += gagree[gi]; compared += gcomp[gi]; nsamples += nsamp[gi];
                for (int h = 0; h < NUM_HEAD_IDS; h++) { hagree[h] += gha[gi][h]; hcomp[h] += ghc[gi][h]; }
                for (auto& [h, rec] : ghr[gi]) {
                    auto& b = buf[h];
                    if (long(b.size()) < BUF_CAP) b.push_back(std::move(rec));
                    else { b[bpos[h] % BUF_CAP] = std::move(rec); bpos[h]++; }
                }
                if (games[gi].game.over) new_game(gi, 0);
            }
        }
        // ---- training: per step K_PER_HEAD records of every head (heads weigh equally)
        // plus CRITIC_BATCH critic rows (MSE to the old critic)
        net->train();
        std::vector<double> hkl(NUM_HEAD_IDS, 0.0); std::vector<long> hkn(NUM_HEAD_IDS, 0);
        double vse = 0; long vn = 0;
        for (int step = 0; step < STEPS; step++) {
            std::vector<std::vector<std::pair<int, const HeadRec*>>> per_p(NUM_POLICIES);
            int heads_present = 0;
            for (int h = 0; h < NUM_HEAD_IDS; h++) {
                if (buf[h].empty()) continue;
                heads_present++;
                std::uniform_int_distribution<long> pick(0, long(buf[h].size()) - 1);
                for (int k = 0; k < KH; k++) { const HeadRec* rp = &buf[h][pick(brng)]; per_p[rp->policy].push_back({h, rp}); }
            }
            torch::Tensor loss = torch::zeros({}, g_dev);
            for (int p = 0; p < NUM_POLICIES && heads_present; p++) {
                auto& v = per_p[p];
                if (v.empty()) continue;
                std::vector<const Obs*> rows; for (auto& x : v) rows.push_back(x.second->obs.get());
                auto nl = net->logits(p, batch_raw(rows));
                for (int head : policy_heads(p)) {
                    std::vector<long> ri; std::vector<uint8_t> mk; std::vector<float> tg; const int nh = HEAD_SIZE[head];
                    for (long b = 0; b < long(v.size()); b++) if (v[b].first == head) {
                        ri.push_back(b);
                        mk.insert(mk.end(), v[b].second->mask.begin(), v[b].second->mask.end());
                        tg.insert(tg.end(), v[b].second->tlog.begin(), v[b].second->tlog.end());
                    }
                    if (ri.empty()) continue;
                    const long k = long(ri.size());
                    auto rix = torch::tensor(ri, torch::kLong).to(g_dev);
                    auto mask = torch::from_blob(mk.data(), {k, nh}, torch::kUInt8).to(g_dev).to(torch::kBool);
                    auto tlog = torch::from_blob(tg.data(), {k, nh}, torch::kFloat32).to(g_dev);
                    auto slog = torch::log_softmax(nl.index_select(0, rix).narrow(1, head_offset(p, head), nh).masked_fill(~mask, -1e9f), 1);
                    auto kl = (tlog.exp() * (tlog - slog)).masked_fill(~mask, 0.f).sum(1);
                    loss = loss + kl.mean() / double(heads_present);
                    hkl[head] += kl.sum().item<double>(); hkn[head] += k;
                }
            }
            if (!cbuf.empty() && CW > 0) {
                std::uniform_int_distribution<long> pick(0, long(cbuf.size()) - 1);
                const int n = std::min<int>(CB, int(cbuf.size()));
                std::vector<uint8_t> sp(size_t(n) * fe::CRITIC_SPATIAL); std::vector<int32_t> sc(size_t(n) * fe::CRITIC_SCALARS); std::vector<float> tv(n);
                for (int r = 0; r < n; r++) {
                    const CriticRec& c = cbuf[pick(brng)];
                    std::copy(c.sp.begin(), c.sp.end(), sp.begin() + size_t(r) * fe::CRITIC_SPATIAL);
                    std::copy(c.sc.begin(), c.sc.end(), sc.begin() + size_t(r) * fe::CRITIC_SCALARS);
                    tv[r] = c.v;
                }
                auto spt = torch::from_blob(sp.data(), {n, fe::CRITIC_SPATIAL}, torch::kUInt8).to(g_dev);
                auto sct = torch::from_blob(sc.data(), {n, fe::CRITIC_SCALARS}, torch::kInt32).to(g_dev);
                auto tvt = torch::from_blob(tv.data(), {n}, torch::kFloat32).to(g_dev);
                auto se = (net->value_split(spt, sct) - tvt).pow(2);
                loss = loss + CW * se.mean();
                vse += se.sum().item<double>(); vn += n;
            }
            opt.zero_grad(); loss.backward(); opt.step();
        }
        {
            std::ostringstream o;
            o << "  heads:";
            for (int h = 0; h < NUM_HEAD_IDS; h++) {
                if (!hkn[h] && !hcomp[h]) continue;
                o << " " << HEAD_NAMES[h] << " kl " << std::fixed << std::setprecision(3) << (hkn[h] ? hkl[h] / hkn[h] : 0.0)
                  << " agr " << std::setprecision(0) << (hcomp[h] ? 100.0 * hagree[h] / hcomp[h] : 0.0) << "% (" << hcomp[h] << ")";
            }
            std::cout << o.str() << std::endl;
        }
        double klsum = 0; long kln = 0; for (int h = 0; h < NUM_HEAD_IDS; h++) { klsum += hkl[h]; kln += hkn[h]; }
        std::cout << "iter " << it << (old_plays ? " (old plays)" : " (new plays)") << " samples " << nsamples
                  << " kl " << (kln ? klsum / kln : 0.0) << " critic_mse " << (vn ? vse / vn : 0.0)
                  << " action_agreement " << (compared ? double(agree) / compared : 0.0) << " (" << agree << "/" << compared << ")" << std::endl;
        // ---- eval: new vs old, greedy, full games, sides alternating
        const int EV = env_int("EVAL_EVERY", 40), EG = env_int("EVAL_GAMES", 100);
        if (EV > 0 && (it % EV == 0 || it == iters)) {
            net->eval();
            torch::NoGradGuard ng;
            int nw = 0, ow = 0; long nti = 0, oti = 0;
            std::mt19937 er(0xE7A1u + unsigned(it));
            std::vector<Match> ev(EG);
            for (int e = 0; e < EG; e++) {
                const uint32_t s1 = er(), s2 = er();
                ev[e].load(maps[(e / 2) % maps.size()], s1); ev[e].reset();
                ev[e].game.max_turns = K.game_rounds; ev[e].game.limit_ore(pick_ore_radius(K, s2), s2);
            }
            std::vector<GameStep> gs(EG);
            std::vector<Pend> pend(EG);
            std::vector<std::shared_ptr<Obs>> ob(EG);
            std::vector<char> need(EG, 0);
            for (;;) {
                #pragma omp parallel for schedule(dynamic, 1)
                for (int e = 0; e < EG; e++) {
                    need[e] = gs[e].active && advance(ev[e], gs[e], pend[e], 1 << 30, nullptr);
                    if (need[e]) ob[e] = compact(pend[e].obs);
                }
                std::vector<int> idx;
                for (int e = 0; e < EG; e++) { if (need[e]) idx.push_back(e); else gs[e].active = false; }
                if (idx.empty()) break;
                std::vector<std::vector<float>> OL(EG), NL(EG), OG(EG), NG(EG);
                forward_batch(idx, pend, ob, OL, NL, OG, NG);
                #pragma omp parallel for schedule(dynamic, 1)
                for (size_t k = 0; k < idx.size(); k++) {
                    const int e = idx[k];
                    Pend& pd = pend[e];
                    const bool mine = ev[e].game.ents[pd.a].team == (e & 1);
                    if (pd.gate) {                     // greedy gate, each side its own gate network
                        const float* GL = (mine ? NG[e].data() : OG[e].data()) + head_offset(P_SENTINEL, HS_MODE);
                        if (GL[1] > GL[0]) { ev[e].act(pd.a, pd.gb); gs[e].pos++; continue; }
                    }
                    Decision d = decide(ev[e].game, pd.a, mine ? NL[e].data() : OL[e].data(), greedy(), pd.plan4, pd.own_l, &pd.tg);
                    for (auto& c : d.actions) ev[e].act(pd.a, c);
                    gs[e].pos++;
                }
            }
            for (int e = 0; e < EG; e++) {
                const int nt = e & 1;
                const fe::Game& gm = ev[e].game;
                if (gm.winner == nt) nw++; else if (gm.winner >= 0) ow++;
                nti += gm.teams[nt].ti_collected; oti += gm.teams[1 - nt].ti_collected;
            }
            std::cout << "eval " << it << ": new vs old (greedy, " << EG << " games, " << K.game_rounds << " rounds): new wins "
                      << nw << ", old wins " << ow << ", mean Ti collected new " << nti / std::max(1, EG) << " old " << oti / std::max(1, EG) << std::endl;
        }
        if (it % 5 == 0 || it == iters) torch::save(net, npath);
    }
    std::cout << "DISTILL_TEACHER COMPLETE new teacher=" << npath << std::endl;
}
