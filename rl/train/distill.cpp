// Distil the float teacher into the integer-exact int4 student.
//
//   distill TEACHER.pt OUT_STUDENT.pt OUT.json ITERATIONS MAP [MAP...]
//   env: GAMES (4) ROUNDS (40) EPOCHS (2) LR (3e-3) VECTORS (vector dump path)
//        DEVICE (cuda if available | cpu): teacher, student and the KL fit run there
//        TEMP_MIN / TEMP_MAX (0.5 / 1.5): each game samples with its own temperature
//        BUF_PER_HEAD (16384) K_PER_HEAD (32) STEPS (400): per-head replay buffers; each of
//        the STEPS training steps per iteration samples K_PER_HEAD records of every head
//        (heads weigh equally in the loss). EPOCHS is unused.
//        EVAL_EVERY (40: every N iterations the student plays EVAL_GAMES greedy
//        games against the teacher, sides alternating; 0: off) EVAL_GAMES (100)
//
// The first games start staggered (game i at round i * game_rounds / GAMES) so
// every phase of the game is covered from iteration 1; later games start at 0.
// Training games use the same map mix as neural_train (entropy file: map_gen,
// core_distance, pit_share, ore_radius, game_rounds); eval games use the map files.
// Games advance in lockstep (game logic on OpenMP threads, one batched forward per
// policy for teacher and student on the device), as neural_train's rollout.
// Each iteration plays ROUNDS more rounds of GAMES persistent games (teacher
// plays on iteration 1, the student afterwards: DAgger), records every
// decision's raw observation and its participating heads (masks from the
// acting policy's decode), labels each head with the teacher's masked
// distribution, then minimises KL(teacher || student) per head record.
// At every collected decision it also decodes greedily with both models on the
// same state and counts identical actions (float/quantized agreement).
// Writes the student checkpoint, the JSON export for bots/econ/student.py, and
// optionally test vectors (quantized input -> integer logits) for exactness.
#include <omp.h>
#include <torch/torch.h>
#include "entropy_file.hpp"

#include <fstream>
#include <memory>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <random>

#include "../sim/florent_game.hpp"
#include "../sim/florent_heads.hpp"
#include "../sim/map_gen.hpp"
#include "model.hpp"
#include "student.hpp"

using namespace florent_rl;
static int env_int(const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; }
static float env_float(const char* k, float d) { const char* v = std::getenv(k); return v ? float(std::atof(v)) : d; }

// one head record in a per-head replay buffer: the student's quantized input (shared
// by the heads of one decision), the head's mask and the teacher's log-probabilities
// over it (the teacher is fixed, so old records keep valid labels)
struct HeadRec {
    int policy = 0;
    std::shared_ptr<const std::vector<uint8_t>> q;
    std::vector<uint8_t> mask;
    std::vector<float> tlog;
};
struct Sample {
    int policy;
    std::vector<int16_t> raw;
    std::vector<int32_t> q;
    std::vector<std::pair<int, std::vector<uint8_t>>> heads;   // participating head, mask
};

// lockstep game state and a waiting unit's inputs (see advance / forward_batch in main)
struct GameStep { std::vector<int> order; size_t pos = 0; bool in_round = false; int rounds = 0; bool active = true; };
struct Pend { int a = -1, p = -1; std::vector<float> obs; std::vector<int32_t> q; int8_t plan4[4] = {0, 0, 0, 0}; int own_l = 0; Targets tg; };
static torch::Device g_dev(torch::kCPU);
// the teacher sees exactly what neural_train fed it (Match::encode_compact): spatial
// clamped to 0..255, scalars to int16 (the student quantizes the raw values itself)
// raw encoder values as int16 with saturation (not wrap-around): a large scalar such as
// a team's Ti above 32767 stays large, as in the deployed bot's own quantization
static std::vector<int16_t> to_i16(const std::vector<float>& v) {
    std::vector<int16_t> o(v.size());
    for (size_t i = 0; i < v.size(); i++) o[i] = int16_t(std::min(32767.f, std::max(-32768.f, v[i])));
    return o;
}
static void clamp_teacher_row(float* x) {
    for (int i = 0; i < fe::ACTOR_SPATIAL; i++) x[i] = std::min(255.f, std::max(0.f, x[i]));
    for (int i = 0; i < fe::ACTOR_SCALARS; i++) x[fe::ACTOR_SPATIAL + i] = std::min(32767.f, std::max(-32768.f, x[fe::ACTOR_SPATIAL + i]));
}     // models live here (cuda if available, DEVICE=cpu to force)
static torch::Tensor raw_tensor(const std::vector<int16_t>& r) {
    std::vector<float> v(r.begin(), r.end());
    return torch::from_blob(v.data(), {1, (long)v.size()}, torch::kFloat32).clone().to(g_dev);
}
static torch::Tensor q_tensor(const std::vector<int32_t>& r) {
    std::vector<float> v(r.begin(), r.end());
    return torch::from_blob(v.data(), {1, (long)v.size()}, torch::kFloat32).clone().to(g_dev);
}

static Chooser greedy() {
    return [](int, const float* l, const std::vector<uint8_t>& m, float& lp) { return argmax_choice(l, m, lp); };
}

static void dump_json_int(std::ostream& os, const torch::Tensor& t) {
    auto v = t.detach().to(torch::kCPU).to(torch::kFloat64).contiguous().flatten();
    os << "[";
    for (long i = 0; i < v.numel(); i++) os << (i ? "," : "") << (long long)std::llround(v[i].item<double>());
    os << "]";
}

static void export_student(Student& st, const std::string& path) {
    std::ofstream os(path);
    os << "{\"format\":\"florent-student-v2\",\"bits\":8,\"amax\":" << ST_AMAX << ",\"logit_shift\":" << ST_LOGIT_SHIFT
       << ",\"channels\":" << ST_CH << ",\"hidden\":" << ST_HIDDEN
       << ",\"actor_ch\":" << fe::ACTOR_CH << ",\"scalars\":" << fe::ACTOR_SCALARS << ",\"policies\":[";
    for (int p = 0; p < NUM_POLICIES; p++) {
        auto* a = st->a(p);
        // exactly the integers the forward pass uses (student.hpp qw / qb)
        auto qwt = [](const torch::Tensor& w) { return torch::round(torch::clamp(w * ST_WSCALE, -ST_WMAX, ST_WMAX)); };
        auto qbs = [](const torch::Tensor& b) { return torch::round(torch::clamp(b * ST_BSCALE, -ST_BIAS_MAX, ST_BIAS_MAX)); };
        os << (p ? "," : "") << "{\"policy\":" << p << ",\"nlogits\":" << a->nlogits << ",\"shifts\":[";
        for (int i = 0; i < ST_NSHIFT; i++) os << (i ? "," : "") << a->shifts[i];
        os << "],\"stem_w\":"; dump_json_int(os, qwt(a->stem_w));
        os << ",\"stem_b\":"; dump_json_int(os, qbs(a->stem_b));
        for (int b = 0; b < ST_LAYERS; b++) {
            os << ",\"pw_w" << b << "\":"; dump_json_int(os, qwt(a->pw_w[b]));
            os << ",\"pw_b" << b << "\":"; dump_json_int(os, qbs(a->pw_b[b]));
            os << ",\"dw_w" << b << "\":"; dump_json_int(os, qwt(a->dw_w[b]));
            os << ",\"dw_b" << b << "\":"; dump_json_int(os, qbs(a->dw_b[b]));
        }
        os << ",\"fc_w\":"; dump_json_int(os, qwt(a->fc_w));
        os << ",\"fc_b\":"; dump_json_int(os, qbs(a->fc_b));
        os << ",\"out_w\":"; dump_json_int(os, qwt(a->out_w));
        os << ",\"out_b\":"; dump_json_int(os, qbs(a->out_b));
        os << "}";
    }
    os << "]}\n";
}

int main(int argc, char** argv) {
    if (argc < 6) { std::cerr << "usage: distill TEACHER.pt OUT_STUDENT.pt OUT.json ITERATIONS MAP...\n"; return 2; }
    const std::string tpath = argv[1], spath = argv[2], jpath = argv[3];
    const int iters = std::atoi(argv[4]);
    std::vector<std::string> maps(argv + 5, argv + argc);
    const int G = env_int("GAMES", 4), R = env_int("ROUNDS", 40), EPOCHS = env_int("EPOCHS", 2), MB = 256;
    if (std::getenv("THREADS")) torch::set_num_threads(env_int("THREADS", 0));   // default: torch picks (all cores)
    torch::manual_seed(11);
    {
        const char* d = std::getenv("DEVICE");
        if ((!d || std::string(d) == "cuda") && torch::cuda::is_available()) g_dev = torch::Device(torch::kCUDA);
        std::cout << "distill: device " << (g_dev.is_cuda() ? "cuda" : "cpu") << std::endl;
        // the student's integer-exact forward must not use TF32
        at::globalContext().setAllowTF32CuBLAS(false);
        at::globalContext().setAllowTF32CuDNN(false);
    }
    // checkpoints load on the CPU whatever device saved them, then move to g_dev
    Teacher teacher; torch::load(teacher, tpath, torch::Device(torch::kCPU)); teacher->to(g_dev); teacher->eval();
    Student st;
    bool resumed = false;
    if (const char* init = std::getenv("INIT")) {
        torch::load(st, init, torch::Device(torch::kCPU));
        for (int p = 0; p < NUM_POLICIES; p++) st->a(p)->sync_shifts_from_buffer();
        resumed = true;
    }
    st->to(g_dev);
    torch::optim::Adam opt(st->parameters(), torch::optim::AdamOptions(env_float("LR", 3e-3f)));
    // game_rounds / ore_radius from the same entropy file as neural_train (re-read every iteration)
    const std::string KNOB_PATH = std::getenv("ENTROPY_FILE") ? std::getenv("ENTROPY_FILE") : "entropy_file";
    Knobs K{}; K.lr = 0; K.epochs = 1;
    read_entropy_file(KNOB_PATH, K);
    std::mt19937 master(99);
    std::vector<Match> games(G);
    std::vector<std::mt19937> grng(G);
    // per-game sampling temperature, uniform in [TEMP_MIN, TEMP_MAX] (drawn when the game starts):
    // more varied states; the labels stay the teacher's T = 1 distribution
    const float TMIN = env_float("TEMP_MIN", 0.5f), TMAX = env_float("TEMP_MAX", 1.5f);
    std::vector<float> gtemp(G, 1.f);
    auto new_game = [&](int i, int start_round) {
        games[i] = Match();
        // a generated map with probability map_gen (entropy file: map_gen, core_distance, pit_share), as neural_train
        if (K.map_gen > 0.f && std::uniform_real_distribution<float>(0.f, 1.f)(master) < K.map_gen) {
            MapGenParams P; P.core_gap = K.core_distance; P.pit_share = K.pit_share;
            std::mt19937_64 mr(master());
            const std::string bytes = generate_map(P, mr, nullptr);
            if (!games[i].load_bytes(bytes, master())) { std::cerr << "generated map failed to load\n"; std::exit(3); }
        } else {
            games[i].load(maps[master() % maps.size()], master());
        }
        games[i].reset();
        games[i].game.round = start_round;      // staggered first games (as neural_train): every game phase from iteration 1
        games[i].game.max_turns = K.game_rounds;
        games[i].game.limit_ore(K.ore_radius, master());
        grng[i].seed(master());
        gtemp[i] = std::uniform_real_distribution<float>(TMIN, std::max(TMIN, TMAX))(grng[i]);
    };
    for (int i = 0; i < G; i++) new_game(i, int(long(i) * std::max(1, K.game_rounds) / G));

    // ---- lockstep stepping (shared by the rollout and the eval)
    // advance(): plays the game until a unit needs a forward (true, pend filled) or the
    // game is over / has played `limit` rounds this call series (false). Spikes act here.
    auto advance = [&](Match& m, GameStep& gs, Pend& pd, int limit) -> bool {
        for (;;) {
            if (!gs.in_round) {
                if (m.game.over || gs.rounds >= limit) return false;
                gs.order = m.acting_units(); gs.pos = 0; gs.in_round = true;
            }
            while (gs.pos < gs.order.size()) {
                const int a = gs.order[gs.pos];
                if (a >= int(m.game.ents.size()) || !m.game.ents[a].alive) { gs.pos++; continue; }
                const int type = m.game.ents[a].type;
                if (type == fe::SPIKE) {
                    const bool first = !m.memories.count(a); m.observe(a); Candidate c;
                    if (spike_action(m.game, a, c, first)) m.act(a, c);
                    gs.pos++; continue;
                }
                const int p = policy_of_type(type);
                if (p < 0) { gs.pos++; continue; }
                m.observe(a);                            // memory updates every turn, as deployed
                m.encode_only(a, pd.obs);                // first: the conveyor planner mask reads it
                plan4_from_obs(pd.obs.data(), pd.plan4);
                pd.own_l = m.memories.at(a).own_count(fe::LAUNCHER);
                targets_from_memory(m.memories.at(a), pd.tg);
                if (!has_choice(m.game, a, pd.plan4, pd.own_l, &pd.tg)) { gs.pos++; continue; }
                pd.a = a; pd.p = p;
                const std::vector<int16_t> raw = to_i16(pd.obs);
                pd.q.resize(raw.size()); quantize_actor(raw.data(), pd.q.data());
                return true;
            }
            m.game.end_round(); gs.in_round = false; gs.rounds++;
        }
    };
    // one batched forward per policy over the waiting games: teacher (raw) and student (quantized)
    auto forward_batch = [&](const std::vector<int>& idx, std::vector<Pend>& pend,
                             std::vector<std::vector<float>>* TL, std::vector<std::vector<float>>* SL) {
        torch::NoGradGuard ng;
        for (int p = 0; p < NUM_POLICIES; p++) {
            std::vector<int> sub;
            for (int g : idx) if (pend[g].p == p) sub.push_back(g);
            if (sub.empty()) continue;
            const long D = long(pend[sub[0]].obs.size()), n = long(sub.size());
            std::vector<float> rb(n * D), qb(n * D);
            #pragma omp parallel for
            for (long b = 0; b < n; b++) {
                std::copy(pend[sub[b]].obs.begin(), pend[sub[b]].obs.end(), rb.begin() + b * D);
                clamp_teacher_row(rb.data() + b * D);
                std::copy(pend[sub[b]].q.begin(), pend[sub[b]].q.end(), qb.begin() + b * D);
            }
            auto rt = torch::from_blob(rb.data(), {n, D}, torch::kFloat32).to(g_dev);
            auto qt = torch::from_blob(qb.data(), {n, D}, torch::kFloat32).to(g_dev);
            auto tl = teacher->logits(p, rt).to(torch::kCPU).contiguous();
            auto sl = (st->a(p)->forward(qt) / double(1 << ST_LOGIT_SHIFT)).to(torch::kCPU).contiguous();
            const long nl = tl.size(1);
            for (long b = 0; b < n; b++) {
                (*TL)[sub[b]].assign(tl.data_ptr<float>() + b * nl, tl.data_ptr<float>() + (b + 1) * nl);
                (*SL)[sub[b]].assign(sl.data_ptr<float>() + b * nl, sl.data_ptr<float>() + (b + 1) * nl);
            }
        }
    };

    // per-head replay buffers (DISTILL: every training batch takes K_PER_HEAD records of every head)
    const long BUF_CAP = env_int("BUF_PER_HEAD", 16384);
    const int KH = env_int("K_PER_HEAD", 32), STEPS = env_int("STEPS", 400);
    std::vector<std::vector<HeadRec>> buf(NUM_HEAD_IDS);
    std::vector<long> bpos(NUM_HEAD_IDS, 0);
    std::mt19937 brng(1234);

    for (int it = 1; it <= iters; it++) {
        read_entropy_file(KNOB_PATH, K);
        for (auto& m : games) m.game.max_turns = K.game_rounds;
        bool teacher_plays = it == 1 && !resumed;
        std::vector<Sample> data;                // full positions: only for the calibration / the exactness vectors
        const bool keep_full = (it == 1 && !resumed) || (it == iters && std::getenv("VECTORS"));
        long nsamples = 0;
        long agree = 0, compared = 0;
        std::vector<long> hagree(NUM_HEAD_IDS, 0), hcomp(NUM_HEAD_IDS, 0);   // per-head greedy agreement
        st->eval();
        {
            // ---- lockstep rollout (as neural_train): every game advances to its next deciding
            // unit in parallel (OpenMP), then ONE batched forward per policy for teacher and
            // student over all waiting games (on the device), then the games act in parallel.
            std::vector<GameStep> gs(G);
            std::vector<Pend> pend(G);
            std::vector<std::vector<Sample>> gdata(G);
            std::vector<std::vector<std::pair<int, HeadRec>>> ghr(G);   // (head, record) per game
            std::vector<long> gagree(G, 0), gcomp(G, 0), nsamp_g(G, 0);
            std::vector<std::vector<long>> ghagree(G, std::vector<long>(NUM_HEAD_IDS, 0)), ghcomp(G, std::vector<long>(NUM_HEAD_IDS, 0));
            std::vector<char> need(G, 0);
            for (;;) {
                #pragma omp parallel for schedule(dynamic, 1)
                for (int gi = 0; gi < G; gi++) need[gi] = gs[gi].active && advance(games[gi], gs[gi], pend[gi], R);
                std::vector<int> idx;
                for (int gi = 0; gi < G; gi++) { if (need[gi]) idx.push_back(gi); else gs[gi].active = false; }
                if (idx.empty()) break;
                std::vector<std::vector<float>> TL(G), SL(G);
                forward_batch(idx, pend, &TL, &SL);
                #pragma omp parallel for schedule(dynamic, 1)
                for (size_t k = 0; k < idx.size(); k++) {
                    const int gi = idx[k];
                    Match& m = games[gi];
                    Pend& pd = pend[gi];
                    const float* tl = TL[gi].data();
                    const float* sl = SL[gi].data();
                    Sample smp; smp.policy = pd.p; smp.raw = to_i16(pd.obs); smp.q = pd.q;
                    // agreement: greedy decode with each model on this exact state
                    Decision dt = decide(m.game, pd.a, tl, greedy(), pd.plan4, pd.own_l, &pd.tg);
                    Decision ds = decide(m.game, pd.a, sl, greedy(), pd.plan4, pd.own_l, &pd.tg);
                    gcomp[gi]++;
                    bool same = dt.actions.size() == ds.actions.size();
                    for (size_t j = 0; same && j < dt.actions.size(); j++) {
                        auto &x = dt.actions[j], &y = ds.actions[j];
                        same = x.kind == y.kind && x.a == y.a && x.b == y.b && x.c == y.c && x.d == y.d;
                    }
                    gagree[gi] += same;
                    // per head: both greedy decodes visited the head -> same choice?
                    for (auto& rt : dt.records)
                        for (auto& rs : ds.records)
                            if (rs.head == rt.head) { ghcomp[gi][rt.head]++; ghagree[gi][rt.head] += rs.choice == rt.choice; break; }
                    // act with a sampled decision of the playing model, at this game's temperature
                    const float* L = teacher_plays ? tl : sl;
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
                    for (auto& hr : d.records) smp.heads.push_back({hr.head, hr.mask});
                    for (auto& hr : dt.records) smp.heads.push_back({hr.head, hr.mask});   // the teacher's heads too
                    for (auto& c : d.actions) m.act(pd.a, c);
                    {   // head records for the replay buffers: the teacher's masked log-softmax per head
                        auto qv = std::make_shared<std::vector<uint8_t>>(pd.q.size());
                        for (size_t i = 0; i < pd.q.size(); i++) (*qv)[i] = uint8_t(pd.q[i]);
                        for (auto& hm : smp.heads) {
                            const int head = hm.first, n = HEAD_SIZE[head], off = head_offset(pd.p, head);
                            HeadRec hr; hr.policy = pd.p; hr.q = qv; hr.mask = hm.second; hr.tlog.assign(n, -1e9f);
                            double mx = -1e30; for (int i = 0; i < n; i++) if (hm.second[i]) mx = std::max(mx, double(tl[off + i]));
                            double z = 0; for (int i = 0; i < n; i++) if (hm.second[i]) z += std::exp(double(tl[off + i]) - mx);
                            for (int i = 0; i < n; i++) if (hm.second[i]) hr.tlog[i] = float(double(tl[off + i]) - mx - std::log(z));
                            ghr[gi].push_back({head, std::move(hr)});
                        }
                    }
                    nsamp_g[gi]++;
                    if (keep_full) gdata[gi].push_back(std::move(smp));
                    gs[gi].pos++;
                }
            }
            for (int gi = 0; gi < G; gi++) {
                agree += gagree[gi]; compared += gcomp[gi];
                for (int h = 0; h < NUM_HEAD_IDS; h++) { hagree[h] += ghagree[gi][h]; hcomp[h] += ghcomp[gi][h]; }
                for (auto& x : gdata[gi]) data.push_back(std::move(x));
                nsamples += nsamp_g[gi];
                for (auto& [h, rec] : ghr[gi]) {                 // into the per-head buffers (oldest replaced)
                    auto& b = buf[h];
                    if (long(b.size()) < BUF_CAP) b.push_back(std::move(rec));
                    else { b[bpos[h] % BUF_CAP] = std::move(rec); bpos[h]++; }
                }
                if (games[gi].game.over) new_game(gi, 0);
            }
        }
        // one-time shift calibration on the first batch
        if (it == 1 && !resumed) {
            for (int p = 0; p < NUM_POLICIES; p++) {
                std::vector<float> buf; long n = 0;
                for (auto& s : data) if (s.policy == p && n < 512) { buf.insert(buf.end(), s.q.begin(), s.q.end()); n++; }
                if (!n) continue;
                st->a(p)->calibrate(torch::from_blob(buf.data(), {n, (long)data[0].q.size()}, torch::kFloat32).clone().to(g_dev));
            }
        }
        // ---- KL distillation from the per-head buffers: each step samples K_PER_HEAD records of
        // every head with records; loss = mean over heads of the head's mean KL (heads weigh equally)
        st->train();
        double kl_sum = 0; long kl_n = 0;
        std::vector<torch::Tensor> hkl(NUM_HEAD_IDS);
        std::vector<long> hkn(NUM_HEAD_IDS, 0);
        const long D = long(fe::ACTOR_SPATIAL + fe::ACTOR_SCALARS);
        for (int step = 0; step < STEPS; step++) {
            // sampled (head, record) per policy
            std::vector<std::vector<std::pair<int, const HeadRec*>>> per_p(NUM_POLICIES);
            int heads_present = 0;
            for (int h = 0; h < NUM_HEAD_IDS; h++) {
                if (buf[h].empty()) continue;
                heads_present++;
                std::uniform_int_distribution<long> pick(0, long(buf[h].size()) - 1);
                for (int k = 0; k < KH; k++) { const HeadRec* rp = &buf[h][pick(brng)]; per_p[rp->policy].push_back({h, rp}); }
            }
            if (!heads_present) break;
            torch::Tensor loss = torch::zeros({}, g_dev);
            for (int p = 0; p < NUM_POLICIES; p++) {
                auto& v = per_p[p];
                if (v.empty()) continue;
                const long n = long(v.size());
                std::vector<uint8_t> qb(n * D);
                for (long b = 0; b < n; b++) std::copy(v[b].second->q->begin(), v[b].second->q->end(), qb.begin() + b * D);
                auto qt = torch::from_blob(qb.data(), {n, D}, torch::kUInt8).to(g_dev).to(torch::kFloat32);
                auto sl = st->a(p)->forward(qt) / double(1 << ST_LOGIT_SHIFT);
                for (int head : policy_heads(p)) {
                    std::vector<long> rows; std::vector<uint8_t> mk; std::vector<float> tg; const int nh = HEAD_SIZE[head];
                    for (long b = 0; b < n; b++) if (v[b].first == head) {
                        rows.push_back(b);
                        mk.insert(mk.end(), v[b].second->mask.begin(), v[b].second->mask.end());
                        tg.insert(tg.end(), v[b].second->tlog.begin(), v[b].second->tlog.end());
                    }
                    if (rows.empty()) continue;
                    const long k = long(rows.size());
                    auto ri = torch::tensor(rows, torch::kLong).to(g_dev);
                    auto mask = torch::from_blob(mk.data(), {k, nh}, torch::kUInt8).to(g_dev).to(torch::kBool);
                    auto tlog = torch::from_blob(tg.data(), {k, nh}, torch::kFloat32).to(g_dev);
                    auto slog = torch::log_softmax(sl.index_select(0, ri).narrow(1, head_offset(p, head), nh).masked_fill(~mask, -1e9f), 1);
                    auto kl = (tlog.exp() * (tlog - slog)).masked_fill(~mask, 0.f).sum(1);    // [k]
                    auto hm = kl.mean();
                    loss = loss + hm / double(heads_present);
                    auto ks = kl.sum().detach(); hkl[head] = hkl[head].defined() ? hkl[head] + ks : ks; hkn[head] += k;
                    kl_n += k;
                }
            }
            opt.zero_grad(); loss.backward(); opt.step();
        }
        for (int h = 0; h < NUM_HEAD_IDS; h++) if (hkn[h]) kl_sum += hkl[h].item<double>();
        {   // per head: KL (last epoch pass included) and greedy agreement, heads with records only
            std::ostringstream o;
            o << "  heads:";
            for (int h = 0; h < NUM_HEAD_IDS; h++) {
                if (!hkn[h] && !hcomp[h]) continue;
                o << " " << HEAD_NAMES[h] << " kl " << std::fixed << std::setprecision(2)
                  << (hkn[h] ? hkl[h].item<double>() / double(hkn[h]) : 0.0)
                  << " agr " << std::setprecision(0) << (hcomp[h] ? 100.0 * hagree[h] / hcomp[h] : 0.0) << "% (" << hcomp[h] << ")";
            }
            std::cout << o.str() << std::endl;
        }
        std::cout << "iter " << it << (teacher_plays ? " (teacher plays)" : " (student plays)")
                  << " samples " << nsamples << " kl " << (kl_n ? kl_sum / kl_n : 0)
                  << " action_agreement " << (compared ? double(agree) / compared : 0) << " (" << agree << "/" << compared << ")"
                  << std::endl;
        // ---- match eval: student vs teacher, greedy, full games (game_rounds), sides alternating
        const int EV = env_int("EVAL_EVERY", 40), EG = env_int("EVAL_GAMES", 100);
        if (EV > 0 && (it % EV == 0 || it == iters)) {
            st->eval();
            torch::NoGradGuard ng;
            int sw = 0, tw = 0; long sti = 0, tti = 0;
            std::mt19937 er(0xE7A1u + unsigned(it));
            std::vector<Match> ev(EG);
            for (int e = 0; e < EG; e++) {
                const uint32_t s1 = er(), s2 = er();
                ev[e].load(maps[(e / 2) % maps.size()], s1); ev[e].reset();
                ev[e].game.max_turns = K.game_rounds; ev[e].game.limit_ore(K.ore_radius, s2);
            }
            // lockstep, batched forwards (as the rollout); student plays team e & 1, greedy
            std::vector<GameStep> gs(EG);
            std::vector<Pend> pend(EG);
            std::vector<char> need(EG, 0);
            for (;;) {
                #pragma omp parallel for schedule(dynamic, 1)
                for (int e = 0; e < EG; e++) need[e] = gs[e].active && advance(ev[e], gs[e], pend[e], 1 << 30);
                std::vector<int> idx;
                for (int e = 0; e < EG; e++) { if (need[e]) idx.push_back(e); else gs[e].active = false; }
                if (idx.empty()) break;
                std::vector<std::vector<float>> TL(EG), SL(EG);
                forward_batch(idx, pend, &TL, &SL);
                #pragma omp parallel for schedule(dynamic, 1)
                for (size_t k = 0; k < idx.size(); k++) {
                    const int e = idx[k];
                    Pend& pd = pend[e];
                    const bool stu = ev[e].game.ents[pd.a].team == (e & 1);
                    Decision d = decide(ev[e].game, pd.a, stu ? SL[e].data() : TL[e].data(), greedy(), pd.plan4, pd.own_l, &pd.tg);
                    for (auto& c : d.actions) ev[e].act(pd.a, c);
                    gs[e].pos++;
                }
            }
            for (int e = 0; e < EG; e++) {
                const int st_team = e & 1;
                const fe::Game& gm = ev[e].game;
                if (gm.winner == st_team) sw++; else if (gm.winner >= 0) tw++;
                sti += gm.teams[st_team].ti_collected; tti += gm.teams[1 - st_team].ti_collected;
            }
            std::cout << "eval " << it << ": student vs teacher (greedy, " << EG << " games, " << K.game_rounds << " rounds): student wins "
                      << sw << ", teacher wins " << tw << ", mean Ti collected student " << sti / std::max(1, EG)
                      << " teacher " << tti / std::max(1, EG) << std::endl;
        }
        if (it % 5 == 0 || it == iters) {
            for (int p = 0; p < NUM_POLICIES; p++) st->a(p)->sync_shifts_to_buffer();
            torch::save(st, spath);
            Student cpu; torch::load(cpu, spath, torch::Device(torch::kCPU));   // exports from a CPU copy (bit-exact ints)
            for (int p = 0; p < NUM_POLICIES; p++) cpu->a(p)->sync_shifts_from_buffer();
            export_student(cpu, jpath);
        }
        // exactness vectors: quantized input -> integer logits (student)
        if (it == iters) if (const char* vp = std::getenv("VECTORS")) {
            std::ofstream vf(vp);
            torch::NoGradGuard ng;
            int written = 0;
            Student cpu; torch::load(cpu, spath, torch::Device(torch::kCPU));   // vectors from the exported CPU copy
            for (int p = 0; p < NUM_POLICIES; p++) cpu->a(p)->sync_shifts_from_buffer();
            for (auto& s : data) {
                if (written >= env_int("NVECTORS", 300)) break;
                std::vector<float> qv(s.q.begin(), s.q.end());
                auto li = cpu->a(s.policy)->forward(torch::from_blob(qv.data(), {1, (long)qv.size()}, torch::kFloat32).clone()).squeeze(0);
                vf << "{\"policy\":" << s.policy << ",\"raw\":[";
                for (size_t i = 0; i < s.raw.size(); i++) vf << (i ? "," : "") << s.raw[i];
                vf << "],\"logits\":";
                dump_json_int(vf, li);
                vf << "}\n";
                written++;
            }
        }
    }
    std::cout << "DISTILL COMPLETE student=" << spath << " export=" << jpath << std::endl;
}
