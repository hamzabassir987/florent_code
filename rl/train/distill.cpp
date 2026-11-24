// Distil the float teacher into the integer-exact int4 student.
//
//   distill TEACHER.pt OUT_STUDENT.pt OUT.json ITERATIONS MAP [MAP...]
//   env: GAMES (4) ROUNDS (40) EPOCHS (2) LR (3e-3) VECTORS (vector dump path)
//
// Each iteration plays ROUNDS more rounds of GAMES persistent games (teacher
// plays on iteration 1, the student afterwards: DAgger), records every
// decision's raw observation and its participating heads (masks from the
// acting policy's decode), labels each head with the teacher's masked
// distribution, then minimises KL(teacher || student) per head record.
// At every collected decision it also decodes greedily with both models on the
// same state and counts identical actions (float/quantized agreement).
// Writes the student checkpoint, the JSON export for bots/econ/student.py, and
// optionally test vectors (quantized input -> integer logits) for exactness.
#include <torch/torch.h>
#include "entropy_file.hpp"

#include <fstream>
#include <iostream>
#include <random>

#include "../sim/florent_game.hpp"
#include "../sim/florent_heads.hpp"
#include "model.hpp"
#include "student.hpp"

using namespace florent_rl;
static int env_int(const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; }
static float env_float(const char* k, float d) { const char* v = std::getenv(k); return v ? float(std::atof(v)) : d; }

struct Sample {
    int policy;
    std::vector<int16_t> raw;
    std::vector<int32_t> q;
    std::vector<std::pair<int, std::vector<uint8_t>>> heads;   // participating head, mask
};

static torch::Tensor raw_tensor(const std::vector<int16_t>& r) {
    std::vector<float> v(r.begin(), r.end());
    return torch::from_blob(v.data(), {1, (long)v.size()}, torch::kFloat32).clone();
}
static torch::Tensor q_tensor(const std::vector<int32_t>& r) {
    std::vector<float> v(r.begin(), r.end());
    return torch::from_blob(v.data(), {1, (long)v.size()}, torch::kFloat32).clone();
}

static Chooser greedy() {
    return [](int, const float* l, const std::vector<uint8_t>& m, float& lp) { return argmax_choice(l, m, lp); };
}

static void dump_json_int(std::ostream& os, const torch::Tensor& t) {
    auto v = t.detach().to(torch::kFloat64).contiguous().flatten();
    os << "[";
    for (long i = 0; i < v.numel(); i++) os << (i ? "," : "") << (long long)std::llround(v[i].item<double>());
    os << "]";
}

static void export_student(Student& st, const std::string& path) {
    std::ofstream os(path);
    os << "{\"format\":\"florent-student-v1\",\"logit_shift\":" << ST_LOGIT_SHIFT
       << ",\"channels\":" << ST_CH << ",\"hidden\":" << ST_HIDDEN
       << ",\"actor_ch\":" << fe::ACTOR_CH << ",\"scalars\":" << fe::ACTOR_SCALARS << ",\"policies\":[";
    for (int p = 0; p < NUM_POLICIES; p++) {
        auto* a = st->a(p);
        auto q7 = [](const torch::Tensor& w) { return torch::round(torch::clamp(w, -7, 7)); };
        auto q127 = [](const torch::Tensor& w, double lim = 127) { return torch::round(torch::clamp(w, -lim, lim)); };
        os << (p ? "," : "") << "{\"policy\":" << p << ",\"nlogits\":" << a->nlogits << ",\"shifts\":[";
        for (int i = 0; i < ST_NSHIFT; i++) os << (i ? "," : "") << a->shifts[i];
        os << "],\"stem_w\":"; dump_json_int(os, q7(a->stem_w));
        os << ",\"stem_b\":"; dump_json_int(os, q127(a->stem_b));
        for (int b = 0; b < ST_LAYERS; b++) {
            os << ",\"pw_w" << b << "\":"; dump_json_int(os, q7(a->pw_w[b]));
            os << ",\"pw_b" << b << "\":"; dump_json_int(os, q127(a->pw_b[b]));
            os << ",\"dw_w" << b << "\":"; dump_json_int(os, q7(a->dw_w[b]));
            os << ",\"dw_b" << b << "\":"; dump_json_int(os, q127(a->dw_b[b]));
        }
        os << ",\"fc_w\":"; dump_json_int(os, q127(a->fc_w));
        os << ",\"fc_b\":"; dump_json_int(os, q127(a->fc_b));
        os << ",\"out_w\":"; dump_json_int(os, q127(a->out_w));
        os << ",\"out_b\":"; dump_json_int(os, q127(a->out_b, 127 * 16));
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
    Teacher teacher; torch::load(teacher, tpath); teacher->eval();
    Student st;
    bool resumed = false;
    if (const char* init = std::getenv("INIT")) {
        torch::load(st, init);
        for (int p = 0; p < NUM_POLICIES; p++) st->a(p)->sync_shifts_from_buffer();
        resumed = true;
    }
    torch::optim::Adam opt(st->parameters(), torch::optim::AdamOptions(env_float("LR", 3e-3f)));
    // game_rounds / ore_radius from the same entropy file as neural_train (re-read every iteration)
    const std::string KNOB_PATH = std::getenv("ENTROPY_FILE") ? std::getenv("ENTROPY_FILE") : "entropy_file";
    Knobs K{}; K.lr = 0; K.epochs = 1;
    read_entropy_file(KNOB_PATH, K);
    std::mt19937 master(99);
    std::vector<Match> games(G);
    std::vector<std::mt19937> grng(G);
    auto new_game = [&](int i) {
        games[i] = Match();
        games[i].load(maps[master() % maps.size()], master());
        games[i].reset();
        games[i].game.max_turns = K.game_rounds;
        games[i].game.limit_ore(K.ore_radius, master());
        grng[i].seed(master());
    };
    for (int i = 0; i < G; i++) new_game(i);

    for (int it = 1; it <= iters; it++) {
        read_entropy_file(KNOB_PATH, K);
        for (auto& m : games) m.game.max_turns = K.game_rounds;
        bool teacher_plays = it == 1 && !resumed;
        std::vector<Sample> data;
        long agree = 0, compared = 0;
        st->eval();
        for (int gi = 0; gi < G; gi++) {
            Match& m = games[gi];
            for (int r = 0; r < R && !m.game.over; r++) {
                for (int a : m.acting_units()) {
                    if (a >= int(m.game.ents.size()) || !m.game.ents[a].alive) continue;
                    int type = m.game.ents[a].type;
                    if (type == fe::SPIKE) { m.observe(a); Candidate c; if (spike_action(m.game, a, c)) m.act(a, c); continue; }
                    int p = policy_of_type(type);
                    if (p < 0) continue;
                    m.observe(a);                            // memory updates every turn, as deployed
                    std::vector<float> obs; m.encode_only(a, obs);   // first: the conveyor planner mask reads it
                    int8_t plan4[4]; plan4_from_obs(obs.data(), plan4);
                    const int own_l = m.memories.at(a).own_count(fe::LAUNCHER);
                    if (!has_choice(m.game, a, plan4, own_l)) continue;
                    Sample s; s.policy = p; s.raw.assign(obs.begin(), obs.end());
                    s.q.resize(s.raw.size()); quantize_actor(s.raw.data(), s.q.data());
                    torch::Tensor tl, sl;
                    {
                        torch::NoGradGuard ng;
                        tl = teacher->logits(p, raw_tensor(s.raw)).squeeze(0).contiguous();
                        sl = (st->a(p)->forward(q_tensor(s.q)) / double(1 << ST_LOGIT_SHIFT)).squeeze(0).contiguous();
                    }
                    // agreement: greedy decode with each model on this exact state
                    Decision dt = decide(m.game, a, tl.data_ptr<float>(), greedy(), plan4, own_l);
                    Decision ds = decide(m.game, a, sl.data_ptr<float>(), greedy(), plan4, own_l);
                    compared++;
                    bool same = dt.actions.size() == ds.actions.size();
                    for (size_t k = 0; same && k < dt.actions.size(); k++) {
                        auto &x = dt.actions[k], &y = ds.actions[k];
                        same = x.kind == y.kind && x.a == y.a && x.b == y.b && x.c == y.c && x.d == y.d;
                    }
                    agree += same;
                    // act with a sampled decision of the playing model
                    const float* L = teacher_plays ? tl.data_ptr<float>() : sl.data_ptr<float>();
                    std::mt19937& rng = grng[gi];
                    Decision d = decide(m.game, a, L, [&](int, const float* l, const std::vector<uint8_t>& mk, float& lp) {
                        double mx = -1e30; for (size_t i = 0; i < mk.size(); i++) if (mk[i]) mx = std::max(mx, double(l[i]));
                        std::vector<double> w(mk.size(), 0.0); double z = 0;
                        for (size_t i = 0; i < mk.size(); i++) if (mk[i]) { w[i] = std::exp(double(l[i]) - mx); z += w[i]; }
                        double u = std::uniform_real_distribution<double>(0, z)(rng), c = 0; int pick = -1;
                        for (size_t i = 0; i < mk.size(); i++) if (mk[i]) { pick = int(i); c += w[i]; if (u <= c) break; }
                        lp = 0; return pick;
                    }, plan4, own_l);
                    for (auto& hr : d.records) s.heads.push_back({hr.head, hr.mask});
                    // the teacher's own decode visits heads too: label those as well
                    for (auto& hr : dt.records) s.heads.push_back({hr.head, hr.mask});
                    for (auto& c : d.actions) m.act(a, c);
                    data.push_back(std::move(s));
                }
                m.game.end_round();
            }
            if (m.game.over) new_game(gi);
        }
        // one-time shift calibration on the first batch
        if (it == 1 && !resumed) {
            for (int p = 0; p < NUM_POLICIES; p++) {
                std::vector<float> buf; long n = 0;
                for (auto& s : data) if (s.policy == p && n < 512) { buf.insert(buf.end(), s.q.begin(), s.q.end()); n++; }
                if (!n) continue;
                st->a(p)->calibrate(torch::from_blob(buf.data(), {n, (long)data[0].q.size()}, torch::kFloat32).clone());
            }
        }
        // ---- KL distillation per head record
        st->train();
        double kl_sum = 0; long kl_n = 0;
        std::vector<int> order(data.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = int(i);
        for (int ep = 0; ep < EPOCHS; ep++) {
            std::shuffle(order.begin(), order.end(), master);
            for (size_t s0 = 0; s0 < order.size(); s0 += MB) {
                size_t s1 = std::min(order.size(), s0 + MB);
                torch::Tensor loss = torch::zeros({}); long nrec = 0;
                for (int p = 0; p < NUM_POLICIES; p++) {
                    std::vector<int> idx;
                    for (size_t k = s0; k < s1; k++) if (data[order[k]].policy == p) idx.push_back(order[k]);
                    if (idx.empty()) continue;
                    long D = long(data[idx[0]].raw.size());
                    std::vector<float> rb(idx.size() * D), qb(idx.size() * D);
                    for (size_t b = 0; b < idx.size(); b++) {
                        std::copy(data[idx[b]].raw.begin(), data[idx[b]].raw.end(), rb.begin() + b * D);
                        std::copy(data[idx[b]].q.begin(), data[idx[b]].q.end(), qb.begin() + b * D);
                    }
                    auto rt = torch::from_blob(rb.data(), {(long)idx.size(), D}, torch::kFloat32).clone();
                    auto qt = torch::from_blob(qb.data(), {(long)idx.size(), D}, torch::kFloat32).clone();
                    torch::Tensor tl;
                    { torch::NoGradGuard ng; tl = teacher->logits(p, rt); }
                    auto sl = st->a(p)->forward(qt) / double(1 << ST_LOGIT_SHIFT);
                    for (int head : policy_heads(p)) {
                        std::vector<long> rows; std::vector<uint8_t> mk; int n = HEAD_SIZE[head];
                        for (size_t b = 0; b < idx.size(); b++)
                            for (auto& hm : data[idx[b]].heads)
                                if (hm.first == head) { rows.push_back(long(b)); mk.insert(mk.end(), hm.second.begin(), hm.second.end()); }
                        if (rows.empty()) continue;
                        long k = long(rows.size());
                        auto rt2 = torch::tensor(rows, torch::kLong);
                        auto mask = torch::from_blob(mk.data(), {k, n}, torch::kUInt8).clone().to(torch::kBool);
                        auto tlog = torch::log_softmax(tl.index_select(0, rt2).narrow(1, head_offset(p, head), n).masked_fill(~mask, -1e9f), 1);
                        auto slog = torch::log_softmax(sl.index_select(0, rt2).narrow(1, head_offset(p, head), n).masked_fill(~mask, -1e9f), 1);
                        auto kl = (tlog.exp() * (tlog - slog)).masked_fill(~mask, 0.f).sum(1);
                        loss = loss + kl.sum();
                        nrec += k;
                    }
                }
                if (!nrec) continue;
                loss = loss / double(nrec);
                opt.zero_grad(); loss.backward(); opt.step();
                kl_sum += loss.item<double>() * nrec; kl_n += nrec;
            }
        }
        std::cout << "iter " << it << (teacher_plays ? " (teacher plays)" : " (student plays)")
                  << " samples " << data.size() << " kl " << (kl_n ? kl_sum / kl_n : 0)
                  << " action_agreement " << (compared ? double(agree) / compared : 0) << " (" << agree << "/" << compared << ")"
                  << std::endl;
        if (it % 5 == 0 || it == iters) {
            for (int p = 0; p < NUM_POLICIES; p++) st->a(p)->sync_shifts_to_buffer();
            torch::save(st, spath);
            export_student(st, jpath);
        }
        // exactness vectors: quantized input -> integer logits (student)
        if (it == iters) if (const char* vp = std::getenv("VECTORS")) {
            std::ofstream vf(vp);
            torch::NoGradGuard ng;
            int written = 0;
            for (auto& s : data) {
                if (written >= env_int("NVECTORS", 300)) break;
                auto li = st->a(s.policy)->forward(q_tensor(s.q)).squeeze(0);
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
