// C API for the factorized heads (used by rl/tests/heads_parity.py). Linked
// into librlcapi.so together with the engine's own C API.
#include "florent_heads.hpp"
using namespace florent_rl;
extern "C" {
// plan4: 4 planned conveyor directions (N E S W, Dir code, 0 none), or NULL
// own_launchers: launcher-cap count (< 0: no cap)
// targets: ncore < 0: none; else core tiles, then nbld building tiles (y * W + x), own_turrets
static Targets make_tg(const int* tiles, int ncore, int nbld, int own_turrets) {
    Targets t; for (int i = 0; i < ncore; i++) t.core.push_back(tiles[i]);
    for (int i = 0; i < nbld; i++) t.bld.push_back(tiles[ncore + i]);
    t.own_turrets = own_turrets; return t;
}
int rl_has_choice(fe::Game* g, int a, const int* plan4, int own_launchers, const int* tiles, int ncore, int nbld, int own_turrets) {
    int8_t p[4]; if (plan4) for (int i = 0; i < 4; i++) p[i] = int8_t(plan4[i]);
    Targets t = make_tg(tiles, ncore, nbld, own_turrets);
    return has_choice(*g, a, plan4 ? p : nullptr, own_launchers, ncore < 0 ? nullptr : &t) ? 1 : 0;
}
void rl_pit(fe::Game* g, int* out3) { out3[0] = g->pit_axis; out3[1] = g->pit_at; out3[2] = g->pit_w; }
int rl_policy_of_type(int t) { return policy_of_type(t); }
int rl_policy_logits(int p) { return policy_logit_count(p); }
// Argmax decode. out layout: n_actions, then 5 ints per action (kind,a,b,c,d),
// n_records, then per record: head, choice, mask_len, mask...
int rl_decide(fe::Game* g, int a, const float* logits, int* out, const int* plan4, int own_launchers,
              const int* tiles, int ncore, int nbld, int own_turrets) {
    int8_t p[4]; if (plan4) for (int i = 0; i < 4; i++) p[i] = int8_t(plan4[i]);
    Targets t = make_tg(tiles, ncore, nbld, own_turrets);
    Decision d = decide(*g, a, logits, [](int, const float* l, const std::vector<uint8_t>& m, float& lp) {
        return argmax_choice(l, m, lp);
    }, plan4 ? p : nullptr, own_launchers, ncore < 0 ? nullptr : &t);
    int k = 0;
    out[k++] = int(d.actions.size());
    for (auto& c : d.actions) { out[k++] = int(c.kind); out[k++] = c.a; out[k++] = c.b; out[k++] = c.c; out[k++] = c.d; }
    out[k++] = int(d.records.size());
    for (auto& r : d.records) {
        out[k++] = r.head; out[k++] = r.choice; out[k++] = int(r.mask.size());
        for (auto v : r.mask) out[k++] = v;
    }
    return k;
}
int rl_spike(fe::Game* g, int a, int* xy, int first) { Candidate c; if (!spike_action(*g, a, c, first != 0)) return 0; xy[0] = c.a; xy[1] = c.b; return 1; }
int rl_apply(fe::Game* g, int a, int kind, int p, int q, int r, int s) { Candidate c{ActionKind(kind), p, q, r, s}; return apply_candidate(*g, a, c); }
}
