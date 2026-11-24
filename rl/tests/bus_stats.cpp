// bus_stats N_GAMES ROUNDS MAP...  — how fast the team report bus spreads knowledge.
// Random (masked) play on the real engine with the real memories and bus
// (Match::observe). Per fact (an ore tile; a tile of an enemy core / turret):
// the first round any unit SAW it, then for every other unit alive then, the
// round its memory first holds it (vision, bus or symmetry). Plus late units:
// share of the team's known ore a unit knows k rounds after it appeared.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

#include "../../engine/src/obs.hpp"
#include "florent_game.hpp"
#include "florent_heads.hpp"

using namespace florent_rl;

static double pct(std::vector<int> v, double q) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(q * v.size()))];
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: bus_stats N_GAMES ROUNDS MAP...\n"); return 2; }
    const int NG = std::atoi(argv[1]), R = std::atoi(argv[2]);
    std::vector<std::string> maps(argv + 3, argv + argc);
    std::mt19937_64 rng(7);
    std::vector<int> ore_lat, bld_lat, ore_before;       // before: known before anyone saw it (symmetry)
    std::map<int, std::vector<double>> late;             // k -> share of team-known ore
    long ore_pairs = 0, ore_never = 0, bld_pairs = 0, bld_never = 0;
    for (int gi = 0; gi < NG; gi++) {
        Match m;
        m.load(maps[gi % maps.size()], 1000 + gi);
        m.reset();
        m.game.max_turns = R;
        const fe::Game& G = m.game;
        const int N = G.W * G.H;
        // first round any unit of team t saw tile x (vision): seen_first[t][x]
        std::vector<std::vector<int>> seen_first(2, std::vector<int>(N, -1));
        // per (unit, tile) first round the unit's memory knew the fact
        std::map<int, std::vector<int>> know_ore, know_bld;
        std::map<int, int> born, last;
        for (int r = 0; r < R && !G.over; r++) {
            for (int a : m.acting_units()) {
                if (a >= int(G.ents.size()) || !G.ents[a].alive) continue;
                const int type = G.ents[a].type;
                if (type == fe::SPIKE) { m.observe(a); Candidate c; if (spike_action(m.game, a, c)) m.act(a, c); continue; }
                const int p = policy_of_type(type);
                if (p < 0) continue;
                m.observe(a);
                std::vector<float> obs; m.encode_only(a, obs);
                int8_t plan4[4]; plan4_from_obs(obs.data(), plan4);
                const int own_l = m.memories.at(a).own_count(fe::LAUNCHER);
                if (!has_choice(m.game, a, plan4, own_l)) continue;
                std::vector<float> L(policy_logit_count(p));
                for (auto& x : L) x = std::uniform_real_distribution<float>(-1, 1)(rng);
                Decision d = decide(m.game, a, L.data(), [&](int, const float* l, const std::vector<uint8_t>& mk, float& lp) {
                    std::vector<int> ok; for (size_t i = 0; i < mk.size(); i++) if (mk[i]) ok.push_back(int(i));
                    lp = 0; return ok[rng() % ok.size()];
                }, plan4, own_l);
                for (auto& c : d.actions) m.act(a, c);
            }
            m.game.end_round();
            // knowledge snapshot after the round
            for (auto& [ent, mem] : m.memories) {
                if (ent >= int(G.ents.size()) || !G.ents[ent].alive) continue;
                if (!born.count(ent)) born[ent] = r;
                last[ent] = r;
                const int team = G.ents[ent].team;
                auto& ko = know_ore[ent]; auto& kb = know_bld[ent];
                if (ko.empty()) { ko.assign(N, -1); kb.assign(N, -1); }
                for (int t = 0; t < N; t++) {
                    if (mem.seen_round[t] >= 0 && seen_first[team][t] < 0) seen_first[team][t] = r;
                    const bool ore = G.env[t] == fe::ORE_TI || G.env[t] == fe::ORE_VO;
                    if (ore && ko[t] < 0 && (mem.terrain[t] == fe::T_ORE_TI || mem.terrain[t] == fe::T_ORE_VO)) ko[t] = r;
                    const int b = G.bld[t];
                    const bool enemy_b = b >= 0 && G.ents[b].alive && G.ents[b].team != team &&
                                         (G.ents[b].type == fe::CORE || G.ents[b].type == fe::GUNNER || G.ents[b].type == fe::SENTINEL ||
                                          G.ents[b].type == fe::LAUNCHER || G.ents[b].type == fe::SPIKE);
                    if (enemy_b && kb[t] < 0 && mem.bld_id[t] >= 0 && mem.bld_team[t] != team && mem.bld_type[t] == G.ents[b].type) kb[t] = r;
                }
            }
            // late units: share of the team's known ore, k rounds after birth
            for (auto& [ent, mem] : m.memories) {
                if (ent >= int(G.ents.size()) || !G.ents[ent].alive) continue;
                const int k = r - born[ent], team = G.ents[ent].team;
                if (born[ent] < 5 || (k != 0 && k != 5 && k != 10 && k != 20 && k != 40)) continue;   // units born after round 20
                int team_known = 0, mine = 0;
                for (int t = 0; t < N; t++) {
                    const bool ore = G.env[t] == fe::ORE_TI || G.env[t] == fe::ORE_VO;
                    if (!ore) continue;
                    bool any = false;
                    for (auto& [e2, m2] : m.memories) if (e2 < int(G.ents.size()) && G.ents[e2].alive && G.ents[e2].team == team && e2 != ent &&
                                                           (m2.terrain[t] == fe::T_ORE_TI || m2.terrain[t] == fe::T_ORE_VO)) { any = true; break; }
                    if (!any) continue;
                    team_known++;
                    mine += mem.terrain[t] == fe::T_ORE_TI || mem.terrain[t] == fe::T_ORE_VO;
                }
                if (team_known) late[k].push_back(double(mine) / team_known);
            }
        }
        // latencies: for every unit alive when its team first saw the fact, and that did not see it itself first
        for (auto& [ent, ko] : know_ore) {
            const int team = G.ents[ent].team;
            for (int t = 0; t < N; t++) {
                const bool ore = G.env[t] == fe::ORE_TI || G.env[t] == fe::ORE_VO;
                const int s0 = seen_first[team][t];
                if (!ore || s0 < 0 || born[ent] > s0) continue;
                if (m.memories.at(ent).seen_round[t] == s0) continue;       // the unit saw it itself
                if (last[ent] < s0 + 10) continue;                           // died soon after: not counted
                ore_pairs++;
                if (ko[t] < 0) { ore_never++; continue; }
                if (ko[t] < s0) ore_before.push_back(s0 - ko[t]); else ore_lat.push_back(ko[t] - s0);
            }
        }
        for (auto& [ent, kb] : know_bld) {
            for (int t = 0; t < N; t++) {
                if (kb[t] < 0) {
                    const int b = G.bld[t];
                    if (b >= 0 && G.ents[b].team != G.ents[ent].team && G.ents[b].type == fe::CORE) bld_never++;
                    continue;
                }
                const int s0 = seen_first[G.ents[ent].team][t];
                if (s0 < 0 || born[ent] > s0) continue;
                bld_pairs++;
                bld_lat.push_back(std::max(0, kb[t] - s0));
            }
        }
    }
    std::printf("ore tiles, units that did not see the tile themselves: %ld (unit, tile) pairs\n", ore_pairs);
    std::printf("  known BEFORE any teammate saw it (symmetry): %zu\n", ore_before.size());
    std::printf("  learned after a teammate saw it: %zu   rounds: median %.0f  p75 %.0f  p90 %.0f  max %.0f\n",
                ore_lat.size(), pct(ore_lat, .5), pct(ore_lat, .75), pct(ore_lat, .9), pct(ore_lat, 1.0));
    std::printf("  never learned (unit alive 10+ rounds after it was seen): %ld\n", ore_never);
    std::printf("enemy buildings (core / turrets), units alive when first seen: %ld pairs  rounds: median %.0f  p75 %.0f  p90 %.0f\n",
                bld_pairs, pct(bld_lat, .5), pct(bld_lat, .75), pct(bld_lat, .9));
    std::printf("late units (born after round 5): share of the team's known ore they know\n");
    for (auto& [k, v] : late) {
        double s = 0; for (double x : v) s += x;
        std::printf("  %2d rounds after birth: %5.1f%%  (%zu samples)\n", k, 100 * s / std::max<size_t>(1, v.size()), v.size());
    }
}
