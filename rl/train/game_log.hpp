// Greedy self-play game logs -> official .replay26 (rl/tools/make_replay.py).
//
// The native engine is exact, so a game is fully described by its map, its
// seed and every unit's actions. play_greedy() plays one argmax self-play game
// and writes:
//   NAME.map26  the map as played (curriculum ore placement included)
//   NAME.game   header + one line per action: "round entity_id kind a b c d"
// make_replay.py then runs the official `fcode run` with bots/playback, which
// re-issues the same actions, and gets the official replay.
#pragma once
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "../../engine/src/fengine.hpp"
#include "../sim/florent_game.hpp"
#include "../sim/florent_heads.hpp"

namespace florent_rl {

namespace pbw {
inline void var(std::string& o, uint64_t v) { while (v >= 0x80) { o.push_back(char(v | 0x80)); v >>= 7; } o.push_back(char(v)); }
inline bool rd(const std::string& b, size_t& i, uint64_t& v) {
    v = 0; int s = 0;
    while (i < b.size()) { uint8_t c = uint8_t(b[i++]); v |= uint64_t(c & 0x7F) << s; s += 7; if (!(c & 0x80)) return true; }
    return false;
}
}  // namespace pbw

// The source map with its rows (field 3) replaced by the game's terrain;
// every other field is copied byte for byte.
inline bool write_map(const fe::Game& g, const std::string& src, const std::string& out) {
    std::ifstream f(src, std::ios::binary);
    if (!f) return false;
    std::string b((std::istreambuf_iterator<char>(f)), {}), o;
    bool rows_done = false;
    size_t i = 0;
    while (i < b.size()) {
        size_t start = i; uint64_t key, v;
        if (!pbw::rd(b, i, key)) return false;
        int wt = int(key & 7), fld = int(key >> 3);
        if (wt == 0) pbw::rd(b, i, v);
        else if (wt == 2) { pbw::rd(b, i, v); i += v; }
        else if (wt == 5) i += 4;
        else if (wt == 1) i += 8;
        else return false;
        if (fld != 3) { o.append(b, start, i - start); continue; }
        if (rows_done) continue;
        rows_done = true;
        for (int y = 0; y < g.H; y++) {
            std::string packed, row;
            for (int x = 0; x < g.W; x++) pbw::var(packed, g.env[g.idx(x, y)]);
            pbw::var(row, (1 << 3) | 2); pbw::var(row, packed.size()); row += packed;
            pbw::var(o, (3 << 3) | 2); pbw::var(o, row.size()); o += row;
        }
    }
    std::ofstream w(out, std::ios::binary);
    w << o;
    return bool(w);
}

struct ActionLine { int round, id; Candidate c; };

// Plays one greedy (argmax) self-play game to the end. `logits(p, obs)` must
// return the policy's logits (policy_logit_count(p) floats).
template <class LogitsFn>
inline void play_greedy(Match& m, LogitsFn logits, std::vector<ActionLine>& log) {
    while (!m.game.over) {
        for (int a : m.acting_units()) {
            if (a >= int(m.game.ents.size()) || !m.game.ents[a].alive) continue;
            const int type = m.game.ents[a].type, id = m.game.ents[a].id;
            if (type == fe::SPIKE) {
                const bool first = !m.memories.count(a);   // its first turn (no memory yet)
                m.observe(a);                          // its memory + the team report bus
                Candidate c;
                if (spike_action(m.game, a, c, first) && m.act(a, c) == fe::OK) log.push_back({m.game.round, id, c});
                continue;
            }
            int p = policy_of_type(type);
            if (p < 0) continue;
            m.observe(a);
            std::vector<float> obs; m.encode_only(a, obs);       // first: the conveyor planner mask reads it
            int8_t plan4[4]; plan4_from_obs(obs.data(), plan4);
            const int own_l = m.memories.at(a).own_count(fe::LAUNCHER);
            Targets tg; targets_from_memory(m.memories.at(a), tg);
            if (!has_choice(m.game, a, plan4, own_l, &tg)) continue;
            std::vector<float> L = logits(p, obs);
            Decision d = decide(m.game, a, L.data(), [](int, const float* l, const std::vector<uint8_t>& mk, float& lp) {
                return argmax_choice(l, mk, lp);
            }, plan4, own_l, &tg);
            for (auto& c : d.actions)
                if (m.act(a, c) == fe::OK) log.push_back({m.game.round, id, c});
        }
        m.game.end_round();
    }
}

inline bool write_game(const std::string& path, const std::string& map_file, uint64_t seed,
                       const fe::Game& g, const std::vector<ActionLine>& log) {
    std::ofstream w(path);
    w << "florent-game-v1\n"
      << "map " << map_file << "\n"
      << "seed " << seed << "\n"
      << "rounds " << g.round << "\n"
      << "max_turns " << g.max_turns << "\n"
      << "winner " << g.winner << " " << g.win_condition << "\n"
      << "actions " << log.size() << "\n";
    for (auto& l : log)
        w << l.round << " " << l.id << " " << int(l.c.kind) << " " << l.c.a << " " << l.c.b << " " << l.c.c << " " << l.c.d << "\n";
    return bool(w);
}

}  // namespace florent_rl
