// entropy_file: live training knobs, re-read at the start of every iteration
// (edit it while training runs). Named fields, value on the same line or the
// line below; '#' starts a comment; ':' '=' ',' count as spaces:
//
//     builder_mode 0.008     # target entropy coefficient of one head
//     lr 0.0003              # overrides the 3e-5 -> 1e-5 schedule while present
//     epochs 1               # PPO passes over each rollout (1-8); with more than
//                            # one, a minibatch with KL(old || new) > 0.03 stops the updates
//     game_rounds 200        # games end at this round (official 1000)
//     ore_radius 2           # all ore moved to random tiles within this distance
//                            # of each core, mirrored (same Ti/Vo counts as the
//                            # map); -1: map as is. Applies to games started afterwards
//     eval_every 50          # every N iterations: eval vs the frozen net (0: off)
//     eval_games 200         # games per eval (half on each side)
//
// No file: the current values are kept. A deleted line keeps its last value
// (except lr, which goes back to the schedule).
#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "../sim/florent_heads.hpp"

namespace florent_rl {

inline const char* HEAD_NAMES[NUM_HEAD_IDS] = {
    "core_spawn", "core_convert",
    "builder_mode", "builder_move", "builder_build_type", "builder_build_tile", "builder_build_dir",
    "builder_attack", "builder_heal", "builder_destroy",
    "gunner_mode", "gunner_fire", "gunner_rotate",
    "sentinel_mode", "sentinel_fire",
    "launcher_mode", "launcher_bot", "launcher_dest"};

struct Knobs {
    float head_entropy[NUM_HEAD_IDS];
    double lr;
    int epochs;
    int game_rounds = 1000;
    int ore_radius = -1;
    int eval_every = 50, eval_games = 200;
};

// Returns false if the file does not exist.
inline bool read_entropy_file(const std::string& path, Knobs& k) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line, pending;
    auto set = [&](const std::string& key, double v) {
        if (key == "lr") { k.lr = v; return; }
        if (key == "epochs") { k.epochs = std::clamp(int(v), 1, 8); return; }
        if (key == "game_rounds") { k.game_rounds = std::clamp(int(v), 1, 1000); return; }
        if (key == "ore_radius") { k.ore_radius = int(v); return; }
        if (key == "eval_every") { k.eval_every = std::max(0, int(v)); return; }
        if (key == "eval_games") { k.eval_games = std::max(2, int(v)); return; }
        for (int h = 0; h < NUM_HEAD_IDS; h++)
            if (key == HEAD_NAMES[h]) { k.head_entropy[h] = float(v); return; }
        std::cerr << "entropy_file: unknown field '" << key << "', ignored\n";
    };
    while (std::getline(f, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        for (char& c : line) if (c == '=' || c == ',' || c == ':') c = ' ';
        std::istringstream ls(line);
        std::string tok;
        while (ls >> tok) {
            std::istringstream num(tok);
            double v;
            if (num >> v && num.eof()) {
                if (pending.empty()) std::cerr << "entropy_file: value " << tok << " without a name, ignored\n";
                else { set(pending, v); pending.clear(); }
            } else {
                if (!pending.empty()) std::cerr << "entropy_file: '" << pending << "' has no value, ignored\n";
                pending = tok;
            }
        }
    }
    if (!pending.empty()) std::cerr << "entropy_file: '" << pending << "' has no value, ignored\n";
    return true;
}

}  // namespace florent_rl
