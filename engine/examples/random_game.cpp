// Random self-play games: a smoke test of the engine and a speed benchmark.
//   random_game MAP.map26 [games]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "../src/fengine.hpp"
using namespace fe;

static uint64_t rs = 88172645463325252ull;
static int rnd(int n) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return int(rs % uint64_t(n)); }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s map [games]\n", argv[0]); return 1; }
    int games = argc > 2 ? std::atoi(argv[2]) : 10;
    Game g;
    if (!g.load_map(argv[1])) { std::fprintf(stderr, "bad map\n"); return 1; }
    long actions = 0, turns = 0;
    int wins[2] = {0, 0};
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < games; k++) {
        g.reset(k + 1);
        while (!g.over) {
            for (int a : g.turn_order()) {
                if (!g.ents[a].alive) continue;
                const Entity& e = g.ents[a];
                turns++;
                if (e.type == CORE) {
                    int d = 1 + rnd(8);
                    int x = e.x + (DX[d] > 0 ? 2 : DX[d]), y = e.y + (DY[d] > 0 ? 2 : DY[d]);
                    if (!g.spawn(a, x, y)) actions++;
                    if (g.teams[e.team].vo > 0) g.convert_ammo(a, g.teams[e.team].vo);
                } else if (e.type == BUILDER) {
                    int r = rnd(10), d = 1 + 2 * rnd(4);
                    int x = e.x + DX[d], y = e.y + DY[d];
                    int ok = -1;
                    if (r < 5) ok = g.move(a, d);
                    else if (r < 7) ok = g.build(a, g.inb(x, y) && (g.env[g.idx(x, y)] == ORE_TI || g.env[g.idx(x, y)] == ORE_VO) ? HARVESTER : CONVEYOR, x, y, 1 + 2 * rnd(4));
                    else if (r < 8) ok = g.build(a, GUNNER + rnd(4), x, y, 1 + rnd(8));
                    else if (r < 9) ok = g.attack(a, x, y);
                    else ok = g.heal(a, x, y);
                    if (ok == 0) actions++;
                } else if (e.type == GUNNER || e.type == SENTINEL || e.type == SPIKE) {
                    int d = 1 + rnd(8);
                    if (!g.fire(a, e.x + DX[d] * (1 + rnd(3)), e.y + DY[d] * (1 + rnd(3)))) actions++;
                }
            }
            g.end_round();
        }
        wins[g.winner]++;
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%d games, %.3f s (%.1f games/s), %ld unit turns (%.2f M/s), %ld actions, wins A %d B %d\n",
                games, s, games / s, turns, turns / s / 1e6, actions, wins[0], wins[1]);
}
