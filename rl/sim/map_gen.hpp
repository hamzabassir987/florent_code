// Training map generator: random maps in the official .map26 format, shaped
// like the real ones (maps/*.map26), with the core distance under control
// (a curriculum knob: close cores teach fighting early).
//
//  * size 16..30 x 17..29; symmetry 0 = 180 rotation, 1 = mirror x (left /
//    right), 2 = mirror y (top / bottom), as the file's field 5;
//  * walls in small mirrored clusters (wall_lo..wall_hi of the tiles);
//  * ore in mirrored clusters: Titanium 8..36, Voltium 0..6 (even counts);
//  * a pit (with probability pit_share, mirror maps only): a void band 1..4
//    tiles wide through the middle, on the mirror axis, as the real pit maps;
//  * cores: core B is core A's mirror; their footprints core_gap tiles apart
//    (Chebyshev, -1: random 7..21 like the real maps). On a pit map the cores
//    face each other across the pit (the gap is at least the pit's width);
//  * checked by flood fill: each core walks to most of its own half, and the
//    cores reach each other when there is no pit. A failed map is redrawn.
#pragma once
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace florent_rl {

struct MapGenParams {
    int core_gap = -1;           // Chebyshev tiles between the two cores' footprints; -1 random 7..21
    float pit_share = 0.15f;     // share of maps with a pit
    float wall_lo = 0.01f, wall_hi = 0.13f;
};

struct MapGenInfo { int W = 0, H = 0, sym = 0, gap = 0, ti = 0, vo = 0, walls = 0; bool pit = false; std::string line; };

// The generated map as .map26 bytes (load with fe::Game::load_map_bytes).
// `info` (optional) gets what was generated.
std::string generate_map(const MapGenParams& P, std::mt19937_64& rng, MapGenInfo* info = nullptr);

}  // namespace florent_rl
