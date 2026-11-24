// map_gen_dump OUT_DIR N [CORE_GAP] [PIT_SHARE] [SEED]
// Writes N generated maps (gen_000.map26 ...), checks each loads in the
// engine, and prints one line per map.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "../../engine/src/fengine.hpp"
#include "map_gen.hpp"

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: map_gen_dump OUT_DIR N [CORE_GAP] [PIT_SHARE] [SEED]\n"; return 2; }
    std::filesystem::create_directories(argv[1]);
    florent_rl::MapGenParams P;
    if (argc > 3) P.core_gap = std::atoi(argv[3]);
    if (argc > 4) P.pit_share = float(std::atof(argv[4]));
    std::mt19937_64 rng(argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 1);
    int bad = 0;
    for (int i = 0; i < std::atoi(argv[2]); i++) {
        florent_rl::MapGenInfo info;
        const std::string b = florent_rl::generate_map(P, rng, &info);
        char name[32]; std::snprintf(name, sizeof name, "gen_%03d.map26", i);
        const auto path = std::filesystem::path(argv[1]) / name;
        std::ofstream(path, std::ios::binary) << b;
        fe::Game g;
        const bool ok = g.load_map_bytes(reinterpret_cast<const uint8_t*>(b.data()), b.size());
        bad += !ok;
        std::cout << name << "  " << info.line << (ok ? "" : "  LOAD FAILED") << "\n";
    }
    std::cout << (bad ? "MAP GEN: LOAD FAILURES\n" : "MAP GEN OK\n");
    return bad ? 1 : 0;
}
