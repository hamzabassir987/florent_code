// Recover a replay seed: seedsearch FIRST_ENTITY_ID SECOND_ENTITY_ID LO HI
// build: g++ -O3 -march=native -pthread -o seedsearch seedsearch.cpp
#include "../src/rustrng.hpp"
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <atomic>
int main(int argc, char** argv) {
    uint32_t id1 = strtoul(argv[1], 0, 10), id2 = strtoul(argv[2], 0, 10);
    uint64_t lo = strtoull(argv[3], 0, 10), hi = strtoull(argv[4], 0, 10);
    int T = std::thread::hardware_concurrency();
    std::atomic<long long> found(-1);
    std::vector<std::thread> th;
    for (int k = 0; k < T; k++) th.emplace_back([&, k] {
        for (uint64_t s = lo + k; s < hi && found < 0; s += T) {
            fe::StdRng r(s ^ 0x1D5EED00C0DEF00Dull);
            if (r.gen_range(3, 1000000000) == id1 && r.gen_range(3, 1000000000) == id2) { found = s; std::printf("SEED %llu\n", (unsigned long long)s); }
        }
    });
    for (auto& t : th) t.join();
    if (found < 0) std::printf("none in [%llu,%llu)\n", (unsigned long long)lo, (unsigned long long)hi);
}
