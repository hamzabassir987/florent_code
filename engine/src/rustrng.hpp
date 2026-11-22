// Bit-exact port of Rust rand 0.8 StdRng (rand_chacha 0.3 ChaCha12Rng), as used
// by the official engine: seed_from_u64 (PCG32 key expansion), a 64-word
// block buffer, next_u32 / next_u64 / f64 / gen_range.
#pragma once
#include <cstdint>

namespace fe {

class StdRng {
public:
    StdRng() { seed(0); }
    explicit StdRng(uint64_t s) { seed(s); }
    void seed(uint64_t state) {
        for (int i = 0; i < 8; i++) {
            state = state * 6364136223846793005ull + 11634580027462260723ull;
            uint32_t xs = uint32_t(((state >> 18) ^ state) >> 27);
            uint32_t rot = uint32_t(state >> 59);
            key[i] = (xs >> rot) | (xs << ((32 - rot) & 31));
        }
        counter = 0; idx = 64;
    }
    uint32_t next_u32() {
        if (idx >= 64) refill();
        return buf[idx++];
    }
    uint64_t next_u64() {
        uint32_t lo, hi;
        if (idx < 63) { lo = buf[idx]; hi = buf[idx + 1]; idx += 2; }
        else if (idx >= 64) { refill(); lo = buf[0]; hi = buf[1]; idx = 2; }
        else { lo = buf[63]; refill(); hi = buf[0]; idx = 1; }
        return (uint64_t(hi) << 32) | lo;
    }
    double f64() { return double(next_u64() >> 11) * (1.0 / 9007199254740992.0); }
    // [low, high), rand 0.8 UniformInt<u32>::sample_single
    uint32_t gen_range(uint32_t low, uint32_t high) {
        uint32_t range = high - low;
        int lz = __builtin_clz(range);
        uint32_t zone = (range << lz) - 1;
        for (;;) {
            uint64_t m = uint64_t(next_u32()) * range;
            if (uint32_t(m) <= zone) return low + uint32_t(m >> 32);
        }
    }

private:
    uint32_t key[8];
    uint64_t counter = 0;
    uint32_t buf[64];
    int idx = 64;
    static uint32_t rotl(uint32_t v, int c) { return (v << c) | (v >> (32 - c)); }
    void block(uint64_t ctr, uint32_t* out) {
        uint32_t s[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
                          key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
                          uint32_t(ctr), uint32_t(ctr >> 32), 0, 0};
        uint32_t x[16];
        for (int i = 0; i < 16; i++) x[i] = s[i];
        for (int r = 0; r < 6; r++) {
#define QR(a, b, c, d) x[a] += x[b]; x[d] = rotl(x[d] ^ x[a], 16); x[c] += x[d]; x[b] = rotl(x[b] ^ x[c], 12); \
                       x[a] += x[b]; x[d] = rotl(x[d] ^ x[a], 8); x[c] += x[d]; x[b] = rotl(x[b] ^ x[c], 7);
            QR(0, 4, 8, 12) QR(1, 5, 9, 13) QR(2, 6, 10, 14) QR(3, 7, 11, 15)
            QR(0, 5, 10, 15) QR(1, 6, 11, 12) QR(2, 7, 8, 13) QR(3, 4, 9, 14)
#undef QR
        }
        for (int i = 0; i < 16; i++) out[i] = x[i] + s[i];
    }
    void refill() {
        for (int k = 0; k < 4; k++) block(counter + k, buf + 16 * k);
        counter += 4; idx = 0;
    }
};

}  // namespace fe
