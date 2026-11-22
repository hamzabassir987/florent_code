"""Bit-exact Python port of Rust rand 0.8 StdRng (rand_chacha 0.3 ChaCha12Rng)."""
M32 = 0xFFFFFFFF


def _rotl(v, c):
    return ((v << c) & M32) | (v >> (32 - c))


def _block(key, counter, rounds=12):
    s = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574] + key + [counter & M32, counter >> 32, 0, 0]
    x = s[:]
    for _ in range(rounds // 2):
        for a, b, c, d in ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15),
                           (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)):
            x[a] = (x[a] + x[b]) & M32; x[d] = _rotl(x[d] ^ x[a], 16)
            x[c] = (x[c] + x[d]) & M32; x[b] = _rotl(x[b] ^ x[c], 12)
            x[a] = (x[a] + x[b]) & M32; x[d] = _rotl(x[d] ^ x[a], 8)
            x[c] = (x[c] + x[d]) & M32; x[b] = _rotl(x[b] ^ x[c], 7)
    return [(x[i] + s[i]) & M32 for i in range(16)]


class StdRng:
    def __init__(self, seed_u64):
        state = seed_u64 & 0xFFFFFFFFFFFFFFFF
        key = []
        for _ in range(8):  # rand_core 0.6 seed_from_u64 (PCG32)
            state = (state * 6364136223846793005 + 11634580027462260723) & 0xFFFFFFFFFFFFFFFF
            xs = (((state >> 18) ^ state) >> 27) & M32
            rot = state >> 59
            key.append(((xs >> rot) | (xs << ((32 - rot) & 31))) & M32)
        self.key = key
        self.counter = 0
        self.buf = []
        self.idx = 64

    def _refill(self):
        self.buf = []
        for k in range(4):
            self.buf += _block(self.key, self.counter + k)
        self.counter += 4
        self.idx = 0

    def next_u32(self):
        if self.idx >= 64:
            self._refill()
        v = self.buf[self.idx]; self.idx += 1
        return v

    def next_u64(self):
        if self.idx < 63:
            lo, hi = self.buf[self.idx], self.buf[self.idx + 1]; self.idx += 2
        elif self.idx >= 64:
            self._refill(); lo, hi = self.buf[0], self.buf[1]; self.idx = 2
        else:
            lo = self.buf[63]; self._refill(); hi = self.buf[0]; self.idx = 1
        return (hi << 32) | lo

    def f64(self):  # Standard distribution for f64 in [0,1)
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def gen_range_u32(self, low, high):  # [low, high), rand 0.8 UniformInt::sample_single
        rng = (high - low) & M32
        zone = ((rng << (32 - rng.bit_length())) & M32) - 1 if rng else M32  # (range << lz) - 1
        lz = 32 - rng.bit_length()
        zone = ((rng << lz) & M32) - 1
        while True:
            v = self.next_u32()
            m = v * rng
            hi, lo = m >> 32, m & M32
            if lo <= zone:
                return low + hi
