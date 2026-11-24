// Shared device building blocks of the fused actor trunk (trunk_kernels.cu)
// and critic (critic_kernels.cu): 32-channel layers on square fp32 maps in
// shared memory, [channel][y][x] layout. All threads of the CTA call them;
// the caller synchronizes between layers.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

namespace fl {

constexpr int C = 32;
__host__ __device__ constexpr int sq(int s) { return s * s; }

// one residual block in a flat parameter vector: 1x1 weights transposed and
// as stored, bias, depthwise weights, bias, then the same for the second pair
constexpr int B_WPT = 0, B_WP = 1024, B_BP = 2048, B_WD = 2080, B_BD = 2368, B_WP2T = 2400, B_WP2 = 3424, B_BP2 = 4448,
              B_WD2 = 4480, B_BD2 = 4768, BLK = 4800;
constexpr int WPW = C * C + C;                    // 1056: a 1x1 layer's weights + bias
constexpr int WDW = 9 * C + C;                    // 320: a depthwise layer's weights + bias

// ---------------------------------------------------------------- layers
// All threads of the CTA call these; the caller synchronizes.

// out[o][p] = b[o] + sum_i WT[i][o] * in[i][p] (ReLU if RELU), P pixels, NI
// inputs. WT is [NI][32]. Thread item = (2 pixels, 16 outputs): pixels pp and
// pp + ceil(P/2), so a warp reads consecutive addresses; 32 accumulators.
// MASK: out[o][p] = 0 unless the value already there is > 0 (in-place mask).
// RESC: also add r[o][(y-2)*(S-4) + x-2] inside the centre crop (S = side).
template <typename T, bool RELU, bool MASK, bool RESC>
__device__ __forceinline__ void pw(const T* __restrict__ in, int S, int NI, const float* __restrict__ WT,
                                   const float* __restrict__ b, float* out, const float* r = nullptr) {
    const int P = S * S, Ph = (P + 1) >> 1, items = Ph * 2;
    for (int it = threadIdx.x; it < items; it += blockDim.x) {
        const int pp = it % Ph, og = it / Ph;
        const int p0 = pp, p1 = pp + Ph;
        const bool has1 = p1 < P;
        float a0[16], a1[16];
        if (b) {
#pragma unroll
            for (int k = 0; k < 16; k++) a0[k] = a1[k] = b[og * 16 + k];
        } else {
#pragma unroll
            for (int k = 0; k < 16; k++) a0[k] = a1[k] = 0.f;
        }
        const float4* w4 = reinterpret_cast<const float4*>(WT) + og * 4;
        const T* i0 = in + p0;
        const T* i1 = in + (has1 ? p1 : p0);
#pragma unroll 2
        for (int i = 0; i < NI; i++) {
            const float v0 = float(i0[i * P]), v1 = float(i1[i * P]);
            float w[16];
            *reinterpret_cast<float4*>(w + 0) = w4[i * 8 + 0];
            *reinterpret_cast<float4*>(w + 4) = w4[i * 8 + 1];
            *reinterpret_cast<float4*>(w + 8) = w4[i * 8 + 2];
            *reinterpret_cast<float4*>(w + 12) = w4[i * 8 + 3];
#pragma unroll
            for (int k = 0; k < 16; k++) { a0[k] = fmaf(w[k], v0, a0[k]); a1[k] = fmaf(w[k], v1, a1[k]); }
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int p = h ? p1 : p0;
            if (h && !has1) break;
            const float* acc = h ? a1 : a0;
            bool in_crop = false; int q = 0;
            if (RESC) {
                const int y = p / S, x = p % S, so = S - 4;
                in_crop = y >= 2 && y < S - 2 && x >= 2 && x < S - 2;
                q = (y - 2) * so + (x - 2);
            }
#pragma unroll
            for (int k = 0; k < 16; k++) {
                const int o = og * 16 + k;
                float v = acc[k];
                if (RESC && in_crop) v += r[o * sq(S - 4) + q];
                if (RELU) v = fmaxf(v, 0.f);
                if (MASK) v = out[o * P + p] > 0.f ? v : 0.f;
                out[o * P + p] = v;
            }
        }
    }
}

// Valid depthwise 3x3: out[c][y][x] = b[c] + sum_t w[c][t] in[c][y+ty][x+tx]
// (+ r[c][y][x], the block input's centre crop, if r), then ReLU.
__device__ __forceinline__ void dw(const float* __restrict__ in, int Sin, const float* __restrict__ w,
                                   const float* __restrict__ b, float* out, const float* r = nullptr) {
    const int So = Sin - 2, Po = So * So, Pin = Sin * Sin, items = C * Po;
    for (int it = threadIdx.x; it < items; it += blockDim.x) {
        const int c = it / Po, p = it % Po, y = p / So, x = p % So;
        const float* ic = in + c * Pin + y * Sin + x;
        const float* wc = w + c * 9;
        float acc = b[c];
        acc = fmaf(wc[0], ic[0], acc); acc = fmaf(wc[1], ic[1], acc); acc = fmaf(wc[2], ic[2], acc);
        acc = fmaf(wc[3], ic[Sin], acc); acc = fmaf(wc[4], ic[Sin + 1], acc); acc = fmaf(wc[5], ic[Sin + 2], acc);
        acc = fmaf(wc[6], ic[2 * Sin], acc); acc = fmaf(wc[7], ic[2 * Sin + 1], acc); acc = fmaf(wc[8], ic[2 * Sin + 2], acc);
        if (r) acc += r[c * Po + p];
        out[c * Po + p] = fmaxf(acc, 0.f);
    }
}

// Input gradient of the valid depthwise conv, masked in place by the
// activation already in `out` (> 0): out[c][Y][X] = [out > 0] * sum_t
// w[c][t] g[c][Y-ty][X-tx] over the taps that land inside g (side Sin-2).
__device__ __forceinline__ void dw_t_mask(const float* __restrict__ g, int Sin, const float* __restrict__ w, float* out) {
    const int So = Sin - 2, Po = So * So, Pin = Sin * Sin, items = C * Pin;
    for (int it = threadIdx.x; it < items; it += blockDim.x) {
        const int c = it / Pin, p = it % Pin, Y = p / Sin, X = p % Sin;
        if (!(out[c * Pin + p] > 0.f)) { out[c * Pin + p] = 0.f; continue; }
        const float* gc = g + c * Po;
        const float* wc = w + c * 9;
        float acc = 0.f;
#pragma unroll
        for (int ty = 0; ty < 3; ty++) {
            const int y = Y - ty;
            if (y < 0 || y >= So) continue;
#pragma unroll
            for (int tx = 0; tx < 3; tx++) {
                const int x = X - tx;
                if (x < 0 || x >= So) continue;
                acc = fmaf(wc[ty * 3 + tx], gc[y * So + x], acc);
            }
        }
        out[c * Pin + p] = acc;
    }
}

// Weight gradients. One entry per thread, lane = output channel, so a warp
// reads 32 channel rows that lie P floats apart. With P odd (the actor's
// maps) those are 32 different banks; with P even (the critic's) each lane
// starts its pixel loop at its own offset (rotation by the lane) so they are
// again different banks.

// 1x1 layer: gW[o*so + i*si] += sum_p g[o][p] in[i][p] (i < NI), gb[o] +=
// sum_p g[o][p] if gb. Each warp takes groups of 4 inputs.
template <typename T>
__device__ __forceinline__ void pw_dw(const float* __restrict__ g, const T* __restrict__ in, int P, int NI,
                                      float* gW, int so, int si, float* gb) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int rot = (P & 1) ? 0 : lane;
    const float* go = g + lane * P;
    for (int i0 = warp * 4; i0 < NI; i0 += nw * 4) {
        const int ni = NI - i0 < 4 ? NI - i0 : 4;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0, ab = 0;
        const T* i_0 = in + i0 * P;
        const T* i_1 = in + (ni > 1 ? i0 + 1 : i0) * P;
        const T* i_2 = in + (ni > 2 ? i0 + 2 : i0) * P;
        const T* i_3 = in + (ni > 3 ? i0 + 3 : i0) * P;
        int p = rot;
        for (int q = 0; q < P; q++) {
            const float v = go[p];
            a0 = fmaf(v, float(i_0[p]), a0); a1 = fmaf(v, float(i_1[p]), a1);
            a2 = fmaf(v, float(i_2[p]), a2); a3 = fmaf(v, float(i_3[p]), a3);
            ab += v;
            if (++p == P) p = 0;
        }
        float* w = gW + lane * so + i0 * si;
        w[0] += a0;
        if (ni > 1) w[si] += a1;
        if (ni > 2) w[2 * si] += a2;
        if (ni > 3) w[3 * si] += a3;
        if (gb && i0 == 0) gb[lane] += ab;
    }
}

// Valid depthwise conv: gw[c][t] += sum_p g[c][p] in[c][p + t], gb[c] +=
// sum_p g[c][p]. Item = (tap, channel), lane = channel; t == 9: the bias.
__device__ __forceinline__ void dw_dw(const float* __restrict__ g, const float* __restrict__ in, int Sin, float* gw, float* gb) {
    const int So = Sin - 2, Po = So * So, Pin = Sin * Sin;
    for (int it = threadIdx.x; it < 10 * C; it += blockDim.x) {
        const int c = it & 31, t = it >> 5;
        const float* gc = g + c * Po;
        const int rot = (Po & 1) ? 0 : c;
        float acc = 0.f;
        if (t == 9) {
            int p = rot;
            for (int q = 0; q < Po; q++) { acc += gc[p]; if (++p == Po) p = 0; }
            gb[c] += acc;
        } else {
            const float* ic = in + c * Pin + (t / 3) * Sin + (t % 3);
            int y = rot / So, x = rot % So;
            for (int q = 0; q < Po; q++) {
                acc = fmaf(gc[y * So + x], ic[y * Sin + x], acc);
                if (++x == So) { x = 0; if (++y == So) y = 0; }
            }
            gw[c * 9 + t] += acc;
        }
    }
}

// (no __restrict__ on `from`: the scratch is written by this kernel, and a
// const __restrict__ pointer may be read through the non-coherent cache)
__device__ __forceinline__ void copy(const float* from, float* to, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) to[i] = from[i];
}
__device__ __forceinline__ void crop(const float* __restrict__ from, int S, float* __restrict__ to) {   // centre (S-4)^2
    const int so = S - 4, n = C * so * so;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int c = i / (so * so), q = i % (so * so);
        to[i] = from[c * S * S + (q / so + 2) * S + (q % so + 2)];
    }
}

// Weights to shared memory: `n` floats (a multiple of 4) from w (and nb from
// b, appended). Two steps so the loads overlap a layer's compute: fetch()
// at the start of a phase, put() at its end (before the barrier).
struct Pre {
    float4 v[2];
    __device__ __forceinline__ void fetch(const float* w, int n, const float* b, int nb) {
        const int t = threadIdx.x;
#pragma unroll
        for (int k = 0; k < 2; k++) {
            const int j = (t + k * int(blockDim.x)) * 4;
            if (j < n) v[k] = __ldg(reinterpret_cast<const float4*>(w + j));
            else if (j < n + nb) v[k] = __ldg(reinterpret_cast<const float4*>(b + (j - n)));
        }
    }
    __device__ __forceinline__ void put(float* to, int n) const {
        const int t = threadIdx.x;
#pragma unroll
        for (int k = 0; k < 2; k++) {
            const int j = (t + k * int(blockDim.x)) * 4;
            if (j < n) *reinterpret_cast<float4*>(to + j) = v[k];
        }
    }
};
__device__ __forceinline__ void stage(const float* w, int n, const float* b, int nb, float* to) {
    Pre q; q.fetch(w, n, b, nb); q.put(to, n + nb);
}


}  // namespace fl
