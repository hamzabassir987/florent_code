// Fused CUDA kernels for the actor trunk (see model.hpp, ActorImpl):
//
//   stem 1x1 (TRUNK_IN_CH -> 32) + ReLU on the 17x17 patch, then 4 residual blocks
//   [1x1 -> ReLU -> depthwise 3x3 -> ReLU -> 1x1 -> ReLU -> depthwise 3x3
//   -> + input -> ReLU], read at the centre.
//
// Only the centre is read, so every depthwise conv runs unpadded on the
// shrinking window its centre output depends on: 17 -> 13 -> 9 -> 5 -> 1
// (ResBlock::forward_valid; the same values as the padded trunk's centre).
//
// One CTA per sample at a time (grid-stride over samples). A sample's layers
// all live in shared memory (three fp32 buffers, ~94 KB). All arithmetic is
// fp32 FMA (no TF32), so the result is at least as exact as the cuDNN path.
//
// Backward recomputes the sample's forward, keeping every activation in a
// per-CTA scratch area (~248 KB, stays in L2), then walks the blocks back,
// staging the activations it multiplies with into shared memory. Weight
// gradients accumulate per CTA (each entry owned by one thread: no atomics)
// and a second kernel sums the CTAs.
//
// Parameters come as one flat fp32 vector (layout below), built by
// trunk_fn.cpp with torch::cat, so autograd maps the flat gradient back to the
// module's parameters. 1x1 weights are present twice: transposed (forward
// reads 8 consecutive outputs) and as stored (backward reads 8 consecutive
// inputs); the gradient goes to the as-stored copy only.
#include <cuda_runtime.h>

#include <cstdint>

#include "fused_layers.cuh"
#include "trunk_kernels.hpp"

namespace trunk {
using namespace fl;
#ifdef TRUNK_PROFILE
__device__ long long g_prof[64];
#define PROF(k) do { if (blockIdx.x == 0 && threadIdx.x == 0) g_prof[k] += clock64(); } while (0)
#define PROF0(k) do { if (blockIdx.x == 0 && threadIdx.x == 0) g_prof[k] -= clock64(); } while (0)
#else
#define PROF(k)
#define PROF0(k)
#endif

constexpr int CI = TRUNK_IN_CH, S0 = 17, P0 = S0 * S0;
constexpr int OFF_W0T = 0, OFF_B0 = CI * C, OFF_BLK = CI * C + C;
constexpr int NW = OFF_BLK + 4 * BLK;
static_assert(NW == TRUNK_FLAT, "flat layout");

constexpr int BIG = C * P0;                       // 9248: a 17x17 layer
constexpr int RES = C * 13 * 13;                  // 5408: block 0's residual / output
// shared memory: big1, big2, res, then the weights of the current / next
// 1x1 layer (WPW) and depthwise layer (WDW); 24960 floats = 99840 bytes
constexpr int SMEM_FLOATS = 2 * BIG + RES + WPW + WDW;
// the stem's weights, staged in big2 after the input bytes
constexpr int STEM_W = (CI * P0 + 15) / 16 * 4;   // float offset in big2 (16-byte aligned)
static_assert(STEM_W + CI * C + C <= BIG, "stem staging");

// scratch layout (floats, per CTA): H0, then per block A, B, Cc, O
__host__ __device__ constexpr int blk_base(int b) {
    int o = BIG;
    for (int k = 0; k < b; k++) { const int s = S0 - 4 * k; o += C * (sq(s) + 2 * sq(s - 2) + sq(s - 4)); }
    return o;
}
constexpr int SCRATCH = blk_base(4);              // 61984
static_assert(SCRATCH == TRUNK_SCRATCH, "scratch layout");
struct BlkOff { int A, B, Cc, O; };
__device__ inline BlkOff blk_off(int b) {
    const int s = S0 - 4 * b, base = blk_base(b);
    return {base, base + C * sq(s), base + C * (sq(s) + sq(s - 2)), base + C * (sq(s) + 2 * sq(s - 2))};
}

// The sample's input bytes to shared memory: aligned 32-bit loads, several in
// flight per thread.
__device__ __forceinline__ void load_x(const uint8_t* __restrict__ x, uint8_t* xs) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(x), a0 = a & ~uintptr_t(3);
    const int lead = int(a - a0), nw = (lead + CI * P0 + 3) >> 2;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(a0);
    for (int i0 = threadIdx.x; i0 < nw; i0 += 4 * TRUNK_THREADS) {
        uint32_t v[4];
#pragma unroll
        for (int k = 0; k < 4; k++) { const int i = i0 + k * TRUNK_THREADS; v[k] = i < nw ? __ldg(w + i) : 0u; }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = i0 + k * TRUNK_THREADS;
            if (i >= nw) break;
#pragma unroll
            for (int bt = 0; bt < 4; bt++) {
                const int j = i * 4 + bt - lead;
                if (j >= 0 && j < CI * P0) xs[j] = uint8_t(v[k] >> (8 * bt));
            }
        }
    }
}

// Forward of one sample: ends with the centre (32 values) in big1[0..31].
// With `scr`, every activation is also written to the scratch area.
__device__ void forward_sample(const uint8_t* __restrict__ x, const float* __restrict__ F, float* big1, float* big2,
                               float* res, float* scr) {
    float* wpw = res + RES;          // current / next 1x1 layer: [32][32] then bias
    float* wdw = wpw + WPW;          // current / next depthwise layer: [32][9] then bias
    // the sample's input (bytes) and the stem's weights, staged in big2 (free
    // until block 0); block 0's first 1x1 weights too
    uint8_t* xs = reinterpret_cast<uint8_t*>(big2);
    float* wst = big2 + STEM_W;
    PROF0(0);
    load_x(x, xs);
    stage(F + OFF_W0T, CI * C + C, nullptr, 0, wst);            // W0T then b0 (contiguous)
    stage(F + OFF_BLK + B_WPT, C * C, F + OFF_BLK + B_BP, C, wpw);
    __syncthreads();
    PROF(0); PROF0(1);
    {
        Pre q; q.fetch(F + OFF_BLK + B_WD, 9 * C, F + OFF_BLK + B_BD, C);
        pw<uint8_t, true, false, false>(xs, S0, CI, wst, wst + CI * C, big1);
        q.put(wdw, WDW);
    }
    __syncthreads();
    PROF(1);
    if (scr) copy(big1, scr, BIG);
    for (int b = 0; b < 4; b++) {
        const int S = S0 - 4 * b;
        const float* Wb = F + OFF_BLK + b * BLK;
        BlkOff o{}; if (scr) o = blk_off(b);
        PROF0(2 + 4 * b);
        pw<float, true, false, false>(big1, S, C, wpw, wpw + C * C, big2);       // dw weights already staged
        crop(big1, S, res);
        __syncthreads();
        PROF(2 + 4 * b); PROF0(3 + 4 * b);
        if (scr) copy(big2, scr + o.A, C * sq(S));
        {
            Pre q; q.fetch(Wb + B_WP2T, C * C, Wb + B_BP2, C);
            dw(big2, S, wdw, wdw + 9 * C, big1);
            q.put(wpw, WPW);
        }
        __syncthreads();
        PROF(3 + 4 * b); PROF0(4 + 4 * b);
        if (scr) copy(big1, scr + o.B, C * sq(S - 2));
        {
            Pre q; q.fetch(Wb + B_WD2, 9 * C, Wb + B_BD2, C);
            pw<float, true, false, false>(big1, S - 2, C, wpw, wpw + C * C, big2);
            q.put(wdw, WDW);
        }
        __syncthreads();
        PROF(4 + 4 * b); PROF0(5 + 4 * b);
        if (scr) copy(big2, scr + o.Cc, C * sq(S - 2));
        {
            Pre q;
            if (b < 3) q.fetch(Wb + BLK + B_WPT, C * C, Wb + BLK + B_BP, C);
            dw(big2, S - 2, wdw, wdw + 9 * C, big1, res);
            if (b < 3) q.put(wpw, WPW);
        }
        __syncthreads();
        PROF(5 + 4 * b);
        if (b < 3) {                      // the next block's depthwise weights (its 1x1 ones are staged)
            stage(Wb + BLK + B_WD, 9 * C, Wb + BLK + B_BD, C, wdw);
        }
        if (scr) copy(big1, scr + o.O, C * sq(S - 4));
    }
}

__global__ void __launch_bounds__(TRUNK_THREADS) fwd_kernel(const uint8_t* __restrict__ x, long n,
                                                             const float* __restrict__ F, float* __restrict__ out) {
    extern __shared__ float sm[];
    float *big1 = sm, *big2 = sm + BIG, *res = sm + 2 * BIG;
    for (long s = blockIdx.x; s < n; s += gridDim.x) {
        forward_sample(x + s * long(CI * P0), F, big1, big2, res, nullptr);
        if (threadIdx.x < C) out[s * C + threadIdx.x] = big1[threadIdx.x];
        __syncthreads();
    }
}

__global__ void __launch_bounds__(TRUNK_THREADS) bwd_kernel(const uint8_t* __restrict__ x, long n, const float* __restrict__ F,
                                                             const float* __restrict__ gout, float* __restrict__ scratch,
                                                             float* __restrict__ gacc) {
    extern __shared__ float sm[];
    float *big1 = sm, *big2 = sm + BIG, *res = sm + 2 * BIG;
    float *wpw = res + RES, *wdw = wpw + WPW;
    float* scr = scratch + long(blockIdx.x) * SCRATCH;
    float* G = gacc + long(blockIdx.x) * NW;
    for (long s = blockIdx.x; s < n; s += gridDim.x) {
        const uint8_t* xs = x + s * long(CI * P0);
        forward_sample(xs, F, big1, big2, res, scr);
        // the gradient of the centre: block 3's output is 1x1
        if (threadIdx.x < C) res[threadIdx.x] = gout[s * C + threadIdx.x];
        __syncthreads();
        // res: gradient of the current block's output (not yet masked by its ReLU)
        for (int b = 3; b >= 0; b--) {
            const int S = S0 - 4 * b, Sm = S - 2, So = S - 4;
            const BlkOff o = blk_off(b);
            const float* Wb = F + OFF_BLK + b * BLK;
            float* Gb = G + OFF_BLK + b * BLK;
            // 1. gpre = g * [O > 0]        (res, in place)
            for (int i = threadIdx.x; i < C * sq(So); i += blockDim.x) if (!(scr[o.O + i] > 0.f)) res[i] = 0.f;
            copy(scr + o.Cc, big1, C * sq(Sm));                          // Cc
            stage(Wb + B_WD2, 9 * C, nullptr, 0, wdw);
            stage(Wb + B_WP2, C * C, nullptr, 0, wpw);
            __syncthreads();
            // 2. dw2 weight / bias gradient
            dw_dw(res, big1, Sm, Gb + B_WD2, Gb + B_BD2);
            // 3. gC = dw2^T(gpre) * [Cc > 0]   (big1, in place over Cc)
            __syncthreads();
            dw_t_mask(res, Sm, wdw, big1);
            copy(scr + o.B, big2, C * sq(Sm));                           // B
            __syncthreads();
            // 4. pw2 weight / bias gradient: gC x B
            pw_dw<float>(big1, big2, sq(Sm), C, Gb + B_WP2, C, 1, Gb + B_BP2);
            __syncthreads();
            // 5. gB = pw2^T(gC) * [B > 0]      (big2, in place over B)
            pw<float, false, true, false>(big1, Sm, C, wpw, nullptr, big2);
            __syncthreads();
            copy(scr + o.A, big1, C * sq(S));                            // A
            stage(Wb + B_WD, 9 * C, nullptr, 0, wdw);
            __syncthreads();
            // 6. dw weight / bias gradient: gB x A
            dw_dw(big2, big1, S, Gb + B_WD, Gb + B_BD);
            __syncthreads();
            // 7. gA = dw^T(gB) * [A > 0]      (big1, in place over A)
            dw_t_mask(big2, S, wdw, big1);
            __syncthreads();
            // block input X: H0 or the previous block's output
            copy(b == 0 ? scr : scr + blk_off(b - 1).O, big2, C * sq(S));
            stage(Wb + B_WP, C * C, nullptr, 0, wpw);
            __syncthreads();
            // 8. pw weight / bias gradient: gA x X
            pw_dw<float>(big1, big2, sq(S), C, Gb + B_WP, C, 1, Gb + B_BP);
            __syncthreads();
            // 9. gX = (pw^T(gA) + residual gpre) * [X > 0]  (big2, in place over X;
            //    the mask is the previous block's output ReLU, or the stem's)
            pw<float, false, true, true>(big1, S, C, wpw, nullptr, big2, res);
            __syncthreads();
            if (b > 0) { copy(big2, res, C * sq(S)); __syncthreads(); }
        }
        // stem: gH0 (masked) in big2; gW0T[c][o] += sum_p gH0[o][p] x[c][p] (input bytes staged in big1)
        load_x(xs, reinterpret_cast<uint8_t*>(big1));
        __syncthreads();
        pw_dw<uint8_t>(big2, reinterpret_cast<const uint8_t*>(big1), P0, CI, G + OFF_W0T, 1, C, G + OFF_B0);
        __syncthreads();
    }
}

__global__ void reduce_kernel(const float* __restrict__ gacc, int ctas, float* __restrict__ g) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= NW) return;
    float s = 0.f;
    for (int k = 0; k < ctas; k++) s += gacc[long(k) * NW + j];
    g[j] = s;
}

static int sm_count() {
    static int n = 0;
    if (!n) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); }
    return n;
}
static void set_smem() {
    static bool done = false;
    if (done) return;
    cudaFuncSetAttribute(fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_FLOATS * 4);
    cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_FLOATS * 4);
    done = true;
}

}  // namespace trunk

int trunk_grid(long n) { const long g = trunk::sm_count(); return int(n < g ? n : g); }

void trunk_forward(const uint8_t* x, long n, const float* flat, float* out, void* stream) {
    if (n <= 0) return;
    trunk::set_smem();
    trunk::fwd_kernel<<<trunk_grid(n), TRUNK_THREADS, trunk::SMEM_FLOATS * 4, (cudaStream_t)stream>>>(x, n, flat, out);
}

void trunk_backward(const uint8_t* x, long n, const float* flat, const float* gout, float* scratch, float* gacc,
                    float* gflat, void* stream) {
    trunk::set_smem();
    const int grid = trunk_grid(n);
    cudaMemsetAsync(gacc, 0, sizeof(float) * size_t(grid) * TRUNK_FLAT, (cudaStream_t)stream);
    if (n > 0)
        trunk::bwd_kernel<<<grid, TRUNK_THREADS, trunk::SMEM_FLOATS * 4, (cudaStream_t)stream>>>(x, n, flat, gout, scratch, gacc);
    trunk::reduce_kernel<<<(TRUNK_FLAT + 255) / 256, 256, 0, (cudaStream_t)stream>>>(gacc, grid > 0 ? grid : 0, gflat);
}

#ifdef TRUNK_PROFILE
void trunk_prof(long long* out) { cudaMemcpyFromSymbol(out, trunk::g_prof, sizeof(long long) * 64); }
void trunk_prof_reset() { long long z[64] = {}; cudaMemcpyToSymbol(trunk::g_prof, z, sizeof(z)); }
#endif
const char* trunk_last_error() {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? nullptr : cudaGetErrorString(e);
}
