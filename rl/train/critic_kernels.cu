// Fused CUDA kernels for the critic's spatial part (see model.hpp, CriticImpl):
//
//   stem 1x1 (29 -> 32) + ReLU on the 48x48 board, then 3 x [residual block
//   (1x1 -> ReLU -> depthwise 3x3 -> ReLU -> 1x1 -> ReLU -> depthwise 3x3
//   -> + input -> ReLU, both depthwise convs zero-padded) -> max-pool]
//   with pools 2, 2, 4: 48 -> 24 -> 12 -> 3. Output: [n, 32*3*3] in
//   PyTorch's flatten order.
//
// One launch per level (48, 24, 12). A level is cut into 12x12 output tiles;
// a tile reads its 16x16 input window (2-tile halo) and runs the block with
// the shared valid-conv layers of fused_layers.cuh. Zero padding is exact:
// after each layer that feeds a depthwise conv, the window's positions off the
// board are set to 0 (the padding value). The tile's 12x12 output is
// max-pooled (PyTorch's rule: the first maximum in row-major order) and
// written to the next level's input.
//
// Backward runs the levels in reverse. A tile recomputes its forward (every
// activation into a per-CTA scratch area, L2-resident), sends the pooled
// gradient to the max positions, walks the block back, and adds its input
// gradient to the previous level's with atomics (neighbouring windows
// overlap). Weight gradients accumulate per CTA, then one kernel sums them.
//
// All arithmetic is fp32 FMA (no TF32).
#include <cuda_runtime.h>

#include <cstdint>

#include "fused_layers.cuh"
#include "trunk_kernels.hpp"

namespace critic {
using namespace fl;

constexpr int CI = CRITIC_IN_CH, H0 = 48;
constexpr int OFF_W0T = 0, OFF_B0 = CI * C, OFF_BLK = CI * C + C;
constexpr int NW = OFF_BLK + 3 * BLK;
static_assert(NW == CRITIC_FLAT, "critic flat layout");

constexpr int T = 12, TI = T + 4, PT = TI * TI, TM = T + 2;     // output tile, input window, its pixels, middle side
constexpr int BIG = C * PT;                                     // 8192
constexpr int RES = C * T * T;                                  // 4608
constexpr int SMEM_FLOATS = 2 * BIG + RES + WPW + WDW;          // 22368 floats = 89472 bytes
constexpr int STEM_W = (CI * PT + 15) / 16 * 4;                 // stem weights after the input bytes, in big2
static_assert(STEM_W + CI * C + C <= BIG, "stem staging");
// scratch (floats, per CTA): X, A (16x16), B, Cc (14x14), O (12x12)
constexpr int S_X = 0, S_A = BIG, S_B = 2 * BIG, S_C = 2 * BIG + C * TM * TM, S_O = 2 * BIG + 2 * C * TM * TM;
constexpr int SCRATCH = S_O + RES;
static_assert(SCRATCH == CRITIC_SCRATCH, "critic scratch layout");

__host__ __device__ constexpr int side(int L) { return H0 >> L; }          // 48, 24, 12
__host__ __device__ constexpr int ntiles(int L) { return side(L) / T; }    // 4, 2, 1 per dimension
__host__ __device__ constexpr int pool(int L) { return L == 2 ? 4 : 2; }
__host__ __device__ constexpr int pside(int L) { return side(L) / pool(L); }   // 24, 12, 3

// positions of a window (side S, first position at board (y0, x0)) that are
// off the board (side H) are set to 0
__device__ __forceinline__ void zero_off_board(float* buf, int S, int y0, int x0, int H) {
    const int P = S * S;
    for (int i = threadIdx.x; i < C * P; i += blockDim.x) {
        const int p = i % P, y = y0 + p / S, x = x0 + p % S;
        if (y < 0 || y >= H || x < 0 || x >= H) buf[i] = 0.f;
    }
}

// The input window's bytes (level 0): xs[c][p], 0 off the board.
__device__ __forceinline__ void load_bytes(const uint8_t* __restrict__ x, int y0, int x0, uint8_t* xs) {
    for (int i = threadIdx.x; i < CI * PT; i += blockDim.x) {
        const int c = i / PT, p = i % PT, y = y0 + p / TI, xx = x0 + p % TI;
        xs[i] = (y >= 0 && y < H0 && xx >= 0 && xx < H0) ? x[c * H0 * H0 + y * H0 + xx] : uint8_t(0);
    }
}

// Forward of one tile of level L. On return big1 holds the tile's output O
// (12x12). With `scr`, every activation is also written to the scratch area.
//   x8: the sample's board bytes (level 0); xin: the sample's level input
//   [32][side][side] (levels 1, 2)
__device__ void tile_fwd(int L, const uint8_t* __restrict__ x8, const float* __restrict__ xin, const float* __restrict__ F,
                         int ty, int tx, float* big1, float* big2, float* res, float* wpw, float* wdw, float* scr) {
    const int H = side(L), y0 = ty * T - 2, x0 = tx * T - 2;
    const float* Wb = F + OFF_BLK + L * BLK;
    if (L == 0) {
        uint8_t* xs = reinterpret_cast<uint8_t*>(big2);
        float* wst = big2 + STEM_W;
        load_bytes(x8, y0, x0, xs);
        stage(F + OFF_W0T, CI * C + C, nullptr, 0, wst);
        stage(Wb + B_WPT, C * C, Wb + B_BP, C, wpw);
        __syncthreads();
        {
            Pre q; q.fetch(Wb + B_WD, 9 * C, Wb + B_BD, C);
            pw<uint8_t, true, false, false>(xs, TI, CI, wst, wst + CI * C, big1);
            q.put(wdw, WDW);
        }
    } else {
        for (int i = threadIdx.x; i < C * PT; i += blockDim.x) {
            const int c = i / PT, p = i % PT, y = y0 + p / TI, xx = x0 + p % TI;
            big1[i] = (y >= 0 && y < H && xx >= 0 && xx < H) ? xin[c * H * H + y * H + xx] : 0.f;
        }
        stage(Wb + B_WPT, C * C, Wb + B_BP, C, wpw);
        stage(Wb + B_WD, 9 * C, Wb + B_BD, C, wdw);
    }
    __syncthreads();
    if (scr) copy(big1, scr + S_X, BIG);
    pw<float, true, false, false>(big1, TI, C, wpw, wpw + C * C, big2);          // A
    crop(big1, TI, res);
    __syncthreads();
    zero_off_board(big2, TI, y0, x0, H);                                         // A is the depthwise input
    __syncthreads();
    if (scr) copy(big2, scr + S_A, BIG);
    {
        Pre q; q.fetch(Wb + B_WP2T, C * C, Wb + B_BP2, C);
        dw(big2, TI, wdw, wdw + 9 * C, big1);                                    // B (14x14)
        q.put(wpw, WPW);
    }
    __syncthreads();
    if (scr) copy(big1, scr + S_B, C * TM * TM);
    {
        Pre q; q.fetch(Wb + B_WD2, 9 * C, Wb + B_BD2, C);
        pw<float, true, false, false>(big1, TM, C, wpw, wpw + C * C, big2);      // Cc (14x14)
        q.put(wdw, WDW);
    }
    __syncthreads();
    zero_off_board(big2, TM, y0 + 1, x0 + 1, H);                                 // Cc is the depthwise input
    __syncthreads();
    if (scr) copy(big2, scr + S_C, C * TM * TM);
    dw(big2, TM, wdw, wdw + 9 * C, big1, res);                                   // O (12x12) = relu(dw2 + X crop)
    __syncthreads();
    if (scr) copy(big1, scr + S_O, RES);
}

// max-pool window of pooled cell (py, px) of channel c in the tile output o:
// the position of the first maximum in row-major order (PyTorch's choice)
__device__ __forceinline__ int pool_arg(const float* o, int c, int py, int px, int pl) {
    const float* oc = o + c * T * T;
    int best = (py * pl) * T + px * pl;
    float m = -INFINITY;
    for (int dy = 0; dy < pl; dy++)
        for (int dx = 0; dx < pl; dx++) {
            const int q = (py * pl + dy) * T + (px * pl + dx);
            const float v = oc[q];
            if (v > m || isnan(v)) { m = v; best = q; }
        }
    return best;
}

__global__ void __launch_bounds__(TRUNK_THREADS) fwd_kernel(int L, const uint8_t* __restrict__ x, const float* __restrict__ xin,
                                                             long n, const float* __restrict__ F, float* __restrict__ out) {
    extern __shared__ float sm[];
    float *big1 = sm, *big2 = sm + BIG, *res = sm + 2 * BIG, *wpw = res + RES, *wdw = wpw + WPW;
    const int nt = ntiles(L), H = side(L), PH = pside(L), pl = pool(L), so = T / pl;
    const long items = n * nt * nt;
    for (long w = blockIdx.x; w < items; w += gridDim.x) {
        const long s = w / (nt * nt);
        const int t = int(w % (nt * nt)), ty = t / nt, tx = t % nt;
        tile_fwd(L, x + s * long(CI * H0 * H0), L ? xin + s * long(C * H * H) : nullptr, F, ty, tx, big1, big2, res, wpw, wdw, nullptr);
        for (int i = threadIdx.x; i < C * so * so; i += blockDim.x) {
            const int c = i / (so * so), q = i % (so * so), py = q / so, px = q % so;
            out[(s * C + c) * PH * PH + (ty * so + py) * PH + (tx * so + px)] = big1[c * T * T + pool_arg(big1, c, py, px, pl)];
        }
        __syncthreads();
    }
}

// gout: gradient of this level's pooled output (same layout as fwd's `out`);
// gin: gradient of this level's input (levels 1, 2; accumulated with atomics)
__global__ void __launch_bounds__(TRUNK_THREADS) bwd_kernel(int L, const uint8_t* __restrict__ x, const float* __restrict__ xin,
                                                             long n, const float* __restrict__ F, const float* __restrict__ gout,
                                                             float* __restrict__ gin, float* __restrict__ scratch,
                                                             float* __restrict__ gacc) {
    extern __shared__ float sm[];
    float *big1 = sm, *big2 = sm + BIG, *res = sm + 2 * BIG, *wpw = res + RES, *wdw = wpw + WPW;
    float* scr = scratch + long(blockIdx.x) * SCRATCH;
    float* G = gacc + long(blockIdx.x) * NW;
    float* Gb = G + OFF_BLK + L * BLK;
    const float* Wb = F + OFF_BLK + L * BLK;
    const int nt = ntiles(L), H = side(L), PH = pside(L), pl = pool(L), so = T / pl;
    const long items = n * nt * nt;
    for (long w = blockIdx.x; w < items; w += gridDim.x) {
        const long s = w / (nt * nt);
        const int t = int(w % (nt * nt)), ty = t / nt, tx = t % nt;
        const int y0 = ty * T - 2, x0 = tx * T - 2;
        const uint8_t* x8 = x + s * long(CI * H0 * H0);
        tile_fwd(L, x8, L ? xin + s * long(C * H * H) : nullptr, F, ty, tx, big1, big2, res, wpw, wdw, scr);
        // pooled gradient -> the max positions of O (res)
        for (int i = threadIdx.x; i < RES; i += blockDim.x) res[i] = 0.f;
        __syncthreads();
        for (int i = threadIdx.x; i < C * so * so; i += blockDim.x) {
            const int c = i / (so * so), q = i % (so * so), py = q / so, px = q % so;
            res[c * T * T + pool_arg(big1, c, py, px, pl)] = gout[(s * C + c) * PH * PH + (ty * so + py) * PH + (tx * so + px)];
        }
        __syncthreads();
        // 1. gpre = gO * [O > 0]  (res, in place)
        for (int i = threadIdx.x; i < RES; i += blockDim.x) if (!(scr[S_O + i] > 0.f)) res[i] = 0.f;
        copy(scr + S_C, big1, C * TM * TM);                                       // Cc
        stage(Wb + B_WD2, 9 * C, nullptr, 0, wdw);
        stage(Wb + B_WP2, C * C, nullptr, 0, wpw);
        __syncthreads();
        // 2. dw2 weight / bias gradient
        dw_dw(res, big1, TM, Gb + B_WD2, Gb + B_BD2);
        __syncthreads();
        // 3. gC = dw2^T(gpre) * [Cc > 0]   (big1, in place over Cc)
        dw_t_mask(res, TM, wdw, big1);
        copy(scr + S_B, big2, C * TM * TM);                                       // B
        __syncthreads();
        // 4. pw2 weight / bias gradient
        pw_dw<float>(big1, big2, TM * TM, C, Gb + B_WP2, C, 1, Gb + B_BP2);
        __syncthreads();
        // 5. gB = pw2^T(gC) * [B > 0]     (big2, in place over B)
        pw<float, false, true, false>(big1, TM, C, wpw, nullptr, big2);
        __syncthreads();
        copy(scr + S_A, big1, BIG);                                               // A
        stage(Wb + B_WD, 9 * C, nullptr, 0, wdw);
        __syncthreads();
        // 6. dw weight / bias gradient
        dw_dw(big2, big1, TI, Gb + B_WD, Gb + B_BD);
        __syncthreads();
        // 7. gA = dw^T(gB) * [A > 0]      (big1, in place over A)
        dw_t_mask(big2, TI, wdw, big1);
        __syncthreads();
        copy(scr + S_X, big2, BIG);                                               // X
        stage(Wb + B_WP, C * C, nullptr, 0, wpw);
        __syncthreads();
        // 8. pw weight / bias gradient
        pw_dw<float>(big1, big2, PT, C, Gb + B_WP, C, 1, Gb + B_BP);
        __syncthreads();
        // 9. gX = (pw^T(gA) + residual gpre) * [X > 0]   (big2, in place over X)
        pw<float, false, true, true>(big1, TI, C, wpw, nullptr, big2, res);
        __syncthreads();
        if (L > 0) {
            // to the previous level's output gradient (windows overlap: atomics)
            float* gi = gin + s * long(C * H * H);
            for (int i = threadIdx.x; i < BIG; i += blockDim.x) {
                const int c = i / PT, p = i % PT, y = y0 + p / TI, xx = x0 + p % TI;
                const float v = big2[i];
                if (v != 0.f && y >= 0 && y < H && xx >= 0 && xx < H) atomicAdd(gi + c * H * H + y * H + xx, v);
            }
        } else {
            // stem: gW0T[c][o] += sum_p gH0[o][p] x[c][p], gb0[o] += sum_p gH0[o][p]
            uint8_t* xs = reinterpret_cast<uint8_t*>(big1);
            load_bytes(x8, y0, x0, xs);
            __syncthreads();
            pw_dw<uint8_t>(big2, xs, PT, CI, G + OFF_W0T, 1, C, G + OFF_B0);
        }
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
static int grid_for(long items) { const long g = sm_count(); return int(items < g ? items : g); }

}  // namespace critic

int critic_grid(long n) { return critic::grid_for(n * critic::ntiles(0) * critic::ntiles(0)); }

void critic_forward(const uint8_t* x, long n, const float* flat, float* p1, float* p2, float* out, void* stream) {
    if (n <= 0) return;
    critic::set_smem();
    cudaStream_t st = (cudaStream_t)stream;
    const size_t sm = critic::SMEM_FLOATS * 4;
    const float* ins[3] = {nullptr, p1, p2};
    float* outs[3] = {p1, p2, out};
    for (int L = 0; L < 3; L++) {
        const long items = n * critic::ntiles(L) * critic::ntiles(L);
        critic::fwd_kernel<<<critic::grid_for(items), TRUNK_THREADS, sm, st>>>(L, x, ins[L], n, flat, outs[L]);
    }
}

void critic_backward(const uint8_t* x, long n, const float* flat, const float* p1, const float* p2, const float* gout,
                     float* gp1, float* gp2, float* scratch, float* gacc, float* gflat, void* stream) {
    critic::set_smem();
    cudaStream_t st = (cudaStream_t)stream;
    const int grid = critic_grid(n);
    cudaMemsetAsync(gacc, 0, sizeof(float) * size_t(grid) * CRITIC_FLAT, st);
    if (n > 0) {
        cudaMemsetAsync(gp1, 0, sizeof(float) * size_t(n) * 32 * 24 * 24, st);
        cudaMemsetAsync(gp2, 0, sizeof(float) * size_t(n) * 32 * 12 * 12, st);
        const size_t sm = critic::SMEM_FLOATS * 4;
        const float* ins[3] = {nullptr, p1, p2};
        const float* gouts[3] = {gp1, gp2, gout};
        float* gins[3] = {nullptr, gp1, gp2};
        for (int L = 2; L >= 0; L--) {
            const long items = n * critic::ntiles(L) * critic::ntiles(L);
            // every level's grid fits the scratch / accumulators sized for level 0
            critic::bwd_kernel<<<critic::grid_for(items), TRUNK_THREADS, sm, st>>>(L, x, ins[L], n, flat, gouts[L], gins[L], scratch, gacc);
        }
    }
    critic::reduce_kernel<<<(CRITIC_FLAT + 255) / 256, 256, 0, st>>>(gacc, grid > 0 ? grid : 0, gflat);
}
