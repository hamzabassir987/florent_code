// Fused CUDA actor trunk (trunk_kernels.cu); plain C interface, no torch.
#pragma once
#include <cstdint>

#define TRUNK_IN_CH 59          // fe::ACTOR_CH
#define TRUNK_FLAT 21120        // flat parameter vector length (layout in trunk_kernels.cu)
#define TRUNK_SCRATCH 61984     // backward scratch floats per CTA
#define TRUNK_THREADS 320

// grid size (CTAs) for n samples: backward needs grid * TRUNK_SCRATCH scratch
// floats and grid * TRUNK_FLAT gradient-accumulator floats
int trunk_grid(long n);
// x: [n, 59*17*17] uint8 (device); flat: [TRUNK_FLAT]; out: [n, 32]
void trunk_forward(const uint8_t* x, long n, const float* flat, float* out, void* stream);
// gout: [n, 32]; gflat: [TRUNK_FLAT] (overwritten)
void trunk_backward(const uint8_t* x, long n, const float* flat, const float* gout, float* scratch, float* gacc,
                    float* gflat, void* stream);
const char* trunk_last_error();   // nullptr if none

// ---- the critic's spatial part (critic_kernels.cu)
#define CRITIC_IN_CH 29         // fe::CRITIC_CH
#define CRITIC_FLAT 15360       // flat parameter vector length
#define CRITIC_SCRATCH 33536    // backward scratch floats per CTA
#define CRITIC_OUT 288          // 32 x 3 x 3
// CTAs of the backward: it needs grid * CRITIC_SCRATCH scratch floats and
// grid * CRITIC_FLAT gradient-accumulator floats
int critic_grid(long n);
// x: [n, 29*48*48] uint8; p1: [n, 32*24*24], p2: [n, 32*12*12] (level
// outputs, kept for the backward); out: [n, 288]
void critic_forward(const uint8_t* x, long n, const float* flat, float* p1, float* p2, float* out, void* stream);
// gout: [n, 288]; gp1 / gp2: scratch gradients like p1 / p2; gflat overwritten
void critic_backward(const uint8_t* x, long n, const float* flat, const float* p1, const float* p2, const float* gout,
                     float* gp1, float* gp2, float* scratch, float* gacc, float* gflat, void* stream);
