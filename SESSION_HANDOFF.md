# Florent RL — New Session Handoff

## User’s locked architecture

Do not substitute an MLP or a single flat action head.

### Actor
- Production actor observation: local 17x17 patch plus 40 scalars.
- Spatial trunk: 32 channels, four residual blocks.
- Each block: 1x1 pointwise convolution, then depthwise 3x3 convolution, then residual addition.
- Separate actor trunk/head per acting type: core, builder, gunner, sentinel, launcher.
- Spike uses deterministic fire/wait behavior initially.
- Units with no legal action beyond wait must skip actor forward entirely.

### Critic
- Centralized, privileged global observation: 48x48 board plus critic scalars.
- Spatial critic, not an MLP:
  - residual convolution block
  - 2x2 max-pool
  - residual convolution block
  - 2x2 max-pool
  - residual convolution block
  - 2x2 max-pool
  - concatenate critic scalars
  - value projection
- Geometry: 48 -> 24 -> 12 -> 6.

### PPO policy semantics (important)
Use the supplied pipeline’s factorized-head semantics:
- Each head has its own categorical distribution and masked softmax.
- Store each participating head’s old log-probability separately.
- Recompute each participating head’s new log-probability separately.
- Apply the same turn/round advantage independently to every participating head.
- PPO ratio/clipping is per head.
- Aggregate losses, never log-probabilities.
- Do not sum head log-probs into a joint log-probability.
- Do not compose component logits into one candidate logit and softmax over candidates.
- Builder movement/action exclusivity needs an explicit committed mode or equivalent pipeline head structure.

## Current repository

Path: `/home/hamza/florent/florent_code`

Important directories:
- `engine/`: native exact engine and observation encoder.
- `bots/econ/`: heuristic bot, production encoder, ABI and policy loader.
- `rl/sim/`: native Florent rollout/action wrapper.
- `rl/train/`: LibTorch trainer/model/trajectory code.
- `rl/bot/`: deployment ABI/export/policy/SWAR work.
- `research/nn_bench/`: SWAR benchmark experiments.
- `docs/`: observation, team memory and RL layout documents.

## What is validated

- Native engine build.
- Observation C++/Python parity: zero mismatches and zero leaks in tested run.
- Native action smoke and random rollout smoke.
- Official heuristic bot runs under `--tle 10` on gen00 and pit01.
- Spatial actor + spatial critic model builds.
- One spatial training iteration produced finite loss/checkpoint.
- Structured candidate-head prototype builds and a candidate-gather PPO smoke produced a checkpoint.

## What is NOT finished

The current `rl/train/neural_train.cpp` still contains the candidate-level `policy_scores()`/`masked_logp()` path. This is not the final PPO semantics. Replace it with factorized per-head rollout records and per-head PPO losses according to the locked semantics above.

The current `ActionHeadImpl` is only a prototype and must be redesigned to expose named/typed factorized heads and an exact decoder mapping. Do not retain arbitrary `72 -> 1024` output logic.

The SWAR runtime exists in `rl/bot/swar.py` and has primitive numerical checks, but final spatial checkpoint export, quantization parity, and learned-policy official match are not complete.

The final gate is:
1. factorized head PPO implementation;
2. exact C++/Python head/action/mask parity;
3. checkpoint reload test;
4. spatial float-to-SWAR/int4 export;
5. float/quantized action agreement test;
6. `FLORENT_POLICY=... fcode run ... --tle 10` with proof learned inference is active;
7. replay verification.

## Core commands

```bash
cmake -S rl -B /tmp/florent_rl_build -DCMAKE_PREFIX_PATH=/home/hamza/libtorch
cmake --build /tmp/florent_rl_build -j2

CUDA_VISIBLE_DEVICES=-1 LD_LIBRARY_PATH=/home/hamza/libtorch/lib \
  /tmp/florent_rl_build/neural_train maps/gen00.map26 1 /tmp/checkpoint.pt

python3 -I -m py_compile bots/econ/*.py rl/bot/*.py
.venv/bin/python3 engine/tools/obs_parity.py --games 1 --rounds 20 --seed 7 --map maps/gen00.map26
/tmp/florent_rl_build/action_smoke maps/gen00.map26
/tmp/florent_rl_build/rollout_smoke maps/gen00.map26 16
.venv/bin/fcode run econ starter maps/gen00.map26 --seed 3 --tle 10
```

Never include `.env`, credentials, checkpoints, replay artifacts, `.git`, or `__pycache__` in a handoff archive.
