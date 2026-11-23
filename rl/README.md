# Florent RL workspace

This directory is the Florent adaptation of the supplied PPO pipeline. The cambc simulator and game rules are intentionally excluded.

## Current status

- `sim/florent_actions.*`: native-engine candidate enumeration, legality filtering, and application.
- `sim/florent_game.*`: native match wrapper with one critic observation per round and terminal-only reward.
- `train/florent_layout.hpp`: ABI dimensions, unit policy types, PPO constants, and `{+1,0,-1}` reward.
- `train/policy_math.hpp`: stable legal-choice masked log-softmax and entropy.
- `bot/action_abi.py`: production-shaped candidate/action contract.
- `tests/action_smoke.cpp`: random-first-legal rollout across native maps.

## Build

```bash
cmake -S rl -B /tmp/florent_rl_build
cmake --build /tmp/florent_rl_build -j2
/tmp/florent_rl_build/action_smoke maps/gen00.map26
```

## Reward contract

Training uses terminal reward only: the winner receives `+1`, the loser `-1`, and every non-terminal or draw state receives `0`. No economy, combat, delivery, damage, or survival shaping is included.

## Next integration layers

The next layers are the spatial model, trajectory storage/GAE, PPO update, and float-to-Python quantized exporter. They must consume this ABI and the exact observation encoders in `engine/src/obs.*` and `bots/econ/encoder.py`.
