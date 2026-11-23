# HANDOFF — Florent Code League / RL continuation

This file is the complete handoff for a fresh Claude Code session. Read it before doing anything. The repository is `/home/hamza/florent/florent_code` and is a git repository, currently with uncommitted work. The user wants the entire train-to-deploy RL system completed, not a sequence of milestone explanations.

## 0. User requirements and working rules

- Continue working autonomously; do not stop after a compile or smoke test.
- Do not repeatedly explain that the work is incomplete. Implement the next missing layer.
- The supplied `rl_pipeline.zip` is an architecture/template from another game. Reuse its generic PPO/GAE/rollout/checkpoint/distillation/parity ideas, but do not port its game rules, token layout, action semantics, maps, or reward shaping.
- Reward must be **only terminal**:
  - winner: `+1`
  - loser: `-1`
  - non-terminal/draw: `0`
  - no economy, damage, delivery, survival, or combat shaping.
- Exactness matters for the native engine and trainer/deployment parity.
- Production bot is pure Python in a sub-interpreter. No NumPy. Official per-unit budget is 10 ms plus banked buffer.
- Do not ask the user to choose routine implementation details. Make sensible decisions and continue.
- Do not claim the system is complete until a learned policy is actually integrated into the bot, exported, and tested under official `fcode run --tle 10`.

## 1. Game/runtime facts

Florent Code League package/runtime is `fcode 2.6.0`.

- Two symmetric teams, up to 1000 rounds.
- Entry point: `main.py`, top-level `class Player`, `run(self, ct)`.
- `run()` is called once per round for every living acting unit.
- Units: core, builder, gunner, sentinel, launcher, spike.
- Buildings: conveyor, splitter, harvester, barrier, mine.
- Builders are the mobile units. A builder may move or perform a normal action per turn; builders move cardinally.
- Team store has 16 slots; writes are visible next round, not immediately.
- Fog is per unit. Out-of-vision controller calls can raise exceptions.
- An uncaught exception destroys the unit. Existing bot wraps `run()` defensively.
- Official test must use `--tle 10`; default local `--tle 0` does not enforce timing.
- NumPy cannot be used reliably in the runtime.

Important official commands from repo root:

```bash
.venv/bin/fcode run econ starter maps/gen00.map26 --seed 3 --replay check_s3.replay26 --tle 10
.venv/bin/python3 engine/tools/verify.py check_s3.replay26
.venv/bin/python3 engine/tools/obs_parity.py --games 8 --rounds 250 --seed 7 --map maps/gen00.map26
```

Official ladder map access requires `fcode login`; credentials are not available in this session. Local maps are under `maps/`.

## 2. Repository layout

- `engine/src/fengine.hpp/.cpp`: exact native engine.
- `engine/src/capi.cpp`: C API for ctypes/trainer.
- `engine/src/obs.hpp/.cpp`: C++ actor/critic observation encoder.
- `engine/Makefile`: builds `engine/build/libfengine.so`.
- `engine/tools/verify.py`: replay-driven exactness verifier.
- `engine/tools/obs_parity.py`: C++/Python observation parity harness.
- `bots/econ/main.py`: current heuristic bot, modified with team-memory report bus and encoder integration.
- `bots/econ/encoder.py`: pure-Python production-shaped actor encoder.
- `docs/obs.md`: observation specification.
- `docs/team-memory.md`: delayed team-store/report-bus specification.
- `docs/rl-layout.md`: original Florent action/layout proposal; some details need correction before final ABI.
- `rl/`: new experimental Florent RL subtree added during this session.
- `rl_pipeline.zip`: supplied architecture reference at `/home/hamza/florent/rl_pipeline.zip`.
- Extracted reference audit directory: `/tmp/rl_pipeline_review/rl_pipeline`.
- `.env` outside the repo contains a secret and must never be committed or included.

## 3. Native engine status: DONE and exact

The native engine was already implemented before the RL work and is the source of truth.

Key APIs in `engine/src/fengine.hpp`:

- `load_map`, `reset`, `turn_order`, `end_round`.
- State: `ents`, `bld`, `bot`, `env`, `teams`, `round`, `winner`, `over`, `last_moves`.
- Actions: `move`, `build`, `attack`, `heal`, `destroy`, `self_destruct`, `spawn`, `convert_ammo`, `fire`, `rotate`, `launch`, `write_store`, `resign`.
- Checks: `can_move`, `can_build`, `can_attack`, `can_heal`, `can_destroy`, `can_spawn`, `can_convert`, `can_fire`, `can_rotate`, `can_launch`.
- Fog/vision: `in_vision`, `dist2_from`, `nearby_tiles`, `nearby_ids`, `visible_building`, `tile_building`, `tile_empty`, `tile_passable`.

Exactness history:

- Replay verifier and official probes established engine parity over many generated and ladder replays.
- Engine speed is approximately 500 full random games/s single-threaded.
- Store timing was tested: same-round reads see the old snapshot; writes become visible after the round.

Do not modify engine rules casually. Any engine change must be verified against official replay/probe behavior.

## 4. Observation system: implemented and parity-tested

### C++

`engine/src/obs.hpp/.cpp` defines:

- Actor patch: 17x17.
- Actor spatial channels: 59.
- Actor scalar features: 40.
- Per-unit `ObsMemory`: terrain, remembered buildings/entities, staleness, builder visit history, direction, HP.
- Raw edge-ray channels including unseen count.
- Turret possible/confirmed reach cues.
- Critic: global 48x48 padded board, 29 channels, 33 scalars.
- Critic is intended to be computed once per round, not once per unit.

Important dimensions:

```text
ACTOR_SPATIAL = 59 * 17 * 17 = 17051
ACTOR_SCALARS = 40
ACTOR_INPUT   = 17091
CRITIC_SPATIAL = 29 * 48 * 48 = 66816
CRITIC_SCALARS = 33
```

`engine/src/capi.cpp` exposes observation functions and `fe_get_store`.

### Python

`bots/econ/encoder.py` contains:

- `ObsMemory`.
- `ControllerAdapter`.
- `ActorEncoder` keyed by unit ID.
- Pure Python, no NumPy.
- Same channel/scalar semantics as C++.

A real bug was fixed in `ControllerAdapter.entity_direction`: official `get_direction()` can throw for non-facing entities such as harvesters, barriers, mines, and cores. The Python adapter now catches that and returns centre/unavailable semantics matching C++.

### Parity result

Latest run passed:

```text
mismatches: 0
leaks: 0
critic failures: 0
PARITY OK
```

The encoder timing measured approximately 0.5 ms median and approximately 1 ms p95 per unit in the parity harness.

Do not regress this parity. The actor must only receive production-observable information. The critic may be privileged and global.

## 5. Current heuristic bot status

`bots/econ/main.py` is a working heuristic economy/siege bot.

Design:

- Per-unit map memory.
- Known resource reports through delayed 16-slot team store.
- Priority report bus:
  - slot 0 core position;
  - slot 1 harvester count;
  - slot 2 threat;
  - slots 3–5 resource reports;
  - slots 6–8 structure reports;
  - slots 9–12 builder sightings;
  - slots 13–14 guard lease;
  - slot 15 phase.
- Packet format has valid bit, action, category, team, tile, round modulo 1024.
- Collision/pending retry logic exists.
- Resource positions persist; builders move freely; conveyor path planning is not builder movement macro-control.
- Known resource is required before path planning/harvester construction.
- Visible harvester masks build-harvester but does not delete resource/path knowledge.

The bot now imports `ActorEncoder` optionally and calls it at the start of each turn, storing `last_observation`. Learned inference is not yet used for decisions; heuristic behavior remains the fallback.

Official runtime regression recently passed:

```text
fcode run econ starter maps/gen00.map26 --seed 3 --tle 10
Winner: econ, core destroyed, turn 611
```

Another recent test:

```text
fcode run econ starter maps/pit01.map26 --seed 3 --tle 10
Completed turn 1000
Winner: econ by titanium tiebreak
No TLE or exception
```

The replay verifier reported the known conveyor tie-order differences in one local run, but core state, HP, totals, and winner checks passed. Re-run verifier after future bot changes.

## 6. User’s final action/policy design

One model/trunk per unit type:

- Core: spawn builder, ammo conversion amount, wait.
- Builder: free movement, all building types, local attack/heal/destroy, wait.
- Gunner: fire, rotate, wait.
- Sentinel: fire, wait.
- Spike: heuristic fire/wait initially.
- Launcher: own policy; select adjacent builder and destination; wait.
- Harvester/conveyor/splitter/barrier/mine do not act independently.

Important corrections still required before final ABI:

1. Builder normal actions are mutually exclusive with movement. Do not model separate independent sampled heads whose product is not the executed action probability. Use one main categorical union of legal actions, or explicitly model free-action heads and one committed main action.
2. `destroy` and `convert_ammo` have special semantics and need exact treatment. Verify whether they are free/repeatable before final policy ABI.
3. Board-sized logits violate fixed checkpoint ABI. Use fixed offset tables/windows for spawn and launcher destination rather than map-dependent board-sized heads.
4. `can_launch` may permit enemy builder targets; do not restrict to friendly builders without verifying the rule.
5. Builder movement is cardinal only; facing/build directions may be 8-way.
6. Every legal choice set must be nonempty. Always retain wait/none.
7. Conveyor/splitter planner masks must be shared exactly between C++ and Python, or omit planner restriction in v1 and let the policy choose all legal builds.

The current `docs/rl-layout.md` is a design draft, not a final validated ABI. Fix it to reflect the above before relying on checkpoints.

## 7. Supplied RL pipeline audit

The archive is Battlecode/cambc-specific. Useful generic pieces:

- PPO clipped policy loss.
- GAE and truncation.
- Legal choice-set masked log-softmax.
- Entropy scheduling.
- KL stopping and gradient clipping.
- Checkpoint save/load/widening patterns.
- Teacher/student distillation.
- Quantization/calibration ideas.
- Evaluation/opponent pool/dashboard patterns.

Reference files:

- `/tmp/rl_pipeline_review/rl_pipeline/train/main.cpp`
- `/tmp/rl_pipeline_review/rl_pipeline/train/model.cpp`
- `/tmp/rl_pipeline_review/rl_pipeline/train/game.hpp`
- `/tmp/rl_pipeline_review/rl_pipeline/train/env.hpp`
- `/tmp/rl_pipeline_review/rl_pipeline/train/layout.hpp`
- `/tmp/rl_pipeline_review/rl_pipeline/train/export_bot.cpp`
- `/tmp/rl_pipeline_review/rl_pipeline/train/qnet.hpp`
- `/tmp/rl_pipeline_review/rl_pipeline/train/parity.cpp`

Do not reuse:

- cambc `sim/game.*`.
- cambc token layout.
- champion/camp/altar/ward/spell/draft logic.
- cambc reward shaping.
- cambc map generation.
- cambc C++ SIMD deployment target.

## 8. Current RL subtree and exact status

Files added under `rl/`:

- `rl/CMakeLists.txt`: builds native RL core, smoke tests, optional LibTorch model/trainer.
- `rl/README.md`: current workspace notes.
- `rl/sim/florent_actions.hpp/.cpp`: candidate enumeration, native `can_*` legality, action application.
- `rl/sim/florent_game.hpp/.cpp`: native match wrapper, critic encoding once per round, actor observation method, terminal reward helper.
- `rl/train/florent_layout.hpp`: dimensions, PPO constants, unit policies, terminal reward helper.
- `rl/train/policy_math.hpp`: masked log-softmax/entropy utility.
- `rl/train/model.hpp/.cpp`: minimal LibTorch `SpatialPolicy`; currently a simple flattened MLP, not the user-selected SWAR spatial network.
- `rl/train/trajectory.hpp`: basic trajectory/advantage scaffold.
- `rl/train/rollout.hpp`: random rollout scaffold.
- `rl/train/main.cpp`: table-style placeholder trainer; not the final PPO trainer.
- `rl/train/neural_train.cpp`: current observation-driven LibTorch smoke trainer; it samples legal actions from logits and runs an optimizer update, but is not yet correct PPO/GAE/critic training.
- `rl/bot/action_abi.py`: Python candidate/action ABI draft.
- `rl/bot/export.py`: currently copies checkpoint bytes and writes metadata; not a true quantized exporter.
- `rl/bot/policy.py`: deployment loader with safe wait fallback; does not run model weights.
- `rl/tests/action_smoke.cpp`: random first-legal native rollout.
- `rl/tests/rollout_smoke.cpp`: random rollout smoke test.

Build command:

```bash
cmake -S rl -B /tmp/florent_rl_build -DCMAKE_PREFIX_PATH=/home/hamza/libtorch
cmake --build /tmp/florent_rl_build -j2
```

LibTorch is installed at `/home/hamza/libtorch` and is CUDA-enabled. In this WSL environment CUDA initialization segfaulted through the NVIDIA PTX JIT. CPU-only runs work with:

```bash
CUDA_VISIBLE_DEVICES=-1 LD_LIBRARY_PATH=/home/hamza/libtorch/lib /tmp/florent_rl_build/neural_train maps/gen00.map26 1 /tmp/florent-real.pt
```

Latest successful smoke output:

```text
NEURAL TRAIN COMPLETE checkpoint=/tmp/florent-real.pt
```

This is **not** completion. It only proves the current minimal model can consume real encoded rows, sample legal native actions, update once, and save a checkpoint.

## 9. Critical known defects / unfinished work

These must be fixed, not merely documented:

### RL correctness

- Current neural trainer does not implement true PPO clipped objective.
- Current trainer does not store old log probabilities correctly per legal choice set.
- Current trainer does not calculate proper round-level GAE from a trained centralized critic.
- Current trainer does not train or use the critic.
- Current trainer uses one generic `H_WAIT` head for all unit types instead of separate unit-type architectures/heads.
- Current trainer handles spikes by waiting, not the intended heuristic fire/wait policy.
- Current trainer’s action candidate ABI is still too broad/map-dependent in places and needs fixed action union/window redesign.
- Current reward propagation is only a minimal terminal helper; implement terminal `{+1,0,-1}` at game end and correct team assignment.
- Current match wrapper lacks a robust turn cursor/round folding layer for batched PPO trajectories.
- Current critic encoder is team-0 oriented and needs team-frame handling or explicit team identity.
- Full 48x48 critic rows are large; do not blindly allocate cambc-sized queues. Use compact critic representation or tightly bounded batches.

### Deployment

- `bots/econ/main.py` calls the encoder but does not use learned logits/actions.
- `rl/bot/export.py` is not a real exporter.
- `rl/bot/policy.py` only validates metadata and returns wait; it does not execute weights.
- No Python SWAR/int4 network is implemented for the chosen 17x17 4-block 1x1+depthwise architecture.
- No quantized-vs-float parity checker exists.
- No CPU-budget guard with `ct.get_cpu_time_elapsed()` is integrated for learned inference.
- No learned-policy official match has been run.

### Parity/tests

- Existing observation parity passes, but action candidate/mask parity between C++ and official Python controller is missing.
- Need reachable-state C++/Python legal-mask parity.
- Need action decode parity.
- Need deterministic replay/trainer-vs-bot row parity.
- Need random legal rollouts over all map families, not only a few maps.
- Need checkpoint reload test.
- Need packaging test ensuring `main.py`, `encoder.py`, weights, and policy code are all in submission shape.

### Existing draft code concerns

- `rl/sim/florent_actions.cpp` currently enumerates board-sized spawn/fire/launcher destinations and has an old broad candidate model. Rewrite before treating it as final.
- `rl/train/model.hpp` currently uses flattened input MLP and 4096-output heads. This is a temporary smoke model, not the selected deployment architecture.
- `rl/train/neural_train.cpp` has a minimal update path and should be replaced/rewritten, not incrementally trusted as PPO.
- `rl/train/main.cpp` is an older table learner and should not be presented as the final trainer.
- `docs/rl-layout.md` contains old assumptions and must be corrected after validating engine semantics.

## 10. Correct implementation order from here

Do not stop after any single gate. Continue through this sequence:

1. **Finalize action ABI** against `fengine.cpp` legality semantics. Use fixed candidate tables, one actual executed-action probability per turn, and free heads only where engine semantics justify them.
2. **Implement `Match` turn cursor**: exact creation-order units, per-unit memory, one critic observation per team per round, round end only after all units, units created during round act next round.
3. **Implement action/mask parity tests** in C++ and Python against reachable native/official states.
4. **Implement compact actor/critic model**: selected 17x17 spatial encoder, separate unit-type policy models, centralized critic with team frame/identity.
5. **Implement real rollout buffers**: actor observations, candidate IDs, legal masks, chosen IDs, old logp, values, round/team/entity metadata, terminal reward.
6. **Implement correct GAE/PPO** using terminal-only reward and per-round critic timing. Add clipped ratio, entropy, value loss, KL guard, advantage normalization, checkpointing.
7. **Run a short PPO training smoke test** with finite loss, nonzero gradients, checkpoint reload, and action distribution changes.
8. **Implement Python float inference first** using exported weights and exact candidate decoder. Wire into `main.py` behind CPU guard with heuristic fallback.
9. **Implement quantization/SWAR int4 export** based on `research/nn_bench/` and `bots/bench/swar.py`. Add float-vs-quantized agreement test.
10. **Run official deployment tests** with `--tle 10`, timing/replay inspection, and compare learned bot against heuristic/starter.
11. Only after all above, report the system as complete.

## 11. Useful command checklist

From `/home/hamza/florent/florent_code`:

```bash
# Native engine
make -C engine

# Build RL tree
cmake -S rl -B /tmp/florent_rl_build -DCMAKE_PREFIX_PATH=/home/hamza/libtorch
cmake --build /tmp/florent_rl_build -j2

# Native action smoke
/tmp/florent_rl_build/action_smoke maps/gen00.map26
/tmp/florent_rl_build/action_smoke maps/pit01.map26
/tmp/florent_rl_build/action_smoke maps/s7_gen02.map26

# Random rollout
/tmp/florent_rl_build/rollout_smoke maps/gen00.map26 32

# CPU-only current neural smoke
CUDA_VISIBLE_DEVICES=-1 LD_LIBRARY_PATH=/home/hamza/libtorch/lib \
  /tmp/florent_rl_build/neural_train maps/gen00.map26 1 /tmp/florent-real.pt

# Python validation
python3 -I -m py_compile bots/econ/main.py bots/econ/encoder.py rl/bot/*.py

# Official runtime with TLE
.venv/bin/fcode run econ starter maps/gen00.map26 --seed 3 --tle 10
.venv/bin/fcode run econ starter maps/pit01.map26 --seed 3 --tle 10

# Observation parity
.venv/bin/python3 engine/tools/obs_parity.py --games 1 --rounds 20 --seed 7 --map maps/gen00.map26
```

## 12. Git/status/security

Current changes are uncommitted. Do not commit `.env`, credentials, checkpoints, or generated replay artifacts. The repository currently includes modified/untracked work in:

- `bots/econ/main.py`
- `bots/econ/encoder.py`
- `engine/Makefile`
- `engine/src/capi.cpp`
- `engine/src/obs.hpp`
- `engine/src/obs.cpp`
- `engine/tools/obs_parity.py`
- `docs/`
- `rl/`

If committing later, commit message must end with:

```text
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```

Do not treat the current RL smoke output as proof of completion. The next session must continue coding from §10, starting with the corrected action ABI and real PPO data flow.
