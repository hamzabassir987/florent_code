# Handoff: Florent Code League project (read this first)

Context for a new Claude Code session continuing this work. Everything below
was done in one long earlier session (cloud container). The repository is
the source of truth: `hamzabassir987/florent_code`, branch
`claude/trusting-babbage-cv9nfu`.

---

## 0. How to work with this user

- **Very concise answers.** The user gets impatient with long explanations or
  with being shown many options at once. Answer exactly what was asked, short.
  When they ask "how fast is X", give the number.
- **Exactness matters.** For the engine, the user explicitly demanded 0%
  difference from the official engine ("it should be at 0%") and asked to
  consider edge cases, not rely on replays only. Don't settle for "close".
- Don't add things they didn't ask for (e.g. I once benchmarked a ResNet
  variant unprompted and they were confused: "when did I tell you to use
  resnet?").
- They make the design decisions (network shape etc.); give data, then do
  what they choose.
- They may hit usage limits mid-task; resume where things stopped.

---

## 1. The game (summary; full rules in `engine/RULES.md`)

"Florent Code League" (`fcode` on PyPI, v2.6.0): two teams on a symmetric
grid (8x8 to 30x30 per docs, but ladder maps go up to 44x24 and 36x36). Each
team writes a Python `class Player` with `run(self, ct)` in `main.py`; `run`
is called once per round for **every living unit** (core, builder bots,
turrets). 1000 rounds.

- Win: destroy enemy core (500 HP), else tiebreaks at round 1000: titanium
  delivered to core → harvesters alive → titanium stored → coin flip.
- Resources: titanium (Ti; start 500, +10 every 4 rounds) pays for everything;
  Voltium only converts to ammo (2 ammo per Voltium, core `convert_ammo`).
  Harvesters on ore output a stack of 10 every 4 rounds; stacks travel along
  conveyors to the core.
- Units: builder bots (only mobile unit, 1 move OR 1 action per round, cardinal
  moves, everything targets orthogonally adjacent tiles), gunners (3-tile line,
  7 dmg, 4 ammo), sentinels (5-tile line, ignores obstacles, 18 dmg, 10 ammo),
  launchers (throw builders), spikes (12 dmg to all 8 neighbours, no ammo),
  mines (hidden from enemy, 40 dmg). Buildings: conveyor, splitter, harvester,
  barrier.
- Costs: `floor(base × scale)` with one per-team scale (sum of increments of
  living entities).
- **Fog of war**: each unit sees only within its vision radius² (core 36,
  builder 20, gunner 13, sentinel 32, launcher 26, spike 8). Querying outside
  vision raises GameError. Ore/walls/enemy core are unknown until seen.
- **CPU**: 10 ms per unit per turn plus a banked buffer of up to 5% (0.5 ms).
  Over budget → turn interrupted (unit survives, restarts fresh next round).
  Uncaught exception → unit destroyed permanently.
- Shared team memory: 16 int store slots, writes visible next round.
- Bot runs in a Python sub-interpreter: **numpy is not usable** (fails to load
  twice in one process; the server likely doesn't have it anyway). Pure Python
  only.
- Server drops printed stdout from replays (local runs keep it).

The original full API/spec text was pasted by the user at the start of the
earlier session; `engine/RULES.md` has everything verified, and the API stubs
are in the installed package (`fcode/_types.py`).

---

## 2. Repository layout

```
bot.zip                 submission zip of bots/econ (main.py at root)
bench.zip               CPU benchmark submission (bots/bench)
fcode.toml              fcode project config (bots_dir=bots, maps_dir=maps)
maps/                   28 generated test maps (gen*, pit*, s7_*)
tools/genmap.py         random symmetric map generator (+ pit maps); `encode()` writes .map26
tools/readreplay.py     prints per-unit CPU time per turn from a replay
bots/econ/              MY CURRENT BOT (handwritten heuristic, see §4)
bots/v1, bots/v2        older snapshots of econ (for A/B tests)
bots/starter/           official starter bot
bots/chaos/             random/probe bot (copy of engine/probe/chaos)
bots/bench/             server CPU benchmark bot (SWAR conv timing)
engine/                 NATIVE C++ ENGINE (exact clone of the official engine)
  RULES.md              all verified rules, incl. exact distribution algorithm
  src/fengine.hpp/.cpp  the engine (Game class)
  src/rustrng.hpp       bit-exact Rust rand 0.8 StdRng (ChaCha12) port
  src/capi.cpp          C API (ctypes) used by the verifier
  Makefile              builds build/libfengine.so
  examples/random_game.cpp  speed benchmark (~500 random games/s, 1 thread)
  tools/replay.py       .replay26 decoder (no deps)
  tools/verify.py       replay verifier (see §3)
  tools/rustrand.py     Python port of the Rust StdRng
  tools/seedsearch.cpp  recover a replay's 32-bit seed from its first 2 entity ids
  probe/                bots run in the OFFICIAL engine to measure rules:
    run_probe.py, bot/  scripted scenario runner (custom map + build list)
    chaos/              random play + logs can_* answers ("Q ...") and vision ("V ...")
    pitthrow/           launcher throws across pits of width 2/3/5
    minetest/           mines: spawn-onto, walk-onto, enemy build checks
    unitcap/            reaches the 50-unit cap (uses econ.py copy)
research/nn_bench/      pure-Python NN speed benchmarks (SWAR), see §5
research/reverse_engineering/  gdb tracing scripts + replay-generation scripts
                        (these reference /tmp/claude-0 paths from the old container; adjust)
```

Not in the repo (were in the old container's /tmp, gone): the user's 32 ladder
replays (`test.zip`, user has it), the RL pipeline zip (`rl_pipeline.zip`, user
has it), ~100 locally generated official replays (regenerate with `fcode run
--seed N --replay X_s<N>.replay26`), recovered ladder seeds (rerun seedsearch).

Setup on a new machine:
```
pip install fcode            # official CLI + engine (Linux x86 wheel was used)
cd engine && make            # builds build/libfengine.so
fcode run econ starter maps/gen00.map26 --seed 3 --replay check_s3.replay26
python3 engine/tools/verify.py check_s3.replay26   # expect 0 mismatches
```
`fcode` needs `fcode.toml` in cwd (repo root has it). Official maps need
`fcode login` (we never had credentials), so all maps were generated locally.

---

## 3. The native engine (DONE, exact)

Goal (user's): a fast native engine for RL self-play that reproduces the
official engine **exactly**. Status: achieved and verified.

### How it was made exact
The official engine is a compiled Rust library
(`fcode/fcode_engine.cpython-313-x86_64-linux-gnu.so`, built from a private
repo `battlecode-platform/engine/rust/...`, no source available anywhere). It
still has symbol names. Methods used:
1. **Replays** (`tools/verify.py`): rebuild every action from a replay's events,
   run them through the engine, compare HP, deaths, ids, resource moves (incl.
   stack ids), team totals each round, winner.
2. **Disassembly + gdb tracing** of `Game::distribute_resources` and its
   `edge_priority` (objdump; gdb Python breakpoints logging heap push/pop).
3. **Probe bots in the official engine** that log `can_*` answers and vision
   answers; the verifier re-asks the native engine in the identical state.

### Key findings (all in RULES.md)
- Turn order = creation order (both teams interleaved).
- Resource distribution = max-heap of edges (source→target), priority
  `i32::MAX` if source has 1 edge and target 1 source, else −(round edge last
  used); plus uniform [0,1) tie-break from `StdRng(seed)`. Waiting edges
  re-pushed when a tile empties. Grid scan row-major; pushes in target (x,y)
  order. Harvester cooldown 4.
- RNG: `StdRng::seed_from_u64(seed)` for distribution tie-breaks and the coin
  flip (`gen::<bool>()`, top bit of next_u32, true = A);
  `StdRng::seed_from_u64(seed ^ 0x1D5EED00C0DEF00D)` for entity AND stack ids:
  `gen_range(3..1e9)`, redrawn if the id was EVER issued (never reused).
- Legality details found by probing: builder attack needs a building; only
  conveyor/splitter can be built under a builder bot; launcher can't target
  void; throw across pit >2 wide is legal and the bot falls; gunner rotation
  must change direction; shots hit the bot on a tile before the building;
  spike hits one entity per tile (bot first); enemy can't build on a hidden
  mine's tile; spawning onto an enemy mine triggers it; unit cap 50; the map's
  "economy" game-mode flag has no effect.
- Vision: core radius from nearest footprint tile; lists row-major, building
  then bot, self included; enemy mines invisible; `is_tile_passable` false
  under any bot.
- Ladder seeds are 32-bit; recover with `seedsearch ID1 ID2 0 4294967296`
  (~4 min each, 4 cores). Name replays `*_s<seed>.replay26` so verify.py uses
  the seed.

### Verification totals (all zero differences)
- 32/32 user ladder replays (seeds recovered): 1.04 M moves, 10,846 ids,
  4.6 M HP checks, 1,041 deaths, all totals, 32 winners.
- ~80 locally generated official games (bot pairings, pit/wide-pit maps,
  chaos, probes): 6+ M legality answers, 1.3 M resource moves, all ids/HP/
  totals/winners; vision 1.63 M answers.

### Engine API (C++ `fe::Game`, C API in capi.cpp)
`load_map`, `reset(seed)`, `turn_order()`, actions (`move, build, attack,
heal, destroy, self_destruct, spawn, convert_ammo, fire, rotate, launch,
write_store, resign`), checks (`can_*`), `end_round()`, state (`ents`, `bld`,
`bot`, `teams`, `env`, `round`, `winner`, `over`), vision (`in_vision`,
`nearby_tiles`, `nearby_ids`, `tile_building`, `tile_empty`,
`tile_passable`). Speed: ~500 full random games/s single thread.

Store (`write_store`) is implemented trivially (buffered); not verified (no
rule depends on it).

---

## 4. Current submission bot: `bots/econ/main.py` (= bot.zip)

Handwritten heuristic. Beat the starter 10/10 locally and earlier versions
(v1, v2) clearly. Main ideas:
- Builders keep per-unit map memory, claim ore via store slots, build a
  harvester only if its whole conveyor chain is affordable, chain planned by
  Dijkstra into the core or into an existing chain that reaches the core.
- **Conveyor throughput is ~1 stack/round**, so trunks are limited to
  `MAX_TRUNK = 3` harvesters each (big income gain).
- Spawn rate `want = 4 + harvesters//2` (tuned).
- Siege from round ≥150 with ≥500 Ti: idle builders find the enemy core via
  map symmetry, ring it with gunners, heal them, attack the core.
- Danger-aware pathing, core healing, a home guard laying corner mines.
- Worst CPU turn measured ~2.7 ms.

Not yet tuned: siege trigger timing (a test was interrupted).

---

## 5. Neural network plan and measured speeds (pure Python)

The user wants an RL-trained policy. Pure Python only, 10 ms/turn.

**SWAR big-integer trick** (research/nn_bench): pack a whole grid channel
into one Python int (16-bit slots with 4-bit weights/activations, or 32-bit),
so one int op processes all cells. Bit-exact vs a naive reference. Cost scales
with number of int ops (≈ channels² × kernel), NOT grid area.

Measured (this container; the server looked about equally fast):
- normal 3x3 conv 26→26 on 30x30: 8.2 ms/layer; 14x14: ~2.2 ms/layer.
- 13x13, 26ch: 2/3/4 layers = 3.6/5.6/7.4 ms. 14x14: 4.5/6.7/9.1 ms.
- 15x15, 32ch, 3 layers: normal 8.0 ms; 1x1+depthwise3x3: 1.5 ms.
- ResNet (2 layers per block, 1x1+dw3x3, 32ch): 3 blocks 15x15 3.3 ms;
  20x20 5.2 ms; 17x17 4 blocks 5.4 ms (36ch: 6.3 ms).
- **Cropping** each layer to the region the centre still needs: 17x17, 4 blocks,
  32ch → **4.4 ms** (centre output identical).
- Depthwise 5x5 variant (2 blocks): 3.6 ms; user chose to keep 3x3.
- Tried and rejected: whole-layer-as-one-multiply (Kronecker): 10-40x slower;
  stacking output channels in one int: 10x slower.

**Chosen design (user's decision):** input = 17×17 patch centred on the unit,
filled from the unit's map memory (with an "unknown" channel), plus distance
features for things outside the patch (precomputed BFS distance maps,
recomputed only when things change); network = 4 ResNet blocks of
(1x1 conv → depthwise 3x3 → ReLU) with skips, 32 channels, cropped, read the
centre logits. Receptive field 17x17. ~4.4 ms. Must be trained with 4-bit
quantisation in mind. Run only when the turn has CPU left
(`ct.get_cpu_time_elapsed()`), fall back to heuristics otherwise.
MLP for scoring candidate decisions was also discussed as an option.

Server CPU: `bench.zip` measured; one server replay showed the server about as
fast as the container (a setup step took 11.3 ms there vs 12.7 ms locally).

---

## 6. RL pipeline (user's zip, to port)

`rl_pipeline.zip` (user has it): a C++/libtorch PPO self-play trainer built
for Battlecode 2026a "cambc" (`README_FIRST.md`, `train/README.md`,
`train/NEXT_RUN.md`). Pieces: native engine wrapper (`sim/`), `train/layout.hpp`
(inputs/heads), `train/model.cpp` (encoder `begin_turn`, legal-action filter
`plan_turn`, `finish_turn`, rewards), `train/main.cpp` (PPO, GAE, eval,
distillation), `export_bot.cpp` (int8 SIMD export), parity tools. Lessons in
its README (parity is everything; path distances not straight-line; train the
student on the degraded inputs the bot can afford).

Porting gaps identified:
1. Engine: DONE (this repo's `engine/`, replaces their wasm2c engine).
2. Deployed bot is pure Python with the SWAR int4 network (their export
   targets C++ int8 SIMD): need a new exporter + Python inference code with a
   parity checker.
3. Inputs/heads: their policy is token-based; ours is the 17x17 patch + distance
   features + MLP head. Heads needed: builder (move N/E/S/W/stay, build type ×
   tile × direction, attack, heal, destroy, mine), core (spawn tile, convert),
   turrets (target), launcher (bot, target). Legal-action masking via the
   engine's `can_*`.
4. No GPU / no PyTorch in the old container; training runs on the user's
   machine.

---

## 7. Next steps (where we stopped)

The engine is complete (rules + randomness + vision, all exact). Next:
1. Observation encoder from the engine state honouring fog: 17x17 patch from
   per-unit memory + unknown channel + distance features; must be computable
   identically in the Python bot (parity tests).
2. Action heads + legal-action filter.
3. Plug into the PPO loop from the pipeline (C++ trainer linking fengine).
4. Quantised (4-bit) training/distillation, exporter to the Python SWAR
   format, parity checker (bot vs trainer), CPU budget guard.
5. Optional: tune the heuristic bot's siege timing; keep `bot.zip` as fallback.

---

## 8. Gotchas

- `fcode run` with `--tle 0` (default locally) does not enforce the 10 ms
  limit; use `--tle 10` to test timing.
- Replay event order: an action's effects (hp, remove) are logged BEFORE the
  action event; spike fire logs one event per tile; rotations appear as a
  `place` with an existing id; launches as a long `move` plus launcher acd.
- verify.py takes the seed from the filename `_s<seed>.replay26`; without it,
  only seed-independent checks are exact.
- Probe bots must ask `can_*` questions AFTER acting in a turn (the verifier
  checks at the end of the unit's turn), otherwise you get false
  "cooldown"/"dead" differences.
- My econ bot can kill an idle opponent's core quickly (siege); use pit maps
  when you need long games against idle bots.
