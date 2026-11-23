# Florent RL Layout

This document defines the Florent-specific adapter layered on top of the supplied PPO pipeline. The pipeline's rollout, GAE, PPO, entropy, checkpoint, and distillation code are retained; game-specific cambc layout and actions are not.

## Observation inputs

### Actor

The actor receives the production-reproducible observation from `bots/econ/encoder.py`:

- 17x17 local patch;
- terrain and remembered terrain;
- remembered dynamic entity types;
- tile and attribute staleness;
- last observed builder visit;
- visible direction and HP;
- possible and confirmed turret reach;
- raw edge-ray summaries;
- actor scalar features.

The C++ trainer calls the matching encoder in `engine/src/obs.cpp`. Actor observations never read omniscient state.

### Critic

The centralized critic receives one omniscient global observation per round, not one critic observation per unit turn. It may use the full board, all entities, resources, mines, and exact engine state.

## Policy architecture

There is one policy trunk and action head set per acting unit type:

- core;
- builder;
- gunner;
- sentinel;
- launcher.

Spike initially uses a deterministic heuristic fire/wait policy. Harvester, conveyor, splitter, barrier, and mine have no turns and therefore no actor policy.

The spatial actor patch is processed by a shared local convolutional encoder within each unit-type model: **four residual blocks, each consisting of a 1x1 pointwise layer followed by a depthwise 3x3 layer**, with cropped-radius evaluation. Scalar features are concatenated before the action heads. Models are not shared across unit types.

## Candidate ordering

All candidate lists are fixed, deterministic, and shared by trainer and production decoder.

- Directions: `C, N, NE, E, SE, S, SW, W, NW` where `C` means no movement/rotation direction.
- Neighbor offsets: row-major `(dy, dx)` over `[-1, 1] x [-1, 1]`, excluding `(0, 0)`.
- Local target offsets: row-major over the engine-visible target window.
- Building types: `CONVEYOR, SPLITTER, HARVESTER, GUNNER, SENTINEL, LAUNCHER, SPIKE, BARRIER, MINE`.
- Conversion amounts: fixed small discrete set `{1, 2, 4, 8, 16}`; illegal amounts are masked by `can_convert`.

Candidate ordering is part of the checkpoint ABI and must not depend on map contents.

## Core heads

The core model emits:

1. `spawn_tile`: one logit for every board tile in row-major order;
2. `convert_amount`: one logit for each fixed amount `{1,2,4,8,16}`;
3. `wait`: one logit.

The legal mask keeps only tiles accepted by `can_spawn`, amounts accepted by `can_convert`, and always keeps wait. The sampled action is exactly one of spawn, conversion, or wait.

## Builder heads

The builder model emits:

1. `move_direction`: eight movement directions;
2. `build_type`: nine building types;
3. `build_offset`: adjacent construction offsets and facing direction where required;
4. `attack_offset`;
5. `heal_offset`;
6. `destroy_offset`;
7. `wait`.

Building a conveyor or splitter is allowed only when the selected adjacent cell is marked by the planner as a valid planned-path cell. The planner supplies an annotation/mask; it is not a macro action and does not control builder movement. Builders remain free to move anywhere legal.

Harvester construction is allowed only on a known resource tile and only when the engine's legal check succeeds. A visible harvester masks the build-harvester choice on that resource, while the delivery/path annotation remains active.

The decoder evaluates legal combinations with the engine's `can_*` methods. Illegal combinations are masked before sampling; the policy never relies on a post-sample repair.

## Gunner heads

The gunner model emits:

1. `fire_target`: legal visible target cells in the fixed local target ordering;
2. `rotate_direction`: eight directions;
3. `wait`.

`fire_target` candidates are filtered by `can_fire`; rotation candidates by `can_rotate`.

## Sentinel heads

The sentinel model emits:

1. `fire_target`;
2. `wait`.

Targets are filtered by `can_fire`.

## Launcher heads

The launcher model emits:

1. `builder_target`: friendly adjacent builders in fixed local ordering;
2. `destination_tile`: board tiles in row-major order;
3. `wait`.

A launch choice is legal only when the pair passes `can_launch`. Invalid builder/destination combinations are masked jointly. Launcher remains a normal acting unit and has its own model.

## Choice-set PPO interface

Each acting unit produces one categorical distribution over its currently legal flattened choices. The rollout stores:

- unit type and stable entity id;
- round and turn index;
- observation version;
- flattened legal candidate ids;
- selected candidate id;
- old log probability;
- old value estimate;
- reward and terminal flags.

The model's logits are never renumbered based on the legal set. Legal filtering selects a subset of the fixed candidate ABI, preserving checkpoint compatibility.

## Round/critic sequencing

For each round:

1. capture the centralized critic observation once;
2. enumerate the exact engine turn order;
3. for each living acting unit, encode its actor observation and legal choices;
4. sample/decode one legal action and apply it to the native engine;
5. finish the round in the native engine;
6. compute round reward and next centralized value.

Units created during a round act starting in the next round, matching the native engine.

## Parity requirements

Before PPO training, the adapter must pass:

- C++/Python actor observation parity;
- C++/Python candidate ordering parity;
- legal-mask parity for random engine states;
- action decode parity for every head;
- random legal-policy native-engine rollout;
- deterministic replay under fixed map and seed.
