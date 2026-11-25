# Action masks and scripted units

Masks are applied on top of the engine's own legality checks (`can_*`).
Implemented identically in `rl/sim/florent_heads.cpp` (training) and
`bots/econ/heads.py` (deployed bot); `rl/tests/heads_parity.py` checks that
both make the same decisions.

## In place

| Head / unit | Rule | Why |
|---|---|---|
| builder attack | only a tile holding a **visible enemy building** | a builder attack damages the building on the tile (never a builder); the engine allows hitting our own. Before: 92-98% of attacks hit our own side |
| gunner / sentinel fire | only where the unit the shot hits is a **visible enemy** (the builder on the tile first, else the building) | no friendly fire, no wasted ammo |
| gunner rotate | only to a direction whose line (3 straight, 2 diagonal) first meets a visible enemy; walls and own units block | a rotation costs 10 Ti |
| builder build: conveyor / splitter | only on a tile of the conveyor planner's path, facing its planned direction | routes are planned (BFS from the core over remembered tiles) |
| conveyor planner | at most **4 harvesters per line** into the core; a full line is closed and later ore is routed to another core side | a line moves 1 stack/round, a harvester makes 1 per 4 rounds |
| builder build: launcher | at most **3 own launchers** (from the unit's memory) | |
| heal | engine rule only (own units / buildings that lost HP) | with no friendly fire, heals only repair enemy damage |
| barriers, mines | **no restriction** (keep normal) | walls around the base and mines are good play |
| spike (scripted, no network) | fires at the first adjacent visible enemy; **on its first turn it fires once anyway** | a spike shot hits every enemy building around it, hidden enemy mines included (5 HP vs 12 damage), so a spike can be placed to clear mines; otherwise the 4-round cooldown is kept for real targets |
| single-choice heads (trainer) | a head with only one legal option is not recorded | no gradient; it diluted the loss and the logs |

## Discussed, not implemented yet

| Idea | Rule | Notes |
|---|---|---|
| launcher over a wide pit | own builder: no landing across a pit wider than 2 tiles (it dies, `pit_w > 2`); enemy builder: allowed (a kill) | checked on a 2-wide pit (eval_8150): the throw is accepted by the official engine and the builder survives |
| core convert | only when we own a gunner or sentinel (the only ammo users) | the policy mostly stopped converting by itself (0 in some later games) |
| spawn cap | e.g. at most ~1.5 builders per harvester, or a fixed maximum | each builder adds +20% to every price; overspawning is invisible to the Ti-collected win condition |
| gunner / sentinel | copy the spike row of `builder_build_type` to gunner / sentinel (weight surgery), heuristic build direction (face the nearest enemy), convert heuristic for ammo | they cost +20% scale and need ammo |
| greedy anti-stuck | when a builder oscillates N turns, take its second-best action once | greedy (69% vs sampled) sometimes collapses on a map |
| pit maps | drop pit maps from training (`pit_share 0`, map list without `*pit*`); scripted pit play: farm own half, launcher raids across 2-wide pits | |
| low-HP curriculum | `hp_scale` / `core_hp_scale` knobs, ramped back to 1.0 | makes attacks pay off early |
