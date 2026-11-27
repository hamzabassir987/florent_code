# Action masks and scripted units

Masks are applied on top of the engine's own legality checks (`can_*`).
Implemented identically in `rl/sim/florent_heads.cpp` (training) and
`bots/econ/heads.py` (deployed bot); `rl/tests/heads_parity.py` checks that
both make the same decisions.

## In place

| Head / unit | Rule | Why |
|---|---|---|
| builder attack | only a tile holding a **visible enemy building**, or a **hidden enemy mine** on the conveyor plan (detected: `can_fire` legal on a tile with no visible building); builders never step on a detected mine | a builder attack damages the building on the tile (never a builder); the engine allows hitting our own. Before: 92-98% of attacks hit our own side |
| gunners | **removed**: never buildable (bot and trainer); the gunner policy is unused | |
| sentinel gate (separate network) | when a builder can build a sentinel aimed at a known enemy core tile (no cap), the gate network (the teacher's P_SENTINEL actor, on the builder's observation; HS_MODE: 0 no, 1 place) decides first; "place" builds it, "no" lets the builder heads decide (they never offer a sentinel). Trainer: gate samples are carried over iterations until GATE_MIN (256) or GATE_EVERY (4). Student: merged into the builder network as 2 extra logits after the builder heads (one forward per turn). A sentinel fires only at a visible enemy core tile on its line | 18 damage / 2 rounds at the core, not blocked |
| gunner rotate | only to a direction whose line (3 straight, 2 diagonal) first meets a visible enemy; walls and own units block | a rotation costs 10 Ti |
| builder build: conveyor / splitter | only on a tile of the conveyor planner's path, facing its planned direction | routes are planned (BFS from the core over remembered tiles) |
| conveyor planner | at most **4 harvesters per line** into the core; a full line is closed and later ore is routed to another core side | a line moves 1 stack/round, a harvester makes 1 per 4 rounds |
| builder build: launcher | at most **3 own launchers** (from the unit's memory) | |
| core convert | **always convert all Voltium** (100%) when legal; a single option, not recorded | Voltium has no other use: no price, no tie-break. Ammo is ready as soon as a gunner / sentinel exists |
| builder: launcher at a pit (scripted) | on a map whose pit is 1-2 wide, a builder next to a free tile on the pit's edge builds a launcher there when the team knows no own launcher; nothing recorded | a thrown builder crosses a pit of width <= 2 alive (checked in the official engine) |
| pit launcher | a launcher on the edge of a pit 1-2 wide only throws our own builder (first in neighbour order) across the pit to any legal tile there: the network decides launch or wait and where to land (a fixed landing tile was mined by the enemy) | |
| heal | engine rule only (own units / buildings that lost HP) | with no friendly fire, heals only repair enemy damage |
| barriers, mines | **no restriction** (keep normal) | walls around the base and mines are good play |
| spike (scripted, no network) | **on its first turn it fires once anyway**; afterwards only when a visible enemy (builder or building) is on one of its 8 neighbours; a shot hits every enemy around, hidden mines included | a spike shot hits every enemy building around it, hidden enemy mines included (5 HP vs 12 damage), so a spike can be placed to clear mines; otherwise the 4-round cooldown is kept for real targets |
| gunner / sentinel surgery | tool: `surgery IN.pt OUT.pt` (rl/tools/surgery.cpp) copies the spike row of `builder_build_type` to gunner / sentinel and zeroes the gunner / sentinel policies' output layers (uniform heads) | run once on a checkpoint, then resume from OUT.pt |
| single-choice heads (trainer) | a head with only one legal option is not recorded | no gradient; it diluted the loss and the logs |

## Discussed, not implemented yet

| Idea | Rule | Notes |
|---|---|---|
| launcher over a wide pit (still open) | own builder: no landing across a pit wider than 2 tiles (it dies, `pit_w > 2`); enemy builder: allowed (a kill) | checked on a 2-wide pit (eval_8150): the throw is accepted by the official engine and the builder survives |
| spawn cap | e.g. at most ~1.5 builders per harvester, or a fixed maximum | each builder adds +20% to every price; overspawning is invisible to the Ti-collected win condition |
| greedy anti-stuck | when a builder oscillates N turns, take its second-best action once | greedy (69% vs sampled) sometimes collapses on a map |
| pit maps | drop pit maps from training (`pit_share 0`, map list without `*pit*`); scripted pit play: farm own half, launcher raids across 2-wide pits | |
| low-HP curriculum | `hp_scale` / `core_hp_scale` knobs, ramped back to 1.0 | makes attacks pay off early |
