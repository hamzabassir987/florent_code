# Game rules as implemented by the native engine

The engine reproduces the official engine exactly. Rules come from three
sources, all cross-checked:

1. the official engine's binary (`fcode_engine*.so` still carries symbol
   names; the resource distribution was read from `Game::distribute_resources`
   and `edge_priority`, and traced live with gdb);
2. official replays (`tools/verify.py` re-runs every action and compares the
   full outcome);
3. a chaos probe bot (`probe/chaos`) run in the official engine, which logs
   millions of `can_*` answers that the verifier re-asks the native engine in
   the identical state.

## Verification status

55 games played by the official engine (normal and pit maps, 6 bot pairings,
known seeds): **zero differences** in

- 3.37 M legality answers (`can_move`, `can_build` for every type,
  `can_heal`, `can_destroy`, builder attack, `can_spawn`, `can_convert_ammo`,
  turret `can_fire`, `can_rotate`, `can_launch`), asked both before and after
  the unit acts in its turn;
- 1.07 M resource moves, including every stack id;
- every entity id, every HP value every round, every death;
- titanium, Voltium, ammo and collected totals every round;
- the winner and the win condition.

## Randomness

- Two Rust `rand 0.8` `StdRng` (ChaCha12) generators, ported bit for bit
  (`src/rustrng.hpp`): `StdRng::seed_from_u64(seed)` for distribution
  tie-breaks, `StdRng::seed_from_u64(seed ^ 0x1D5EED00C0DEF00D)` for ids.
- Entity and resource-stack ids: `gen_range(3..1_000_000_000)`, redrawn while
  the id was ever issued before (ids are never reused). Cores are ids 1 and 2.
- Ladder seeds are 32-bit; `tools` can recover a replay's seed from its first
  two entity ids by brute force (about 4 minutes).

## Round

- Units (core, builder bots, turrets) act in creation order, both teams
  interleaved. A unit created this round acts from the next round.
- End of round: resource distribution, passive income (+10 Ti each team when
  `round % 4 == 3`), all cooldowns −1 (harvester cooldowns too), store
  writes become visible, round + 1.
- A destroyed core decides the game; the round still plays out. After round
  1000: titanium delivered, then harvesters alive, then titanium stored, then
  a coin flip: `gen::<bool>()` on the distribution generator (top bit of its
  next `next_u32`), true = team A (24/24 seeds checked).
- The map's "economy" game-mode flag has no effect on the rules (identical
  games with and without it).
- At most 50 units per team (core, builders, turrets): spawns and unit
  builds are refused at the cap (checked holding the cap for 600 rounds).

## Costs

- One multiplier per team: `scale = 100% + Σ increments of the team's living
  entities` (conveyor/splitter/barrier/mine +1, harvester +5, launcher/spike
  +10, builder/gunner/sentinel +20); cost = `floor(base × scale)`.
- Builder attack 2 Ti, heal 1 Ti, gunner rotation 10 Ti.

## Builder bots

- One action or one move per round. Move: cardinal, onto a passable tile (no
  wall/void; no building except conveyor/splitter/mine; no other bot).
- Build / attack / heal / destroy target an orthogonally adjacent tile.
- Only a conveyor or splitter can be built under a builder bot.
- Attack: needs a building on the tile; 2 damage.
- Heal: needs a damaged own entity on the tile; +4 (capped) to every own
  entity there.
- Destroy: own building, free, no cooldown.

## Turrets

- Gunner: line of 3 tiles (2 diagonally), blocked by anything in between,
  7 damage, 4 ammo, cooldown 1. Sentinel: line of 5 (4 diagonally), ignores
  obstacles, 18 damage, 10 ammo, cooldown 2. Firing at an empty tile is
  legal (ammo spent). Rotation must change the direction.
- A shot hits one entity on the tile: the builder bot if one stands there,
  else the building.
- Spike: every enemy on its 8 neighbouring tiles, one entity per tile (bot
  first), 12 damage, cooldown 4. Not buildable within distance² 2 of the
  enemy core.
- Launcher: throws a builder from one of its 8 neighbours to a tile within
  distance² 26 that is not wall or void; cooldown 1.
- Mine: the first enemy builder arriving on it (walking, thrown, or spawned
  by its core onto it) takes 40 and the mine is gone. To the enemy the tile
  looks empty and passable and the mine is not listed, but the enemy still
  cannot build on that tile.

## Resources (exact official algorithm)

Each round, after all units acted:

1. Scan tiles row by row. Every building lists its output tiles (conveyor:
   front; splitter: front/left/right; harvester: its 4 neighbours) that are in
   bounds, on its side of a pit and hold a building; each edge is recorded
   under its target (target → sources, in scan order) and counted for its
   source. Sources: conveyors/splitters holding a stack, harvesters whose
   cooldown is 0.
2. Targets in (x, y) order: if the target can take a stack now (a core, or an
   empty conveyor/splitter), push each source's edge it accepts (conveyor: not
   from its front; splitter: only from its back; core: any) into a max-heap.
   Priority = `i32::MAX` if the source has one edge and the target one source,
   else −(round this edge last moved a stack, 0 if never), plus a uniform
   [0, 1) tie-break.
3. Pop the best edge. If its source still has its stack and its target can
   still take one, the stack moves (a harvester draws a new stack id and goes
   on cooldown 4), the edge's round is recorded, and the edges of sources
   waiting on the tile just vacated are pushed.
4. A stack reaching a core (either team's) adds 10 to that team's titanium or
   Voltium and to its collected total.
