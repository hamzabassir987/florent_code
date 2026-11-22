# Game rules as implemented by the native engine

Measured from 32 official replays (`tools/verify.py`); "doc" marks rules taken
from the game documentation because the replays never exercise them.

## Round
- Units (core, builder bots, turrets) act in **creation order**, both teams
  interleaved. A unit created this round acts from the next round.
- End of round, in order: resource distribution, passive income (+10 Ti to
  each team when `round % 4 == 3`), all cooldowns −1, store writes become
  visible, round + 1.
- A destroyed core decides the game, but the round still plays out (units
  after it act, resources still move). After round 1000: titanium delivered,
  then harvesters alive, then titanium stored, then a coin flip.

## Costs
- One cost multiplier per team: `scale = 100% + Σ increments of the team's
  living entities` (conveyor/splitter/barrier/mine +1, harvester +5,
  launcher/spike +10, builder/gunner/sentinel +20). Cost = `floor(base × scale)`.
- Attack 2 Ti, heal 1 Ti, gunner rotation 10 Ti.

## Builder bots
- One action **or** one move per round. Move: cardinal, onto a passable tile
  (no wall/void, no building except conveyor/splitter/mine, no other bot);
  move cooldown 1. Actions set action cooldown 1.
- Build/attack/heal/destroy target an orthogonally adjacent tile.
- Attack: 2 damage to the building on the tile; attacking an empty tile is
  legal and still costs 2 Ti.
- Heal: +4 (capped) to every own entity on the tile.
- Destroy: own building, free, no cooldown (doc).

## Turrets
- Gunner: line of 3 tiles (2 diagonally), blocked by anything in between,
  7 damage, 4 ammo, cooldown 1. Sentinel: line of 5 (4 diagonally), ignores
  obstacles, 18 damage, 10 ammo, cooldown 2. Firing at an empty tile is legal
  (ammo spent).
- A shot hits **one** entity on the tile: the builder bot if one stands there,
  else the building.
- Spike: every enemy on its 8 neighbouring tiles, one entity per tile (bot
  first), 12 damage, cooldown 4, no ammo. Not buildable within distance² 2 of
  the enemy core (doc).
- Launcher: throws a builder from one of its 8 neighbours to a tile within
  distance² 26; cooldown 1. Over a pit wider than 2 the bot falls (doc).
- Mine: the first enemy builder arriving on it takes 40 and the mine is gone.

## Core
- Spawns a builder on a tile within distance² 2 of its footprint; cooldown 1.
- Converts Voltium to ammo (×2), once per round, no cooldown.

## Resources
- Harvester: one stack every 4 rounds, the first on the round it is built.
- Conveyor: holds one stack, passes it to the tile in front, accepts from its
  other three sides. Splitter: from the back only, out front/left/right (doc).
- A whole chain advances one tile per round (a vacated tile is refilled).
- A receiver takes at most one stack per round, from the side it used least
  recently; a harvester outputs to its least-recently-used free neighbour.
  This matches 99.25% of the replays' resource moves; the rest are rare
  contention orders.
- A stack reaching a core (either team's) adds 10 to that team's titanium or
  Voltium and to its "collected" total.

## Verification status (32 replays)
- Every action in the replays is accepted by the engine.
- HP of every entity, every death, every team total: exact.
- Resource moves: 99.25% identical.
- Not exercised by the replays: splitters, falling into pits, economy mode,
  and actions the engine should *refuse* (the replays only show legal ones).
