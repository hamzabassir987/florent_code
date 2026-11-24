# Team-store report bus

The official runtime gives each unit its own Python `Player` state. Cross-unit
communication uses the 16-slot unsigned-32-bit team store. `write_store()` is
visible at the start of the next round; it is not an atomic same-round append.

## Slots

- `0`: own core position
- `1`: harvester count (core-owned; builders do not increment it)
- `2`: nearest threat position
- `3–5`: resource reports
- `6–8`: gunner/sentinel/spike/harvester reports
- `9–12`: builder sightings
- `13–14`: guard lease
- `15`: phase flags

## Packet

`pack_report(action, category, team, tile, round)` stores:

- bit 0: valid
- bits 1–2: action (`0` add/observe, `1` remove, `2` refresh)
- bits 3–5: category (Ti, Vo, gunner, sentinel, spike, harvester, builder)
- bit 6: team (`0` own, `1` enemy)
- bits 7–17: tile index (11 bits)
- bits 18–27: round modulo 1024

The receiver reads every report slot each invocation and merges facts into its
own local memory. Reports are intentionally idempotent. Same-round writers can
collide; a pending report remains queued and is retried on a later turn. Static
resources remain known once merged. Builder sightings are refreshed every three
rounds and are stale after three rounds. Conveyors are not broadcast.

A resource's symmetric counterpart is only a candidate until the receiver's
observed terrain validates the symmetry; no hidden map state is assumed.

## Learned-policy report bus

The learned policy (training and `bots/econ` learned turns) uses its own bus
on the same 16 slots, implemented identically in `ObsMemory` (engine/src/obs.cpp)
and `bots/econ/encoder.py`, checked by `engine/tools/obs_parity.py`. Every unit
takes part every turn: core, builders, gunners, sentinels, launchers and spikes.

Slot groups (the group says what a slot holds; one write per group per turn,
into slot base + (id + round) % size):

| slots  | group                    | content |
|--------|--------------------------|---------|
| 0..3   | our builders' positions  | tag = id % 63 (bits 11..16), round & 63 (bits 17..22); tag 63 = the core's status |
| 4..5   | enemy builder sightings  | tag, round as above; receivers forget them after 10 rounds |
| 6..9   | ore clusters             | bit 11 kind (0 Ti, 1 Vo), bits 12..26 the other cells of the 4x4 block |
| 10..13 | enemy attacking buildings| bits 11..13 core, gunner, sentinel, launcher, spike; bit 14 removal |
| 14..15 | our buildings            | bits 11..13 launcher, spike, gunner, sentinel, mine, harvester; bit 14 removal |

Every word: bit 31 marker, bit 30 spawn, bits 28..29 symmetry + 1 (0 unknown),
bits 0..10 tile. Core status (tag 63) bit 27 = our core is under attack (its HP
dropped or an enemy is in its view); actor scalar 15 = rounds since the last
alarm (cap 15).

- Send queues, one per group. A written item is checked against the next
  snapshot: delivered (any unit's write counts) -> due again after 4 rounds
  (builder groups) or 16 (ore / buildings); not there (overwritten) -> resent.
  Never-delivered items go first; enemy buildings nearest our core first; ties
  by a per-unit hash, so units cover different items. Ore / building refreshes
  only on rounds where (id + round) % 4 == 0.
- A builder posts its position when it moved 2+ tiles since its last delivered
  post, plus the refresh. Enemy builders in view are posted the same way.
- Spawn: the core's status word after a spawn carries bit 30; every unit then
  makes all its items due now (at most once per 4 rounds), so the newborn
  catches up. A builder spawned after round 40 writes nothing for its first
  4 rounds.
- Parameters: BusConfig in engine/src/obs.hpp, measured with
  rl/tests/bus_stats.cpp (24 games, 300 rounds): fresh ore known by the team in
  median 1 round, p90 2; late units know 82% of the team's ore when they appear,
  96% after 5 rounds, 99.6% after 40; own builders believed within 2 tiles 97%.
- Map symmetry: each unit keeps the symmetries (180 rotation, mirror x, mirror y)
  its known terrain does not contradict (and that do not map our core onto
  itself; a pit fixes it: a pit along x means mirror x, along y mirror y; a
  known enemy core tile must be the mirror of one of ours). Once one is left,
  every known tile is mirrored, and so is every tile
  learned later; the enemy core is placed on the mirror of ours where never seen.
- Reported or inferred buildings have no direction and an unchanged attribute
  age, inferred / reported terrain keeps "never seen": the policy can tell them
  from what the unit saw itself.
