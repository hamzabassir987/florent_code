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
and `bots/econ/encoder.py`, checked by `engine/tools/obs_parity.py`.

- Report: one u32. Bit 31 marker (the heuristic packets above never set it),
  bit 30 removal, bits 26..29 category (1 Ti ore, 2 Vo ore, 3 enemy core,
  4 gunner, 5 sentinel, 6 launcher, 7 spike: enemy buildings), bits 0..10 tile.
- Each turn a unit: merges every marked report in the snapshot into its memory
  (ore into unknown terrain; enemy buildings added / removed; a reported core
  keeps its real id, team + 1), updates its memory from its own vision, queues
  the reportable facts it sees that the snapshot does not show, and writes the
  most important one (core, then turrets, then ore; lowest tile first) to slot
  (unit id + round) % 16. A queued report leaves the queue once a snapshot
  shows it, so an overwritten report is posted again.
- Reported (not seen) buildings have no direction and an unchanged attribute
  age: the policy can tell them from what the unit saw itself.
