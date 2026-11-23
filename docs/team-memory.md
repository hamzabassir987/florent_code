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
