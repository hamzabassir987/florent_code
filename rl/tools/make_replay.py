"""Logged greedy self-play game -> official .replay26.

    python rl/tools/make_replay.py RUN/replays/iter_000050.game [OUT.replay26]
    python rl/tools/make_replay.py RUN/replays            # every .game without a replay

Runs the official `fcode run playback playback MAP --seed S` with the playback
bot re-issuing the logged actions (FLORENT_PLAYBACK), then checks the official
result against the log: same winner, no failed action.
Needs the fcode CLI (FCODE env var, default `fcode`), run from the repo root.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "engine", "tools"))
from replay import Replay  # noqa: E402


def header(path):
    h = {}
    with open(path) as f:
        f.readline()
        for line in f:
            p = line.split()
            if p[0] == "actions":
                h["actions"] = int(p[1])
                break
            h[p[0]] = p[1:]
    return h


def convert(game, out=None):
    out = out or game[:-len(".game")] + ".replay26"
    h = header(game)
    env = dict(os.environ, FLORENT_PLAYBACK=os.path.abspath(game))
    cmd = [os.environ.get("FCODE", "fcode"), "run", "playback", "playback", h["map"][0],
           "--seed", h["seed"][0], "--tle", "0", "--mark", "0", "--replay", out]
    subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if not os.path.exists(out):
        print(f"{game}: FAILED, no replay ({' '.join(cmd)})")
        return False
    r = Replay(out)
    fails = sum(1 for t in r.turns for u in t if u[0] == "output" and "PBFAIL" in (u[4] or ""))
    want = int(h["winner"][0])
    got = getattr(r, "winner", None)
    ok = fails == 0 and got == want
    print(f"{out}: {'OK' if ok else 'MISMATCH'} winner log {want} ({' '.join(h['winner'][1:])}) "
          f"official {got} ({getattr(r, 'win_condition', '')}) rounds {h['rounds'][0]} failed_actions {fails}")
    return ok


def main():
    a = sys.argv[1]
    if os.path.isdir(a):
        games = sorted(os.path.join(a, f) for f in os.listdir(a) if f.endswith(".game"))
        ok = all([convert(g) for g in games if not os.path.exists(g[:-5] + ".replay26")])
    else:
        ok = convert(a, sys.argv[2] if len(sys.argv) > 2 else None)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
