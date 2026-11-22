"""Run a scripted scenario in the OFFICIAL engine and return its replay.

    from run_probe import run
    rep = run(name, W, H, core_a, ores, tasks_a, builders=2)

The map is made rotationally symmetric (team B mirrors team A's ore and does
nothing). Needs the official `fcode` CLI (FCODE env var, default /tmp/claude-0/venv/bin/fcode).
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
from genmap import encode  # noqa: E402
from replay import Replay  # noqa: E402

FCODE = os.environ.get("FCODE", "/tmp/claude-0/venv/bin/fcode")
WORK = os.environ.get("PROBE_WORK", "/tmp/claude-0/probes")


def run(name, W, H, core_a, ores=None, tasks_a=(), builders=1, walls=(), tasks_b=(), builders_b=0, turns_note=""):
    ores = ores or {}
    grid = [[0] * W for _ in range(H)]
    for (x, y), code in ores.items():
        grid[y][x] = code
        grid[H - 1 - y][W - 1 - x] = code
    for (x, y) in walls:
        grid[y][x] = 1
        grid[H - 1 - y][W - 1 - x] = 1
    core_b = (W - 2 - core_a[0], H - 2 - core_a[1])
    d = os.path.join(WORK, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    mp = os.path.join(d, "map.map26")
    open(mp, "wb").write(encode(W, H, grid, core_a, core_b, 0))
    bot = os.path.join(d, "bot")
    shutil.copytree(os.path.join(HERE, "bot"), bot)
    json.dump({"A": {"builders": builders, "tasks": [list(t) for t in tasks_a]},
               "B": {"builders": builders_b, "tasks": [list(t) for t in tasks_b]}},
              open(os.path.join(bot, "scenario.json"), "w"))
    rp = os.path.join(d, "replay.replay26")
    out = subprocess.run([FCODE, "run", bot, bot, mp, "--replay", rp, "--mark", "0"],
                         capture_output=True, text=True, cwd=d)
    if not os.path.exists(rp):
        raise RuntimeError(out.stdout + out.stderr)
    return Replay(rp), out.stderr
