"""Benchmark submission: plays the normal econ bot, and the core additionally
times 3x3 conv layers (26 -> N channels) on a 30x30 packed grid on the server.

Results are printed to the replay log as lines starting with BENCH:
  BENCH round=<r> outs=<N> perf_ms=<wall ms> cpu_before=<us> cpu_after=<us> cpu_ms=<conv cpu ms>
If a line for some size never appears, that turn hit the 10 ms limit.
"""
import random
import sys
import time

from econ import Player as EconPlayer
from fcode import EntityType

try:
    import swar
except Exception as e:  # never break the bot
    swar = None
    print("BENCH import failed", repr(e), file=sys.stderr)

CIN = 26


def log(msg):
    print(msg)                    # goes into the replay
    print(msg, file=sys.stderr)   # shows in the console locally
SIZES = (2, 4, 8, 13, 20, 26)  # output channels timed per turn, cycled
START = 10


class Player:
    def __init__(self):
        self.p = EconPlayer()
        self.ready = False
        self.k = 0

    def run(self, ct):
        self.p.run(ct)
        try:
            if swar is not None and ct.get_entity_type() == EntityType.CORE:
                self.bench(ct)
        except Exception as e:
            log("BENCH error " + repr(e))

    def bench(self, ct):
        r = ct.get_current_round()
        if r < START:
            return
        if not self.ready:
            if r == START:
                random.seed(1)
                self.g = swar.G(30, 30, 16)
            elif r == START + 1:
                self.xs = [self.g.pack([random.randint(0, 15) for _ in range(900)]) for _ in range(CIN)]
            elif r == START + 2:
                self.Wp, self.Wn = [], []
                for co in range(26):
                    p, n = [], []
                    for ci in range(CIN):
                        wp, wn = [0] * 9, [0] * 9
                        for k in range(9):
                            w = random.randint(-7, 7)
                            if w > 0:
                                wp[k] = w
                            else:
                                wn[k] = -w
                        p.append(wp)
                        n.append(wn)
                    self.Wp.append(p)
                    self.Wn.append(n)
                self.ready = True
                log(f"BENCH setup done cpu_us={ct.get_cpu_time_elapsed()}")
            return
        if r > 400:
            return
        outs = SIZES[self.k % len(SIZES)]
        self.k += 1
        c0 = ct.get_cpu_time_elapsed()
        t0 = time.perf_counter()
        swar.conv(self.g, self.xs, self.Wp[:outs], self.Wn[:outs], 5)
        t1 = time.perf_counter()
        c1 = ct.get_cpu_time_elapsed()
        log(
            f"BENCH round={r} outs={outs} perf_ms={(t1 - t0) * 1000:.2f} "
            f"cpu_before={c0} cpu_after={c1} cpu_ms={(c1 - c0) / 1000:.2f}"
        )
