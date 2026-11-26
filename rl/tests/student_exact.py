"""Exactness: the Python SWAR int8 student (bots/econ/student.py) must reproduce the
C++ torch student's integer logits bit for bit, on vectors dumped by
`distill` (VECTORS=...). Also cross-checks SWAR against a plain-integer
reference on a few vectors, and times the SWAR forward.

    python student_exact.py STUDENT.json VECTORS.jsonl
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bots", "econ"))
import student as S  # noqa: E402

model = S.StudentPolicy(sys.argv[1])
n = bad = ref_checked = ref_bad = 0
times = []
for line in open(sys.argv[2]):
    v = json.loads(line)
    q = S.quantize(v["raw"])
    t0 = time.perf_counter()
    got = model.logits_int(v["policy"], q)
    times.append((time.perf_counter() - t0) * 1000)
    n += 1
    if got != v["logits"]:
        bad += 1
        if bad <= 3:
            print("MISMATCH policy", v["policy"], [(i, a, b) for i, (a, b) in enumerate(zip(got, v["logits"])) if a != b][:5])
    if ref_checked < 3:
        ref_checked += 1
        if S.reference_logits(model, v["policy"], q) != got:
            ref_bad += 1
times.sort()
print(f"vectors {n}  mismatches {bad}  reference cross-check {ref_checked - ref_bad}/{ref_checked} ok")
print(f"SWAR forward (all logits): median {times[len(times)//2]:.2f} ms  p95 {times[int(len(times)*0.95)]:.2f} ms")
print("STUDENT EXACT" if bad == 0 and ref_bad == 0 else "STUDENT NOT EXACT")
