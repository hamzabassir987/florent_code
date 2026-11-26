"""Build the submission zip: bots/econ (main, encoder, heads, student) plus the
student's weights embedded as student_weights.py, so the bot needs no file or
environment variable on the server.

    python rl/tools/make_submission.py run/student.json submission.zip
"""
import json
import os
import sys
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
FILES = ("main.py", "encoder.py", "heads.py", "student.py")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    text = open(sys.argv[1]).read()
    d = json.loads(text)
    assert d.get("format") == "florent-student-v2", "not an int8 student.json"
    compact = json.dumps(d, separators=(",", ":"))
    with zipfile.ZipFile(sys.argv[2], "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(os.path.join(ROOT, "bots", "econ", f), f)
        meta = {k: v for k, v in d.items() if k != "policies"}
        meta["n_policies"] = len(d["policies"])
        assert meta["n_policies"] <= 8, "main.py imports at most sw0..sw7"
        files = [("sw_meta.py", "META = %r\n" % meta)]
        for k in range(8):        # main.py imports sw0..sw7 at top level: unused ones hold None
            pol = d["policies"][k] if k < len(d["policies"]) else None
            files.append(("sw%d.py" % k, "POLICY = %r\n" % (pol,)))
        parts = files
        for name, body in files:
            zi = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o100644 << 16       # writestr defaults to 0600
            z.writestr(zi, "# student weights (rl/tools/make_submission.py)\n" + body)
    print("wrote", sys.argv[2], ":", ", ".join(FILES), "+ %d weight modules" % len(parts))


if __name__ == "__main__":
    main()
