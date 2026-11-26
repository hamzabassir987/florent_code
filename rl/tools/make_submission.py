"""Build the submission zip: bots/econ (main, encoder, heads, student) plus the
student's weights embedded as student_weights.py, so the bot needs no file or
environment variable on the server.

    python rl/tools/make_submission.py run/student.json submission.zip
"""
import base64
import json
import os
import sys
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CHUNK = 28000   # chars per weights module: the server dropped a 279 KB file
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
        def pack(x):        # int8 lists -> base85 of their bytes (pure-Python decoder in main.py)
            if isinstance(x, dict):
                return {k: pack(v) for k, v in x.items()}
            if isinstance(x, list):
                if x and all(isinstance(v, int) and -128 <= v <= 127 for v in x):
                    raw = bytes(v & 255 for v in x)
                    return {"b85": base64.b85encode(raw + bytes(-len(raw) % 4)).decode(), "n": len(x)}
                return [pack(v) for v in x]
            return x
        blob = json.dumps(pack(d), separators=(",", ":"))
        parts = [blob[i:i + CHUNK] for i in range(0, len(blob), CHUNK)]
        assert len(parts) <= 8, "main.py imports at most sw0..sw7"
        for k, part in enumerate(parts):
            zi = zipfile.ZipInfo("sw%d.py" % k, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o100644 << 16       # writestr defaults to 0600
            z.writestr(zi, "# student weights part %d (rl/tools/make_submission.py)\nW = %r\n" % (k, part))
    print("wrote", sys.argv[2], ":", ", ".join(FILES), "+ %d weight parts" % len(parts))


if __name__ == "__main__":
    main()
