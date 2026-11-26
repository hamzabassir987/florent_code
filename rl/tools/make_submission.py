"""Build the submission zip: bots/econ (main, encoder, heads, student) plus the
student's weights embedded as student_weights.py, so the bot needs no file or
environment variable on the server.

    python rl/tools/make_submission.py run/student.json submission.zip
"""
import base64
import json
import zlib
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
        blob = base64.b85encode(zlib.compress(compact.encode(), 9)).decode()
        parts = [blob[i:i + CHUNK] for i in range(0, len(blob), CHUNK)]
        for k, part in enumerate(parts):
            zi = zipfile.ZipInfo("sw%d.py" % k, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o100644 << 16       # writestr defaults to 0600
            head = "N = %d\n" % len(parts) if k == 0 else ""
            z.writestr(zi, "# student weights part %d (rl/tools/make_submission.py)\n%sW = %r\n" % (k, head, part))
    print("wrote", sys.argv[2], ":", ", ".join(FILES), "+ %d weight parts" % len(parts))


if __name__ == "__main__":
    main()
