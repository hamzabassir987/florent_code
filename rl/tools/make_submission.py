"""Build the submission: ONE self-contained main.py (zipped), with encoder, heads,
student and the student's weights embedded, so the bot imports nothing but the
standard library and fcode, however the server runs it.

    python rl/tools/make_submission.py run/student.json submission.zip
"""
import json
import os
import sys
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MODULES = ("encoder", "heads", "student")


def bundle(student_json):
    d = json.loads(student_json)
    assert d.get("format") == "florent-student-v2", "not an int8 student.json"
    compact = json.dumps(d, separators=(",", ":"))
    src = {m: open(os.path.join(ROOT, "bots", "econ", m + ".py")).read() for m in MODULES}
    src["student_weights"] = "STUDENT_JSON = " + repr(compact) + "\n"
    boot = ["# --- embedded modules (rl/tools/make_submission.py) ---",
            "import sys as _sys, types as _types",
            "def _embed(_name, _src):",
            "    if _name in _sys.modules:",
            "        return",
            "    _m = _types.ModuleType(_name)",
            "    _m.__file__ = '<embedded %s>' % _name",
            "    _sys.modules[_name] = _m",
            "    exec(compile(_src, _m.__file__, 'exec'), _m.__dict__)"]
    for m in ("student_weights",) + MODULES:
        boot.append("_embed(%r, %r)" % (m, src[m]))
    boot.append("# --- end embedded modules ---\n")
    return "\n".join(boot) + "\n" + open(os.path.join(ROOT, "bots", "econ", "main.py")).read()


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    text = bundle(open(sys.argv[1]).read())
    zi = zipfile.ZipInfo("main.py", date_time=(2026, 1, 1, 0, 0, 0))
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = 0o100644 << 16
    with zipfile.ZipFile(sys.argv[2], "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zi, text)
    print("wrote", sys.argv[2], ": single main.py,", len(text), "bytes")


if __name__ == "__main__":
    main()
