"""Local testing only (not in the submission): the float teacher through a C library."""


class TeacherPolicy:
    """Local testing only: the float teacher through a ctypes library (same logits() as StudentPolicy)."""
    def __init__(self, lib_path, model_path):
        import ctypes
        self.ct = ctypes
        self.L = ctypes.CDLL(lib_path)
        self.L.teacher_logits.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        assert self.L.teacher_load(model_path.encode()) == 0

    def logits(self, policy, raw):
        c = self.ct
        x = (c.c_float * len(raw))(*raw)
        out = (c.c_float * 2048)()
        n = self.L.teacher_logits(policy, x, out)
        return list(out[:n])   # int8 student forward: ~3.5 ms median, p95 ~5.1 ms
