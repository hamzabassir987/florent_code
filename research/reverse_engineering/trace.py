import gdb
out = open('/tmp/claude-0/trace.log', 'w')
FN = '_RNvMNtNtCsdmn19SHeTy9_16battlecode_titan4game10distributeNtB4_4Game20distribute_resources'
inf = None
def rd(addr, fmt):
    import struct
    m = gdb.selected_inferior().read_memory(addr, struct.calcsize(fmt))
    return struct.unpack(fmt, bytes(m))
def reg(r): return int(gdb.parse_and_eval('$' + r)) & 0xFFFFFFFFFFFFFFFF
class Probe(gdb.Breakpoint):
    def __init__(self, spec, fn):
        super().__init__(spec, internal=True); self.fn = fn
    def stop(self):
        self.fn(); return False
armed = [False]
def on_round():
    out.write('ROUND\n')
    if not armed[0]:
        armed[0] = True
        base = reg('pc')
        def push():
            a = reg('rax') + reg('rcx')
            p, sx, sy, dx, dy = rd(a, '<diiii'); out.write(f'PUSH {p:.6f} {sx} {sy} -> {dx} {dy}\n')
        def repush():
            a = reg('rsp') + 0x20
            p, sx, sy, dx, dy = rd(a, '<diiii'); out.write(f'REPUSH {p:.6f} {sx} {sy} -> {dx} {dy}\n')
        def pop():
            a = reg('rsp') + 0x1b0
            tag, = rd(a, '<i')
            if tag == 1:
                p, sx, sy, dx, dy = rd(a + 8, '<diiii'); out.write(f'POP {p:.6f} {sx} {sy} -> {dx} {dy}\n')
            else:
                out.write('POP none\n')
        Probe(f'*{base + 0xe07}', push)
        Probe(f'*{base + 0x1613}', repush)
        Probe(f'*{base + 0xffb}', pop)
gdb.execute('set pagination off'); gdb.execute('set breakpoint pending on'); gdb.execute('set print thread-events off')
Probe(FN, on_round)
gdb.execute('run')
out.close()
