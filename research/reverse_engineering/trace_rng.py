import gdb, struct
out = open('/tmp/claude-0/rng.log', 'w')
def reg(r): return int(gdb.parse_and_eval('$' + r)) & 0xFFFFFFFFFFFFFFFF
def mem(a, n): return bytes(gdb.selected_inferior().read_memory(a, n))
n = [0]
class B(gdb.Breakpoint):
    def __init__(s, spec, fn): super().__init__(spec, internal=True); s.fn = fn
    def stop(s): s.fn(); return False
def seed():
    out.write(f"seed_from_u64 rdi={reg('rdi'):#x} rsi={reg('rsi')}\n")
def init():
    out.write(f"init_chacha rdi={reg('rdi'):#x} rsi={reg('rsi'):#x} key={mem(reg('rsi'),32).hex()} nonce_ptr={reg('rdx'):#x}\n")
class Fin(gdb.FinishBreakpoint):
    def __init__(s, frame, args): super().__init__(frame, internal=True); s.args = args
    def stop(s):
        out.write(f"gen_range{s.args} -> {int(s.return_value) if s.return_value is not None else reg('rax') & 0xffffffff}\n"); return False
def gr():
    if n[0] < 12:
        n[0] += 1
        args = (reg('rsi') & 0xffffffff, reg('rdx') & 0xffffffff)
        try: Fin(gdb.newest_frame(), args)
        except Exception as e: out.write(f"fin err {e}\n")
gdb.execute('set pagination off'); gdb.execute('set breakpoint pending on')
B('_RNvYNtNtNtCslInYw19IZ90_4rand4rngs3std6StdRngNtCs2sT3bNnfrkI_9rand_core11SeedableRng13seed_from_u64Csdmn19SHeTy9_16battlecode_titan', seed)
B('_RNvNtCsacF5FZDheys_11rand_chacha4guts11init_chacha', init)
B('_RINvYNtNtNtCslInYw19IZ90_4rand4rngs3std6StdRngNtNtB9_3rng3Rng9gen_rangelINtNtNtCs8uuN1nrGffB_4core3ops5range5RangelEECsdmn19SHeTy9_16battlecode_titan', gr)
gdb.execute('run')
out.close()
