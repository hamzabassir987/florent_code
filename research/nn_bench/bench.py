import time,random
def conv(inp,W,H,cin,cout,w):
    out=[[0.0]*(W*H) for _ in range(cout)]
    for co in range(cout):
        o=out[co]
        for ci in range(cin):
            x=inp[ci]
            for k in range(9):
                dy,dx=k//3-1,k%3-1; wt=w[co][ci][k]
                for y in range(max(0,-dy),H-max(0,dy)):
                    b=y*W; s=(y+dy)*W+dx
                    for xx in range(max(0,-dx),W-max(0,dx)):
                        o[b+xx]+=wt*x[s+xx]
    return out
for W,c in ((30,4),(30,8),(15,8),(15,16)):
    inp=[[random.random() for _ in range(W*W)] for _ in range(c)]
    w=[[[random.random() for _ in range(9)] for _ in range(c)] for _ in range(c)]
    t=time.perf_counter(); conv(inp,W,W,c,c,w); dt=time.perf_counter()-t
    macs=W*W*9*c*c; print(f"{W}x{W} {c}->{c}: {dt*1000:.1f} ms, {macs/dt/1e6:.1f} M MAC/s")
