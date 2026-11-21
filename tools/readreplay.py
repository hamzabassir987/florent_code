import sys
def rv(b,i):
    r=0;s=0
    while True:
        c=b[i];i+=1;r|=(c&0x7f)<<s;s+=7
        if not c&0x80: return r,i
def fields(b):
    i=0;out=[]
    while i<len(b):
        k,i=rv(b,i);f,t=k>>3,k&7
        if t==0: v,i=rv(b,i)
        elif t==2:
            l,i=rv(b,i);v=b[i:i+l];i+=l
        elif t==5: v=b[i:i+4];i+=4
        elif t==1: v=b[i:i+8];i+=8
        else: raise ValueError(t)
        out.append((f,v))
    return out
b=open(sys.argv[1],'rb').read()
top=fields(b)
print("top fields:",sorted(set(f for f,_ in top)))
for f,v in top:
    if f in (4,5,6): print("field",f,v if not isinstance(v,bytes) else v[:80])
turns=[v for f,v in top if f==3]
print("turns",len(turns))
outs=[]
for r,t in enumerate(turns):
    for f,u in fields(t):
        for k,val in fields(u):
            if k==9:
                d=dict(fields(val)); outs.append((r,d))
print("bot outputs",len(outs))
for r,d in outs[:3]: print(r,d)
import collections
mx=collections.defaultdict(int); tl=0; bench=[]
for r,d in outs:
    et=d.get(3,0); uid=d.get(1)
    if et>mx[uid]: mx[uid]=et
    if d.get(4): tl+=1
    s=d.get(2,b'')
    if b'BENCH' in s: bench.append((r,s.decode(errors='replace').strip()))
print("TLE count",tl)
print("max exec_us top:",sorted(mx.values())[-5:])
for r,s in bench[:40]: print(r,s)
print("bench lines",len(bench))
print("---- per-core exec time, rounds 10-40")
for uid in (1,2):
    print("unit",uid,[ (r,d.get(3),'TLE' if d.get(4) else '') for r,d in outs if d.get(1)==uid and 10<=r<=40])
print("TLE at", [(r,d) for r,d in outs if d.get(4)])
