"""Post-hoc uncertainty for QUALITY-V2 accuracy deltas (reported, never gating; added after the 2026-11-06
independent statistics review). Two-level bootstrap (runs within arm, then cases), run-level Welch t, and the
share of two-level bootstrap resamples with delta below -2 pp (a resampling proportion,
not a probability of the true loss).
Usage (from quality/): quality_v2_uncertainty.py <label> <c,c,...> <n,n,...>
"""
import json
import math
import random
import statistics as st
import sys

label, C, N = sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(',')
ok = lambda o: bool(o['correct'] and o['valid'] and o['transport_success'] and not o['length_limited'])
R = {r: {o['case_id']: ok(o) for o in json.load(open(f'{r}/score.json'))['outcomes']} for r in C + N}
cases = sorted(R[C[0]])
rnd = random.Random(0)
B = 20000
ds = []
for _ in range(B):
    cr = [rnd.choice(C) for _ in C]
    nr = [rnd.choice(N) for _ in N]
    cs = [rnd.choice(cases) for _ in cases]
    ds.append(100 * (sum(R[r][c] for r in cr for c in cs) / (len(cr) * len(cs))
                     - sum(R[r][c] for r in nr for c in cs) / (len(nr) * len(cs))))
ds.sort()
ca = [100 * sum(R[r].values()) / 200 for r in C]
na = [100 * sum(R[r].values()) / 200 for r in N]
d = st.mean(ca) - st.mean(na)
va, vb = st.variance(ca) / len(ca), st.variance(na) / len(na)
se = math.sqrt(va + vb)
df = (va + vb) ** 2 / (va ** 2 / (len(ca) - 1) + vb ** 2 / (len(na) - 1))
from statistics import NormalDist
tq = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}.get(round(df), 2.0)
print(f'{label}: delta {d:+.2f} pp | run SD C {st.stdev(ca):.2f} N {st.stdev(na):.2f} pp')
print(f'  two-level bootstrap 95%: [{ds[int(.025*B)]:+.2f}, {ds[int(.975*B)]:+.2f}] pp; share of resamples with delta < -2 pp = {sum(x < -2 for x in ds)/B:.0%}')
print(f'  run-level Welch t 95% (df {df:.1f}): [{d - tq*se:+.2f}, {d + tq*se:+.2f}] pp')
