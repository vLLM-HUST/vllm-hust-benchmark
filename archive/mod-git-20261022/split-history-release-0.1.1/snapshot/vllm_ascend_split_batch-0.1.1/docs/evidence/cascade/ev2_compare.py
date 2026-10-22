"""Divergence comparator (same口径 as cascade-c3-results/probes/c3_compare.py).

Usage: ev2_compare.py A.json B.json [NAME]
Prints #diverged arms, arm indices, first-divergent step per arm, and the
fraction diverged (the C-口径 metric: diverged_arms / total_arms).
"""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
name = (
    sys.argv[3]
    if len(sys.argv) > 3
    else f"{sys.argv[1].split('/')[-1]} vs {sys.argv[2].split('/')[-1]}"
)
assert len(a) == len(b), f"arm count mismatch {len(a)} vs {len(b)}"
div = []
for i, (ta, tb) in enumerate(zip(a, b)):
    if ta != tb:
        first = next((k for k, (x, y) in enumerate(zip(ta, tb)) if x != y), -1)
        div.append((i, first, len(ta), len(tb)))
print(f"[{name}] total_arms={len(a)} diverged={len(div)} "
      f"({len(div)}/{len(a)})")
print(f"  diverged_arms={[d[0] for d in div]}")
for i, first, la, lb in div:
    print(f"  arm {i}: first_diff_step={first} lenA={la} lenB={lb}")
