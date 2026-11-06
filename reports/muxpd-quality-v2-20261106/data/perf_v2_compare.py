"""QUALITY-V2 performance rule: gain/regression only if every candidate pass beats/loses to every native pass.

Usage: perf_v2_compare.py <native_dir> <candidate_dir>   (dirs with pass*/<plan>.json)
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from aggregate import slo  # noqa: E402
from loadgen import summarize  # noqa: E402

HIGHER = {'out_tok_s', 'goodput_frac'}
METRICS = ['out_tok_s', 'goodput_frac', 'ttft_p50', 'ttft_p99', 'tpot_p50', 'itl_p99', 'worst_stall']


def passes(root):
    out = {}
    for f in sorted(glob.glob(f'{root}/pass*/*.json')):
        out.setdefault(os.path.basename(f)[:-5], []).append(summarize(json.load(open(f)), slo()))
    return out


N, C = passes(sys.argv[1]), passes(sys.argv[2])
gains, regs = [], []
print(f'native={sys.argv[1]} candidate={sys.argv[2]}')
for plan in sorted(N):
    if plan not in C:
        continue
    cells = []
    for m in METRICS:
        nv, cv = [p[m] for p in N[plan]], [p[m] for p in C[plan]]
        better = min(cv) > max(nv) if m in HIGHER else max(cv) < min(nv)
        worse = max(cv) < min(nv) if m in HIGHER else min(cv) > max(nv)
        tag = 'GAIN' if better else 'REG' if worse else ''
        (gains if better else regs if worse else []).append((plan, m))
        cells.append(f'{m}={sum(nv)/len(nv):.4g}->{sum(cv)/len(cv):.4g}' + (f'[{tag}]' if tag else ''))
    print(f'{plan:14s} n={len(N[plan])}/{len(C[plan])} ' + ' '.join(cells))
print('GAIN', len(gains), gains)
print('REG', len(regs), regs)
