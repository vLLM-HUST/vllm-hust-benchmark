"""MuxPD#36 single-pass comparison: native vs MuxPD default per plan, existing SLO (tools/aggregate.py slo()).
No significance labels: one pass per arm, differences are as measured.
Usage: perf36_single.py <native_pass_dir> <mux_pass_dir>
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from aggregate import slo  # noqa: E402
from loadgen import summarize  # noqa: E402

METRICS = ['out_tok_s', 'goodput_frac', 'ttft_p50', 'ttft_p99', 'tpot_p50', 'tpot_p99', 'itl_p50', 'itl_p99',
           'worst_stall', 'errors']
N, C = sys.argv[1], sys.argv[2]
plans = sorted(os.path.basename(p)[:-5] for p in glob.glob('/root/workspace/muxpd-eval/plans/formal/*.json'))
print(f'native={N} mux={C} (single pass each; SLO {vars(slo())})')
missing = []
for plan in plans:
    fn, fc = f'{N}/{plan}.json', f'{C}/{plan}.json'
    if not (os.path.exists(fn) and os.path.exists(fc)):
        missing.append(plan)
        continue
    n, c = summarize(json.load(open(fn)), slo()), summarize(json.load(open(fc)), slo())
    cells = []
    for m in METRICS:
        a, b = n.get(m), c.get(m)
        rel = f'({(b - a) / a * 100:+.1f}%)' if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a else ''
        cells.append(f'{m}={a:.4g}->{b:.4g}{rel}' if a is not None and b is not None else f'{m}={a}->{b}')
    print(f'{plan:14s} ' + ' '.join(cells))
print('missing plans:', missing or 'none')
