"""Per-kind breakdown of #36 single-pass results (existing SLO), to choose scenario classes for #37 workload validation."""
import glob
import json
import os
import sys

sys.path.insert(0, '/root/workspace/muxpd-eval/tools')
from aggregate import slo  # noqa: E402
from loadgen import summarize, pct  # noqa: E402

R = '/root/workspace/muxpd-eval/results'
for tp in (2, 4):
    print(f'===== TP{tp}')
    for f in sorted(glob.glob(f'{R}/perf36-tp{tp}/native-s16/pass1/*.json')):
        plan = os.path.basename(f)[:-5]
        out = []
        for arm in ('native-s16', 'mux43-s16'):
            res = json.load(open(f'{R}/perf36-tp{tp}/{arm}/pass1/{plan}.json'))
            s = summarize(res, slo())
            # why requests fail SLO, per kind
            fail = {}
            for r in res['rows']:
                ev = r['events']
                if r['error'] or not ev:
                    fail.setdefault(r['kind'], {}).setdefault('err', 0)
                    fail[r['kind']]['err'] += 1
                    continue
                g = [y - x for x, y in zip(ev, ev[1:])]
                n = (r['usage'] or {}).get('completion_tokens') or r['out']
                tp_ = (ev[-1] - ev[0]) / (n - 1) if n > 1 else 0
                lim = 3.0 if r['kind'] == 'long' else 1.0
                k = fail.setdefault(r['kind'], {'ttft': 0, 'tpot': 0, 'gap': 0, 'n': 0})
                k['n'] += 1
                k['ttft'] += (ev[0] - r['sched']) > lim
                k['tpot'] += tp_ > 0.1
                k['gap'] += (max(g) if g else 0) > 0.5
            bk = {kk: f"{v['good']}/{v['n']} ttft50={v['ttft_p50']} tpot50={v['tpot_p50']}" for kk, v in s['by_kind'].items()}
            out.append(f"  {arm:10s} good={s['goodput_frac']} tok/s={s['out_tok_s']} | {bk} | viol={fail}")
        print(plan)
        print('\n'.join(out))
