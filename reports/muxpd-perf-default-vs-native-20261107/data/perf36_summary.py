"""MuxPD#36: build per-plan summaries (existing SLO) for the four single-pass arms, plus aggregates and a
historical-native reference check. Output: JSON to argv[1], markdown tables to argv[2].
"""
import glob
import hashlib
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(__file__))
from aggregate import slo  # noqa: E402
from loadgen import summarize  # noqa: E402

E = '/root/workspace/muxpd-eval'
PLANS = sorted(os.path.basename(p)[:-5] for p in glob.glob(f'{E}/plans/formal/*.json'))
ARMS = {'tp2': {'native': 'results/perf36-tp2/native-s16/pass1', 'muxpd': 'results/perf36-tp2/mux43-s16/pass1'},
        'tp4': {'native': 'results/perf36-tp4/native-s16/pass1', 'muxpd': 'results/perf36-tp4/mux43-s16/pass1'}}
HIST = {'tp2': 'results/tp2/native-s16', 'tp4': 'results/tp4/native-s16'}
KEEP = ['requests', 'errors', 'span_s', 'out_tok_s', 'goodput_req', 'goodput_frac', 'goodput_req_s', 'ttft_p50',
        'ttft_p99', 'tpot_p50', 'tpot_p99', 'itl_p50', 'itl_p99', 'worst_stall', 'maxgap_p90']
HIGHER = {'out_tok_s', 'goodput_frac', 'goodput_req_s'}
out = {'slo': vars(slo()), 'single_pass': True, 'tp': {}}
md = []
for tp, arms in ARMS.items():
    rows, raw = {}, {}
    for arm, d in arms.items():
        for plan in PLANS:
            p = f'{E}/{d}/{plan}.json'
            b = open(p, 'rb').read()
            raw[f'{arm}/{plan}.json'] = hashlib.sha256(b).hexdigest()
            s = summarize(json.loads(b), slo())
            rows.setdefault(plan, {})[arm] = {k: s[k] for k in KEEP}
    agg = {}
    for arm in arms:
        req = sum(rows[p][arm]['requests'] for p in PLANS)
        good = sum(rows[p][arm]['goodput_req'] for p in PLANS)
        agg[arm] = {'requests': req, 'goodput_req': good, 'goodput_frac_pooled': round(good / req, 4),
                    'errors': sum(rows[p][arm]['errors'] for p in PLANS)}
    better = {}
    for m in ['out_tok_s', 'goodput_frac', 'ttft_p50', 'ttft_p99', 'tpot_p50', 'tpot_p99', 'itl_p50', 'itl_p99',
              'worst_stall']:
        b = w = t = 0
        for p in PLANS:
            n, c = rows[p]['native'][m], rows[p]['muxpd'][m]
            if c == n:
                t += 1
            elif (c > n) == (m in HIGHER):
                b += 1
            else:
                w += 1
        better[m] = {'muxpd_better': b, 'muxpd_worse': w, 'equal': t}
    hist = {}
    for p in PLANS:
        hv = [summarize(json.load(open(f)), slo()) for f in sorted(glob.glob(f'{E}/{HIST[tp]}/pass*/{p}.json'))]
        hist[p] = {m: {'hist_min': min(h[m] for h in hv), 'hist_max': max(h[m] for h in hv),
                       'new': rows[p]['native'][m]} for m in ['out_tok_s', 'goodput_frac', 'ttft_p50', 'tpot_p50']}
    out['tp'][tp] = {'arms': arms, 'plans': rows, 'pooled': agg, 'plan_counts': better,
                     'native_vs_historical_native_reference': hist, 'raw_result_sha256': raw}
    md.append(f'### {tp.upper()}\n')
    md.append('| Plan | Out tok/s N -> M | Goodput N -> M | TTFT p50 s N -> M | TTFT p99 s N -> M | TPOT p50 ms N -> M | '
              'TPOT p99 ms N -> M | ITL p99 ms N -> M | Worst stall s N -> M |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    for p in PLANS:
        n, c = rows[p]['native'], rows[p]['muxpd']
        def pair(m, scale=1, fmt='{:.3g}'):
            return fmt.format(n[m] * scale) + ' -> ' + fmt.format(c[m] * scale)
        md.append(f"| {p} | {pair('out_tok_s', fmt='{:.1f}')} | {pair('goodput_frac', 100, '{:.1f}%')} | "
                  f"{pair('ttft_p50')} | {pair('ttft_p99')} | {pair('tpot_p50', 1000, '{:.1f}')} | "
                  f"{pair('tpot_p99', 1000, '{:.1f}')} | {pair('itl_p99', 1000, '{:.1f}')} | {pair('worst_stall', 1, '{:.2f}')} |")
    md.append('')
    md.append(f"Pooled Goodput over all plans: native {agg['native']['goodput_req']}/{agg['native']['requests']} "
              f"({agg['native']['goodput_frac_pooled']:.1%}), MuxPD {agg['muxpd']['goodput_req']}/{agg['muxpd']['requests']} "
              f"({agg['muxpd']['goodput_frac_pooled']:.1%}); request errors native {agg['native']['errors']}, MuxPD {agg['muxpd']['errors']}.\n")
    md.append('| Metric | MuxPD better | MuxPD worse | Equal |')
    md.append('|---|---|---|---|')
    for m, v in better.items():
        md.append(f"| {m} | {v['muxpd_better']} | {v['muxpd_worse']} | {v['equal']} |")
    md.append('')
    inside = {m: sum(hist[p][m]['hist_min'] <= hist[p][m]['new'] <= hist[p][m]['hist_max'] for p in PLANS)
              for m in ['out_tok_s', 'goodput_frac', 'ttft_p50', 'tpot_p50']}
    out['tp'][tp]['native_new_inside_historical_range'] = inside
    md.append(f'New native inside the 3-pass historical native range (plans of 13): {inside}\n')
json.dump(out, open(sys.argv[1], 'w'), indent=1)
open(sys.argv[2], 'w').write('\n'.join(md) + '\n')
