"""Native vs MuxPD default on the derived real-text workloads; #36 SLO (tools/aggregate.py slo()) and metrics.
Single pass per arm: differences are as measured, no significance labels.
Usage: python -I compare.py <out.json>   (prints Chinese markdown tables)
"""
import json
import os
import sys

sys.path.insert(0, '/root/workspace/muxpd-eval/tools')
from aggregate import slo  # noqa: E402
from loadgen import summarize  # noqa: E402

R = '/root/workspace/muxpd-eval/results'
PLANS = ['wl-docchat-r1.0', 'wl-docchat-r1.6', 'wl-chat-r2']
REF36 = {'wl-docchat-r1.0': 'mixed-r1.0-s1..s3', 'wl-docchat-r1.6': 'mixed-r1.6-s1..s2', 'wl-chat-r2': 'short-r2-s1..s2'}
out = {}


def viol(res):
    d = {}
    for r in res['rows']:
        ev = r['events']
        k = d.setdefault(r['kind'], {'n': 0, 'err': 0, 'ttft': 0, 'tpot': 0, 'gap': 0})
        k['n'] += 1
        if r['error'] or not ev:
            k['err'] += 1
            continue
        g = [y - x for x, y in zip(ev, ev[1:])]
        n = (r['usage'] or {}).get('completion_tokens') or r['out']
        tp = (ev[-1] - ev[0]) / (n - 1) if n > 1 else 0
        k['ttft'] += (ev[0] - r['sched']) > (3.0 if r['kind'] == 'long' else 1.0)
        k['tpot'] += tp > 0.1
        k['gap'] += (max(g) if g else 0) > 0.5
    return d


def pctd(a, b):
    return f'{(b - a) / a * 100:+.1f}%' if a else 'n/a'


for tp in (2, 4):
    print(f'\n### TP{tp}\n')
    print('| workload | 输出 tok/s N -> M | Goodput N -> M | TTFT p50 s N -> M | TTFT p99 s N -> M | TPOT p50 ms N -> M | '
          'TPOT p99 ms N -> M | ITL p50 ms N -> M | ITL p99 ms N -> M | 最大停顿 s N -> M | 错误 N/M |')
    print('|' + '---|' * 11)
    for p in PLANS:
        fn, fm = f'{R}/wl37-tp{tp}/native/pass1/{p}.json', f'{R}/wl37-tp{tp}/mux43/pass1/{p}.json'
        if not (os.path.exists(fn) and os.path.exists(fm)):
            print(f'| {p} | 缺失（native={os.path.exists(fn)} mux={os.path.exists(fm)}） |')
            continue
        rn, rm = json.load(open(fn)), json.load(open(fm))
        n, m = summarize(rn, slo()), summarize(rm, slo())
        out.setdefault(f'tp{tp}', {})[p] = {'native': n, 'mux': m, 'viol_native': viol(rn), 'viol_mux': viol(rm)}
        f = lambda k, s=1, d=3: f"{n[k] * s:.{d}g} -> {m[k] * s:.{d}g}"  # noqa: E731
        print(f"| {p} | {n['out_tok_s']:.1f} -> {m['out_tok_s']:.1f} ({pctd(n['out_tok_s'], m['out_tok_s'])}) | "
              f"{n['goodput_frac'] * 100:.1f}% -> {m['goodput_frac'] * 100:.1f}% | {f('ttft_p50')} | {f('ttft_p99')} | "
              f"{f('tpot_p50', 1000)} | {f('tpot_p99', 1000)} | {f('itl_p50', 1000)} | {f('itl_p99', 1000)} | "
              f"{f('worst_stall')} | {n['errors']}/{m['errors']} |")
    print('\n按请求类型（good/n，TTFT p50 s，TPOT p50 ms；违规计数 ttft/tpot/gap）：\n')
    print('| workload | kind | native | MuxPD |')
    print('|---|---|---|---|')
    for p in PLANS:
        e = out.get(f'tp{tp}', {}).get(p)
        if not e:
            continue
        for k in sorted(e['native']['by_kind']):
            cell = []
            for arm, va in (('native', 'viol_native'), ('mux', 'viol_mux')):
                b, v = e[arm]['by_kind'][k], e[va][k]
                cell.append(f"{b['good']}/{b['n']}，TTFT {b['ttft_p50']:.3g}，TPOT {b['tpot_p50'] * 1000:.3g}；"
                            f"违规 {v['ttft']}/{v['tpot']}/{v['gap']}" + (f"，错误 {v['err']}" if v['err'] else ''))
            print(f'| {p} | {k} | {cell[0]} | {cell[1]} |')
json.dump(out, open(sys.argv[1], 'w'), indent=1)
