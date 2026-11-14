"""wl38 (+ wl37) native vs MuxPD default; #36 SLO; single pass. Applies the pre-frozen classification rule from wl38/PLAN.md.
Usage: python -I compare.py <out.json>
"""
import json
import os
import sys

sys.path.insert(0, '/root/workspace/muxpd-eval/tools')
from aggregate import slo  # noqa: E402
from loadgen import summarize  # noqa: E402

R = '/root/workspace/muxpd-eval/results'
SETS = [('wl37', ['wl-docchat-r1.0', 'wl-docchat-r1.6', 'wl-chat-r2']),
        ('wl38', ['wl-docchat-r1.3', 'wl-2wiki-r1.6', 'wl-hotpot-r1.0', 'wl-govsum-r1.0', 'wl-code-r1.0', 'wl-chat-r1.5'])]
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


def classify(e):
    dg = {tp: (e[tp]['mux']['goodput_frac'] - e[tp]['native']['goodput_frac']) * 100 for tp in e}
    dt = {tp: (e[tp]['mux']['out_tok_s'] / e[tp]['native']['out_tok_s'] - 1) * 100 for tp in e}
    err = sum(e[tp][a]['errors'] for tp in e for a in ('mux', 'native'))
    if len(e) < 2:
        return 'incomplete', dg, dt
    if any(v <= -3 for v in dg.values()) or any(v <= -3 for v in dt.values()):
        c = 'cost'
    elif all(v >= 15 for v in dg.values()) and all(v >= -2 for v in dt.values()) and err == 0:
        c = 'very good'
    elif all(v >= 5 for v in dg.values()) or all(v >= 5 for v in dt.values()):
        c = 'gain'
    else:
        c = 'flat'
    return c, dg, dt


for tag, plans in SETS:
    for p in plans:
        e = {}
        for tp in (2, 4):
            fn, fm = f'{R}/{tag}-tp{tp}/native/pass1/{p}.json', f'{R}/{tag}-tp{tp}/mux43/pass1/{p}.json'
            if not (os.path.exists(fn) and os.path.exists(fm)):
                print(f'MISSING {tag} tp{tp} {p} native={os.path.exists(fn)} mux={os.path.exists(fm)}')
                continue
            rn, rm = json.load(open(fn)), json.load(open(fm))
            e[f'tp{tp}'] = {'native': summarize(rn, slo()), 'mux': summarize(rm, slo()), 'vn': viol(rn), 'vm': viol(rm)}
        if not e:
            continue
        c, dg, dt = classify(e)
        out[f'{tag}/{p}'] = {'class': c, 'd_goodput_pp': dg, 'd_tok_pct': dt, 'arms': e}
        print(f'\n== {tag}/{p}: {c}')
        for tp, x in e.items():
            n, m = x['native'], x['mux']
            f = lambda k, s=1: f'{n[k] * s:.3g}->{m[k] * s:.3g}'  # noqa: E731
            print(f"  {tp} req={n['requests']} tok/s {n['out_tok_s']:.1f}->{m['out_tok_s']:.1f} ({dt[tp]:+.1f}%) "
                  f"good {n['goodput_frac'] * 100:.1f}->{m['goodput_frac'] * 100:.1f}% ({dg[tp]:+.1f}pp) "
                  f"ttft50 {f('ttft_p50')} ttft99 {f('ttft_p99')} tpot50ms {f('tpot_p50', 1000)} tpot99ms {f('tpot_p99', 1000)} "
                  f"itl50ms {f('itl_p50', 1000)} itl99ms {f('itl_p99', 1000)} stall {f('worst_stall')} err {n['errors']}/{m['errors']}")
            for k in sorted(n['by_kind']):
                a, b = n['by_kind'][k], m['by_kind'][k]
                print(f"    {k}: N {a['good']}/{a['n']} ttft50 {a['ttft_p50']} viol {x['vn'][k]} | M {b['good']}/{b['n']} ttft50 {b['ttft_p50']} viol {x['vm'][k]}")
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print('\nSUMMARY')
for k, v in out.items():
    print(k, v['class'], {t: round(x, 1) for t, x in v['d_goodput_pp'].items()}, {t: round(x, 1) for t, x in v['d_tok_pct'].items()})
