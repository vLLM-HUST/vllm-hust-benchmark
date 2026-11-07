"""Score QUALITY-V2-SYNC-TP4.md part A (frozen): QUALITY-V2 endpoints on the TP4 synchronous protocol.

Per-case correctness comes from the unchanged frozen scorer (score_quality.score): complete natural stop,
valid in-range letter equal to gold; truncation/invalid/missing/error count wrong.
Usage: quality_v2_sync_score.py <runs_dir> <native,..> <raw,..> [<group,..>]
"""
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

Q = Path('/root/workspace/muxpd-eval/sync-v2/quality')
sys.path.insert(0, str(Q))
from score_quality import load_frozen, score  # noqa: E402

runs_dir = Path(sys.argv[1])
N, RAW = sys.argv[2].split(','), sys.argv[3].split(',')
GRP = sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] else []
plan, cases, extract = load_frozen(Q / 'frozen-v2')
WATCH = ['mmlu-pro-03766', 'mmlu-pro-10672', 'mmlu-pro-01741',
         'mmlu-pro-06074', 'mmlu-pro-06741', 'mmlu-pro-00882', 'mmlu-pro-08107']

raw_runs, S_ = {}, {}
for r in N + RAW + GRP:
    run = json.loads((runs_dir / r / 'results/run.json').read_text())
    raw_runs[r] = run
    S_[r] = score(run, plan, cases, extract)
ok = {r: {row['case_id']: row['correct'] for row in S_[r]['rows']} for r in S_}
tr = {r: {row['case_id']: row['finish_reason'] == 'length' for row in S_[r]['rows']} for r in S_}
ids = [c['case_id'] for c in cases]

for r in N + RAW + GRP:
    s = S_[r]
    print(f"{r:14s} correct={s['correct']} length={sum(tr[r].values())} invalid={s['invalid_answers']} "
          f"incomplete={len(s['incomplete'])} all_complete={s['all_complete']} errors={len(raw_runs[r].get('errors', []))} "
          f"uncovered_mixed={raw_runs[r].get('mixed_coverage', {}).get('uncovered_batches')}")


def tokens(r):
    return {o['case_id']: o['token_ids'] for o in raw_runs[r]['outputs']}


for arm in (N, RAW, GRP):
    if len(arm) > 1:
        t0 = tokens(arm[0])
        print('token-identical to', arm[0], {r: sum(tokens(r).get(c) == t0.get(c) for c in ids) for r in arm[1:]})
if N and RAW:
    print('raw vs native token-identical', sum(tokens(RAW[0]).get(c) == tokens(N[0]).get(c) for c in ids))


def evaluate(label, C):
    acc = lambda arm: 100 * sum(sum(ok[r].values()) for r in arm) / len(arm) / 200
    trc = lambda arm: 100 * sum(sum(tr[r].values()) for r in arm) / len(arm) / 200
    d_acc, d_tr = acc(C) - acc(N), trc(C) - trc(N)
    # Endpoint 3 = missing/errored/non-terminal outputs and collection errors. A length-limited output is a
    # complete response scored wrong and is covered by endpoint 2, so it is not a failure here.
    def failures(r):
        bad = [row for row in S_[r]['rows'] if not row['complete'] and row['finish_reason'] != 'length']
        return len(bad) + len(raw_runs[r].get('errors', [])) + (not raw_runs[r].get('collection_complete'))
    fails = sum(failures(r) for r in C)
    S = [c for c in ids if all(not ok[r][c] for r in C) and all(ok[r][c] for r in N)]
    res = {'1_accuracy': (round(acc(C), 2), round(acc(N), 2), round(d_acc, 2), d_acc >= -2.0),
           '2_truncation': (round(trc(C), 2), round(trc(N), 2), round(d_tr, 2), d_tr <= 2.0),
           '3_failures': (fails, fails == 0),
           '4_systematic': (S, len(S) <= 2)}
    print(f'\n== {label} ({len(C)} runs) vs native ({len(N)} runs)')
    for k, v in res.items():
        print(' ', k, v, 'PASS' if v[-1] else 'FAIL')
    rnd = random.Random(0)
    d = [sum(ok[r][c] for r in C) / len(C) - sum(ok[r][c] for r in N) / len(N) for c in ids]
    bs = sorted(sum(d[rnd.randrange(200)] for _ in range(200)) / 2 for _ in range(4000))
    print('  case bootstrap 95% CI pp:', round(bs[100], 2), round(bs[3899], 2))
    rnd = random.Random(0)
    ds = []
    for _ in range(20000):
        cr, nr, cs = [rnd.choice(C) for _ in C], [rnd.choice(N) for _ in N], [rnd.choice(ids) for _ in ids]
        ds.append(100 * (sum(ok[r][c] for r in cr for c in cs) / (len(cr) * 200)
                         - sum(ok[r][c] for r in nr for c in cs) / (len(nr) * 200)))
    ds.sort()
    print(f'  two-level bootstrap 95%: [{ds[500]:+.2f}, {ds[19500]:+.2f}] pp; share < -2 pp = {sum(x < -2 for x in ds) / 20000:.0%}')
    ca = [100 * sum(ok[r].values()) / 200 for r in C]
    na = [100 * sum(ok[r].values()) / 200 for r in N]
    if len(C) > 1 and len(N) > 1 and (st.variance(ca) + st.variance(na)) > 0:
        va, vb = st.variance(ca) / len(ca), st.variance(na) / len(na)
        se = math.sqrt(va + vb)
        print(f'  run-level Welch: delta {st.mean(ca) - st.mean(na):+.2f} pp, se {se:.2f}')
    else:
        print('  run-level Welch: undefined (zero run variance in both arms)')
    maj = lambda arm, c: sum(ok[r][c] for r in arm) * 2 > len(arm)
    print('  majority right->wrong', [c for c in ids if maj(N, c) and not maj(C, c)])
    print('  majority wrong->right', [c for c in ids if not maj(N, c) and maj(C, c)])
    both = [c for c in ids if all(ok[r][c] for r in N)]
    print(f'  info only, strict v1 veto: new wrong on all-native-correct ({len(both)}):',
          [c for c in both if any(not ok[r][c] for r in C)])
    for c in WATCH:
        print('  ', c, 'C', ''.join('1' if ok[r][c] else '0' for r in C), 'N', ''.join('1' if ok[r][c] else '0' for r in N))
    verdict = all(v[-1] for v in res.values())
    print(f'  {label} QUALITY_V2:', 'PASS' if verdict else 'FAIL')
    return verdict


raw_pass = evaluate('raw (gating)', RAW)
if GRP:
    evaluate('group (opt-in, reported)', GRP)
print('\nPART_A_RAW:', 'PASS' if raw_pass else 'FAIL')
