"""Score QUALITY-V2.md endpoints exactly as frozen. Usage: quality_v2_score.py (from quality/)."""
import json
import random

C = ['qv2-c1', 'qv2-c2', 'qv2-c3', 'qv2-c4']
N = ['qv2-n1', 'qv2-n2', 'native-q1', 'native-q2', 'native-q3']
CTX = ['native-det-q1', 'native-det-q2', 'muxopt-q1', 'muxopt-q2', 'bsh-q1', 'bsh-q2', 'row-q1', 'row-q2']
WATCH = ['mmlu-pro-04748', 'mmlu-pro-06908', 'mmlu-pro-11119', 'mmlu-pro-00168']


def load(d):
    return {o['case_id']: o for o in json.load(open(f'{d}/score.json'))['outcomes']}


def ok(o):
    return bool(o['correct'] and o['valid'] and o['transport_success'] and not o['length_limited'])


R = {r: load(r) for r in C + N + CTX}
cases = sorted(R['qv2-c1'])
assert all(len(R[r]) == 200 and set(R[r]) == set(cases) for r in R)


def per_run(r):
    v = R[r].values()
    return dict(gate=sum(map(ok, v)), scorer=sum(bool(o['correct']) for o in v),
                trunc=sum(bool(o['length_limited']) for o in v), invalid=sum(not o['valid'] for o in v),
                transport_fail=sum(not o['transport_success'] for o in v))


for r in C + N + CTX:
    print(f'{r:14s} {per_run(r)}' + ('   [context only]' if r in CTX else ''))

acc = lambda arm: sum(per_run(r)['gate'] for r in arm) / len(arm) / 2  # pp
tr = lambda arm: sum(per_run(r)['trunc'] for r in arm) / len(arm) / 2
d_acc, d_tr = acc(C) - acc(N), tr(C) - tr(N)
transport = sum(per_run(r)['transport_fail'] for r in C)
S = [c for c in cases if all(not ok(R[r][c]) for r in C) and all(ok(R[r][c]) for r in N)]
S_rev = [c for c in cases if all(ok(R[r][c]) for r in C) and all(not ok(R[r][c]) for r in N)]
random.seed(0)
d = [sum(ok(R[r][c]) for r in C) / len(C) - sum(ok(R[r][c]) for r in N) / len(N) for c in cases]
bs = sorted(sum(d[random.randrange(200)] for _ in range(200)) / 2 for _ in range(4000))
maj = lambda arm, c: sum(ok(R[r][c]) for r in arm) * 2 > len(arm)
r2w = [c for c in cases if maj(N, c) and not maj(C, c)]
w2r = [c for c in cases if not maj(N, c) and maj(C, c)]
res = {
    '1_accuracy': (round(acc(C), 2), round(acc(N), 2), round(d_acc, 2), d_acc >= -2.0),
    '2_truncation': (round(tr(C), 2), round(tr(N), 2), round(d_tr, 2), d_tr <= 2.0),
    '3_transport': (transport, transport == 0),
    '4_systematic': (S, len(S) <= 2),
}
print()
for k, v in res.items():
    print(k, v, 'PASS' if v[-1] else 'FAIL')
print('bootstrap 95% CI of acc(C)-acc(N) pp:', round(bs[100], 2), round(bs[3899], 2))
print('majority right->wrong', len(r2w), r2w)
print('majority wrong->right', len(w2r), w2r)
print('reverse systematic (all C right, all N wrong)', S_rev)
scorer_acc = lambda arm: sum(per_run(r)['scorer'] for r in arm) / len(arm) / 2
print('shared-scorer acc C %.2f N %.2f delta %.2f' % (scorer_acc(C), scorer_acc(N), scorer_acc(C) - scorer_acc(N)))
for c in WATCH:
    print(c, 'C', ''.join('1' if ok(R[r][c]) else '0' for r in C), 'N', ''.join('1' if ok(R[r][c]) else '0' for r in N))
print('QUALITY_V2:', 'PASS' if all(v[-1] for v in res.values()) else 'FAIL')
