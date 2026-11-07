"""QUALITY-ROW-SOLO part A gate: complete token_ids equality of every ROW x native-solo pair (200/200).
Usage (from sync-v2/): row_solo_compare.py <row,row> <native,native>
"""
import json
import sys

rows, nats = sys.argv[1].split(','), sys.argv[2].split(',')


def load(label):
    r = json.load(open(f'runs/{label}/results/run.json'))
    c = json.load(open(f'runs/{label}/controller.json'))
    return r, c, {o['case_id']: o for o in r['outputs']}


runs = {l: load(l) for l in rows + nats}
for l, (r, c, o) in runs.items():
    print(f"{l}: outputs={len(o)} complete={r['collection_complete']} shutdown={r['shutdown_returned']} "
          f"errors={len(r['errors'])} controller_passed={c['passed']} exit={c.get('exit_code')} "
          f"max_in_flight={r.get('max_in_flight_observed', 'batched')} muxpd={r['sources']['vllm_muxpd']['commit'][:7]} "
          f"vllm={r['sources']['vllm']['commit'][:7]} ascend={r['sources']['vllm_ascend']['commit'][:7]} "
          f"HCCL_DET={r['runtime_environment']['environment']['HCCL_DETERMINISTIC']}")
cases = [json.loads(x)['case_id'] for x in open('quality/frozen-v2/cases.jsonl')]
ok = True


def pair(a, b):
    A, B = runs[a][2], runs[b][2]
    tok = txt = fin = 0
    diffs = []
    for cid in cases:
        x, y = A.get(cid), B.get(cid)
        if x is None or y is None:
            diffs.append((cid, 'missing'))
            continue
        tok += x['token_ids'] == y['token_ids']
        txt += x['text'] == y['text']
        fin += x['finish_reason'] == y['finish_reason']
        if x['token_ids'] != y['token_ids']:
            p = next((i for i, (u, v) in enumerate(zip(x['token_ids'], y['token_ids'])) if u != v),
                     min(len(x['token_ids']), len(y['token_ids'])))
            diffs.append((cid, f'first diff at token {p} (len {len(x["token_ids"])} vs {len(y["token_ids"])})'))
    print(f'{a} vs {b}: token {tok}/200 text {txt}/200 finish {fin}/200')
    for d in diffs[:10]:
        print('   ', *d)
    return tok == 200 and not diffs


for group in (nats, rows):
    for i in range(1, len(group)):
        pair(group[0], group[i])
for r in rows:
    for n in nats:
        ok &= pair(r, n)
for l, (r, c, o) in runs.items():
    ok &= bool(r['collection_complete'] and r['shutdown_returned'] and not r['errors'] and c['passed'])
print('PART_A_TP4:', 'PASS' if ok else 'FAIL')
