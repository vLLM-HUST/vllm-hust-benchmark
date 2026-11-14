"""Build round-2 derived real-text workload plans (wl38) for MuxPD default vs native. Same plan format as tools/loadgen.py.

Request text: LongBench v1 subsets (official prompt as one user message, official output budget) and OASST1 validation
history continuations; rendered with the served chat template (enable_thinking=false), tokenized offline.
Output length (derived, identical for both arms): reference token count clipped to [16, budget], forced by ignore_eos.
Rows with input + budget > 16384 are excluded (recorded), never truncated.
Selection: each pool shuffled once with random.Random(f'{pool}:38'); plans take consecutive disjoint slices.
Arrival (derived): Poisson; component drawn per request by fixed probabilities.
Usage: python -I build_plans.py <wl37 data-raw dir> <out dir>
"""
import hashlib
import json
import os
import random
import sys
import zipfile

from transformers import AutoTokenizer

D, OUT = sys.argv[1], sys.argv[2]
MAXLEN = 16384
SEED = 38
EXPECT = {
    'longbench-data.zip': 'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64',
    'dataset2prompt.json': '56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30',
    'dataset2maxlen.json': '72966b3c0933e214591637fb085798c5e687ebff4ddaab5d99bbc31120532022',
    'oasst1-validation.parquet': '24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416',
    'oasst1-validation.jsonl': '3f1290a7f2b821516f7f7f66d7dcbf87816779656a35a6def084d26548fb2cec',
}
# name, rate req/s, duration s, [(pool, probability)]; fixed before any run
PLANS = [
    ('wl-docchat-r1.3', 1.3, 120.0, [('qasper', 0.2), ('oasst', 0.8)]),
    ('wl-2wiki-r1.6', 1.6, 120.0, [('2wikimqa', 0.2), ('oasst', 0.8)]),
    ('wl-hotpot-r1.0', 1.0, 120.0, [('hotpotqa', 0.2), ('oasst', 0.8)]),
    ('wl-govsum-r1.0', 1.0, 120.0, [('gov_report', 0.2), ('oasst', 0.8)]),
    ('wl-code-r1.0', 1.0, 120.0, [('lcc', 0.5), ('repobench-p', 0.5)]),
    ('wl-chat-r1.5', 1.5, 90.0, [('oasst', 1.0)]),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


for fn, want in EXPECT.items():
    assert sha(f'{D}/{fn}') == want, fn
tok = AutoTokenizer.from_pretrained('/root/workspace/models/Qwen3.5-35B-A3B')


def ids(messages):
    t = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(t, add_special_tokens=False)['input_ids']


def ntok(s):
    return len(tok(s, add_special_tokens=False)['input_ids'])


P = json.load(open(f'{D}/dataset2prompt.json'))
M = json.load(open(f'{D}/dataset2maxlen.json'))
z = zipfile.ZipFile(f'{D}/longbench-data.zip')
needed = sorted({p for _, _, _, comp in PLANS for p, _ in comp} - {'oasst'})
pools, excluded = {}, {}
for sub in needed:
    pool, exc = [], []
    for i, line in enumerate(z.read(f'data/{sub}.jsonl').decode().splitlines()):
        r = json.loads(line)
        p = ids([{'role': 'user', 'content': P[sub].format(context=r['context'], input=r['input'])}])
        src = {'source': f'LONGBENCH/{sub}', 'row': i, '_id': r['_id']}
        if len(p) + M[sub] > MAXLEN:
            exc.append(src | {'in': len(p)})
            continue
        pool.append(src | {'prompt': p, 'out': min(M[sub], max(16, ntok(r['answers'][0]))), 'kind': 'long'})
    pools[sub], excluded[sub] = pool, exc

rows = [json.loads(l) for l in open(f'{D}/oasst1-validation.jsonl')]
by = {r['message_id']: r for r in rows}
op = []
for i, r in enumerate(rows):
    if r['role'] != 'assistant' or r['deleted'] or r['synthetic']:
        continue
    path, cur, ok = [], by.get(r['parent_id']), True
    while cur is not None:
        if cur['deleted'] or cur['synthetic']:
            ok = False
            break
        path.append(cur)
        if not cur['parent_id']:
            break
        nxt = by.get(cur['parent_id'])
        if nxt is None:
            ok = False
            break
        cur = nxt
    if not ok or not path:
        continue
    path.reverse()
    roles = [x['role'] for x in path]
    if roles[0] != 'prompter' or roles[-1] != 'prompter' or any(a == b for a, b in zip(roles, roles[1:])):
        continue
    p = ids([{'role': 'user' if x['role'] == 'prompter' else 'assistant', 'content': x['text']} for x in path])
    out = min(2048, max(16, ntok(r['text'])))
    if len(p) + out > MAXLEN:
        continue
    op.append({'source': 'OASST1/validation', 'row': i, 'message_id': r['message_id'], 'prompt': p, 'out': out,
               'kind': 'short'})
pools['oasst'] = op
for k in pools:
    random.Random(f'{k}:{SEED}').shuffle(pools[k])
cursor = {k: 0 for k in pools}
os.makedirs(OUT, exist_ok=True)
man = {'seed': SEED, 'max_model_len': MAXLEN, 'source_sha256': EXPECT, 'pool_sizes': {k: len(v) for k, v in pools.items()},
       'excluded_over_context': excluded, 'plans': {}}
for name, rate, dur, comp in PLANS:
    rng = random.Random(f'{name}:{SEED}')
    reqs, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= dur:
            break
        u, acc, pick = rng.random(), 0.0, comp[-1][0]
        for pn, pr in comp:
            acc += pr
            if u < acc:
                pick = pn
                break
        assert cursor[pick] < len(pools[pick]), pick
        s = pools[pick][cursor[pick]]
        cursor[pick] += 1
        reqs.append({'id': f'r{len(reqs)}', 't': round(t, 4), 'kind': s['kind'], 'prompt': s['prompt'],
                     'max_tokens': s['out'], 'src': {k: v for k, v in s.items() if k not in ('prompt', 'out', 'kind')}})
    plan = {'scenario': 'mixed', 'params': {'name': name, 'rate': rate, 'duration': dur, 'components': comp,
                                            'seed': SEED, 'derived': True}, 'requests': reqs}
    json.dump(plan, open(f'{OUT}/{name}.json', 'w'))

    def st(v):
        v = sorted(v)
        return {'n': len(v), 'p50': v[len(v) // 2], 'p90': v[int(len(v) * .9)], 'max': v[-1], 'sum': sum(v)} if v else None
    bys = {}
    for q in reqs:
        bys.setdefault(q['src']['source'], []).append(q)
    man['plans'][name] = {'requests': len(reqs), 'rate': rate, 'duration': dur, 'components': comp,
                          'by_source': {k: {'in': st([len(q['prompt']) for q in v]), 'out': st([q['max_tokens'] for q in v])}
                                        for k, v in bys.items()},
                          'sources': [q['src'] for q in reqs]}
    print(name, len(reqs), json.dumps(man['plans'][name]['by_source']))
json.dump(man, open(f'{OUT}/MANIFEST.json', 'w'), indent=1)
print('pools', man['pool_sizes'], 'excluded', {k: len(v) for k, v in excluded.items()})
