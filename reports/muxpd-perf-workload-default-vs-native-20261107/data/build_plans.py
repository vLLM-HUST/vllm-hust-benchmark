"""Build derived real-text workload plans for MuxPD default-vs-native validation (follow-up to #36).

Plans use the exact format consumed by tools/loadgen.py (token-ID prompts, ignore_eos fixed output lengths), so
the #36 load generator, SLO and metrics are reused unchanged.

Sources (pinned, SHA256 verified before use):
  long  = LongBench v1 qasper.jsonl (official LongBench prompt template, official 128-token output budget)
  short = OASST1 validation: assistant targets with full original ancestor chain (history -> next assistant reply)
Prompt text is rendered with the served model's chat template (enable_thinking=false) and tokenized offline.
Output length per request (derived, fixed for both arms): reference reply/answer token count clipped to
[16, budget]; budget = 128 (qasper official) or 2048 (OASST runner budget in the benchmark repo).
Selection: eligible pool -> random.Random(SEED).shuffle once -> consecutive disjoint slices per plan.
Arrival: Poisson (derived, same rates/long share as the matching #36 synthetic plans).
Usage: python -I build_plans.py <data-raw dir> <out dir>
"""
import hashlib
import json
import os
import random
import sys

from transformers import AutoTokenizer

D, OUT = sys.argv[1], sys.argv[2]
MODEL_DIR = '/root/workspace/models/Qwen3.5-35B-A3B'
MAX_MODEL_LEN = 16384
SEED = 37
EXPECT = {
    'qasper.jsonl': '29aa07d2a63f36f4fb8e8cd200a3428ee3126d750bde6af1c8f9bc41c2366854',
    'oasst1-validation.parquet': '24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416',
    'longbench-data.zip': 'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64',
    'dataset2prompt.json': '56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30',
    'dataset2maxlen.json': '72966b3c0933e214591637fb085798c5e687ebff4ddaab5d99bbc31120532022',
}
PLANS = [  # name, scenario, rate req/s, duration s, long share; fixed before any run
    ('wl-docchat-r1.0', 'mixed', 1.0, 120.0, 0.2),
    ('wl-docchat-r1.6', 'mixed', 1.6, 120.0, 0.2),
    ('wl-chat-r2', 'short', 2.0, 90.0, 0.0),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


for fn, want in EXPECT.items():
    got = sha(f'{D}/{fn}')
    assert got == want, (fn, got)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)


def ids(messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)['input_ids']


def ntok(s):
    return len(tok(s, add_special_tokens=False)['input_ids'])


# ---- long pool: LongBench v1 qasper, official prompt + budget
tpl = json.load(open(f'{D}/dataset2prompt.json'))['qasper']
budget_long = json.load(open(f'{D}/dataset2maxlen.json'))['qasper']
long_pool, long_excluded = [], []
for i, line in enumerate(open(f'{D}/qasper.jsonl')):
    r = json.loads(line)
    p = ids([{'role': 'user', 'content': tpl.format(context=r['context'], input=r['input'])}])
    out = min(budget_long, max(16, ntok(r['answers'][0])))
    src = {'source': 'LONGBENCH/qasper', 'row': i, '_id': r['_id']}
    if len(p) + budget_long > MAX_MODEL_LEN:
        long_excluded.append(src | {'in': len(p)})
        continue
    long_pool.append(src | {'prompt': p, 'out': out})

# ---- short pool: OASST1 validation history continuation
rows = [json.loads(l) for l in open(f'{D}/oasst1-validation.jsonl')]
by = {r['message_id']: r for r in rows}
short_pool = []
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
    if len(p) + out > MAX_MODEL_LEN:
        continue
    short_pool.append({'source': 'OASST1/validation', 'row': i, 'message_id': r['message_id'], 'prompt': p,
                       'out': out})

random.Random(SEED).shuffle(long_pool)
random.Random(SEED + 1).shuffle(short_pool)
li = si = 0
os.makedirs(OUT, exist_ok=True)
manifest = {'seed': SEED, 'max_model_len': MAX_MODEL_LEN, 'source_sha256': EXPECT,
            'long_pool': len(long_pool), 'long_excluded_over_context': long_excluded,
            'short_pool': len(short_pool), 'plans': {}}
for name, scen, rate, dur, lfrac in PLANS:
    rng = random.Random(f'{name}:{SEED}')
    reqs, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= dur:
            break
        if scen == 'mixed' and rng.random() < lfrac:
            s, kind = long_pool[li], 'long'
            li += 1
        else:
            s, kind = short_pool[si], 'short'
            si += 1
        reqs.append({'id': f'r{len(reqs)}', 't': round(t, 4), 'kind': kind, 'prompt': s['prompt'],
                     'max_tokens': s['out'], 'src': {k: v for k, v in s.items() if k not in ('prompt', 'out')}})
    plan = {'scenario': scen, 'params': {'scenario': scen, 'rate': rate, 'duration': dur, 'long_frac': lfrac,
                                         'seed': SEED, 'derived': True, 'name': name}, 'requests': reqs}
    json.dump(plan, open(f'{OUT}/{name}.json', 'w'))
    st = lambda v: {'min': min(v), 'p50': sorted(v)[len(v) // 2], 'p90': sorted(v)[int(len(v) * .9)], 'max': max(v),  # noqa: E731
                    'sum': sum(v)} if v else None
    k = {kd: [q for q in reqs if q['kind'] == kd] for kd in ('long', 'short')}
    manifest['plans'][name] = {
        'requests': len(reqs), 'rate': rate, 'duration': dur, 'long_frac': lfrac,
        'by_kind': {kd: {'n': len(v), 'in': st([len(q['prompt']) for q in v]), 'out': st([q['max_tokens'] for q in v])}
                    for kd, v in k.items() if v},
        'sources': [q['src'] for q in reqs]}
    print(name, len(reqs), {kd: manifest['plans'][name]['by_kind'][kd] for kd in manifest['plans'][name]['by_kind']})
json.dump(manifest, open(f'{OUT}/MANIFEST.json', 'w'), indent=1)
print('long pool', len(long_pool), 'excluded', [(e['row'], e['in']) for e in long_excluded], 'short pool', len(short_pool))
