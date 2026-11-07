"""Profile candidate workload sources with the served model's tokenizer + chat template (enable_thinking=false).
Read-only; prints length distributions used to pick workloads. Usage: python -I profile_sources.py <data-raw dir>
"""
import csv
import json
import statistics
import sys

from transformers import AutoTokenizer

D = sys.argv[1]
tok = AutoTokenizer.from_pretrained('/root/workspace/models/Qwen3.5-35B-A3B')


def chat_len(messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return len(tok(text, add_special_tokens=False)['input_ids'])


def q(v):
    v = sorted(v)
    return {p: v[min(len(v) - 1, int(len(v) * p / 100))] for p in (0, 10, 50, 90, 99, 100)} | {'n': len(v), 'mean': round(statistics.mean(v))}


# Qasper (LongBench v1 official prompt)
tpl = json.load(open(f'{D}/dataset2prompt.json'))['qasper']
qa = [json.loads(l) for l in open(f'{D}/qasper.jsonl')]
ql = [chat_len([{'role': 'user', 'content': tpl.format(**r)}]) for r in qa]
print('qasper input', q(ql), 'fit(in+128<=16384)=', sum(x + 128 <= 16384 for x in ql))

# Dolly (publisher training prefix up to the response slot)
INTRO = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'
dl, dr, cats = [], [], {}
for l in open(f'{D}/databricks-dolly-15k.jsonl'):
    r = json.loads(l)
    if r['context']:
        p = f"{INTRO}\n\n### Instruction:\n{r['instruction']}\n\nInput:\n{r['context']}\n\n### Response:\n"
    else:
        p = f"{INTRO}\n\n### Instruction:\n{r['instruction']}\n\n### Response:\n"
    dl.append(chat_len([{'role': 'user', 'content': p}]))
    dr.append(len(tok(r['response'])['input_ids']))
    cats[r['category']] = cats.get(r['category'], 0) + 1
print('dolly input', q(dl))
print('dolly ref-response', q(dr))
print('dolly cats', cats)

# OASST1 validation: assistant targets with full alternating ancestry, not deleted, not synthetic
rows = [json.loads(l) for l in open(f'{D}/oasst1-validation.jsonl')]
by = {r['message_id']: r for r in rows}
ol, orr, nt = [], [], 0
for r in rows:
    if r['role'] != 'assistant' or r['deleted'] or r['synthetic']:
        continue
    path, cur, ok = [], by.get(r['parent_id']), True
    while cur is not None:
        if cur['deleted'] or cur['synthetic']:
            ok = False
            break
        path.append(cur)
        cur = by.get(cur['parent_id']) if cur['parent_id'] else None
        if cur is None and path[-1]['parent_id']:
            ok = False
    if not ok or not path:
        continue
    path.reverse()
    roles = [p['role'] for p in path]
    if roles[0] != 'prompter' or any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)) or roles[-1] != 'prompter':
        continue
    nt += 1
    msgs = [{'role': 'user' if p['role'] == 'prompter' else 'assistant', 'content': p['text']} for p in path]
    ol.append(chat_len(msgs))
    orr.append(len(tok(r['text'])['input_ids']))
print('oasst targets', nt, 'input', q(ol))
print('oasst ref-reply', q(orr))

# BurstGPT_3 arrivals (metadata only)
ts = []
with open('/data/shared_datasets/strict-inputs/BurstGPT_3.csv', newline='') as f:
    for i, row in enumerate(csv.DictReader(f)):
        ts.append(float(row['Timestamp']))
        if i >= 400000:
            break
ts.sort()
span = ts[-1] - ts[0]
print('burstgpt3 first 400k rows span_s', span, 'mean rate', len(ts) / span)
# per-second counts over the first hour
from collections import Counter
c = Counter(int(t - ts[0]) for t in ts if t - ts[0] < 3600)
per = [c.get(s, 0) for s in range(3600)]
print('first hour per-second', q(per), 'per-120s window counts', [sum(per[i:i + 120]) for i in range(0, 3600, 120)])
