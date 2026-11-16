"""Profile all LongBench v1 subsets in the pinned data.zip with the served model tokenizer + chat template (enable_thinking=false).
Read-only. Usage: python -I profile_lb.py <data-raw dir>
"""
import json
import sys
import zipfile

from transformers import AutoTokenizer

D = sys.argv[1]
tok = AutoTokenizer.from_pretrained('/root/workspace/models/Qwen3.5-35B-A3B')
P = json.load(open(f'{D}/dataset2prompt.json'))
M = json.load(open(f'{D}/dataset2maxlen.json'))
z = zipfile.ZipFile(f'{D}/longbench-data.zip')
names = sorted(n for n in z.namelist() if n.endswith('.jsonl'))
print(len(names), names)


def L(s):
    return len(tok(s, add_special_tokens=False)['input_ids'])


def chat(t):
    return L(tok.apply_chat_template([{'role': 'user', 'content': t}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False))


def q(v):
    v = sorted(v)
    return [v[int((len(v) - 1) * p / 100)] for p in (10, 50, 90, 100)] if v else None


for sub in ['qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique', 'narrativeqa', 'gov_report', 'qmsum',
            'multi_news', 'triviaqa', 'samsum', 'trec', 'lcc', 'repobench-p', 'passage_retrieval_en', 'passage_count']:
    n = f'data/{sub}.jsonl'
    if n not in names or sub not in P:
        print(sub, 'missing')
        continue
    rows = [json.loads(x) for x in z.read(n).decode().splitlines() if x.strip()]
    ins = [chat(P[sub].format(context=r['context'], input=r['input'])) for r in rows]
    refs = [L(r['answers'][0]) for r in rows]
    fit = sum(x + M[sub] <= 16384 for x in ins)
    print(f'{sub} n={len(rows)} maxlen={M[sub]} in p10/50/90/max={q(ins)} fit16384={fit} ref p10/50/90/max={q(refs)}',
          flush=True)
