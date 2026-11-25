"""Frozen, resumable FrontierScience evaluation. Only talks to existing HTTP services."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

from .hashing import sha256_file

REVISION = '25ed67db7da8f4591484e764008ff585544f5a30'
COUNTS = {'olympiad': 100, 'research': 60}
DATA_HASHES = {'olympiad': '450e7fceb53c8140283a752f0d62a3abc69a85e252d4dbbd5d9dfbfb762db4e4',
               'research': '96c0434abfcbadd6ef6f59a03cc374be4caf9c1f2d5e62d8fe921e768f66aa46'}
from .official_prompts import PROMPTS as JUDGE_PROMPTS, SOURCE as PROMPT_SOURCE


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    # Atomic replace: a killed request cannot leave a partially written resume record.
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def code_hashes():
    return {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def config(path):
    c = read(path)
    if set(c) - {'base_url', 'model', 'model_artifact', 'api_key_env', 'concurrency', 'timeout', 'request', 'scoring_policy'}:
        raise ValueError('unknown config keys')
    if c.get('scoring_policy', 'strict_final') not in ('strict_final', 'mmlu_extract_first'):
        raise ValueError('unknown scoring_policy')
    u = urllib.parse.urlsplit(c['base_url'])
    if u.scheme not in ('http', 'https') or not u.hostname or u.username or u.password or u.query or u.fragment:
        raise ValueError('base_url must be HTTP(S), without embedded credentials/query/fragment')
    c['base_url'] = c['base_url'].rstrip('/')
    if c['base_url'].endswith('/v1'):
        c['base_url'] = c['base_url'][:-3]
    if not isinstance(c['model'], str) or not c['model'].strip():
        raise ValueError('model required')
    c.setdefault('concurrency', 1)
    c.setdefault('timeout', 1800)
    c.setdefault('api_key_env', None)
    if type(c['concurrency']) is not int or c['concurrency'] < 1 or c['timeout'] <= 0:
        raise ValueError('positive concurrency and timeout required')
    r = c['request']
    allowed = {'temperature', 'top_p', 'top_k', 'seed', 'max_tokens', 'stop', 'chat_template_kwargs', 'repetition_penalty', 'presence_penalty', 'frequency_penalty'}
    if set(r) - allowed or type(r.get('max_tokens')) is not int or r['max_tokens'] <= 0:
        raise ValueError('invalid request parameters; max_tokens must be positive')
    return c


def fetch(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for track in COUNTS:
        url = f'https://huggingface.co/datasets/openai/frontierscience/resolve/{REVISION}/{track}/test.jsonl'
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        p = out / track / 'test.jsonl'
        p.parent.mkdir()
        p.write_bytes(data)
        if sha256_file(p) != DATA_HASHES[track]:
            raise ValueError('download differs from pinned dataset hash')
        files[track] = {'url': url, 'sha256': sha256_file(p)}
    dump(out / 'SOURCE.json', {'revision': REVISION, 'files': files})
    return {'output': str(out), 'revision': REVISION}


def prepare(a):
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    solver, judge = config(a.solver_config), config(a.judge_config)
    tasks, sources = [], {}
    for track in COUNTS:
        p = Path(a.assets) / track / 'test.jsonl'
        raw = [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
        if len(raw) != COUNTS[track]:
            raise ValueError(f'{track}: expected {COUNTS[track]} rows; got {len(raw)}')
        if sha256_file(p) != DATA_HASHES[track]:
            raise ValueError('dataset bytes differ from pinned official snapshot')
        sources[track] = {'sha256': sha256_file(p), 'rows': len(raw)}
        for i, row in enumerate(raw):
            for field in ('problem', 'answer', 'subject', 'task_group_id'):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f'{track} row {i}: missing/invalid {field}')
            # Group IDs need not be unique; never deduplicate benchmark rows.
            tasks.append(dict(row, case_id=f'{track}-{i:04d}', track=track, row_index=i))
    out.mkdir(parents=True)
    for track in COUNTS:
        dest = out / 'assets' / track
        dest.mkdir(parents=True)
        shutil.copyfile(Path(a.assets) / track / 'test.jsonl', dest / 'test.jsonl')
    dump(out / 'tasks.json', tasks)
    contract = {'schema': 3, 'dataset': 'openai/frontierscience', 'declared_revision': REVISION,
                'revision_verification': 'source bytes match pinned hashes fetched from the revision URL',
                'sources': sources, 'tasks_sha256': sha256_file(out / 'tasks.json'),
                'solver': solver, 'judge': judge, 'judge_prompts': JUDGE_PROMPTS,
                'protocol': 'hust-qwen-judge-v3; Appendix B prompts; Qwen substitute; single trial; scoring_policy=' + judge.get('scoring_policy', 'strict_final') + '; not official replication',
                'scoring_policy': judge.get('scoring_policy', 'strict_final'),
                'scoring_policy_note': 'mmlu_extract_first: first complete explicit VERDICT, including length-ended replies; missing/invalid/failed grades remain in denominator as nonsuccess. This is local extraction, not official FrontierScience scoring.',
                'judge_prompt_source': PROMPT_SOURCE,
                'success_threshold': 7, 'client_code': code_hashes()}
    contract['runtime_contracts'] = {}
    for role in ('solver', 'judge'):
        source = getattr(a, role + '_runtime_contract', None)
        if source:
            dest = out / (role + '-runtime-contract.json')
            shutil.copyfile(source, dest)
            contract['runtime_contracts'][dest.name] = sha256_file(dest)
    (out / 'client_source').mkdir()
    for name in contract['client_code']:
        shutil.copyfile(Path(__file__).parent / name, out / 'client_source' / name)
    dump(out / 'contract.json', contract)
    return {'tasks': len(tasks), 'contract_sha256': sha256_file(out / 'contract.json')}


def load_plan(plan):
    plan = Path(plan)
    c = read(plan / 'contract.json')
    if c['tasks_sha256'] != sha256_file(plan / 'tasks.json'):
        raise ValueError('tasks changed after prepare')
    if c['client_code'] != code_hashes():
        raise ValueError('client code changed: use the original client or prepare a new plan')
    for name, sha in c.get('runtime_contracts', {}).items():
        if sha256_file(plan / name) != sha:
            raise ValueError('runtime contract changed')
    for name, sha in c['client_code'].items():
        if sha256_file(plan / 'client_source' / name) != sha:
            raise ValueError('archived client source changed')
    for track, source in c['sources'].items():
        if sha256_file(plan / 'assets' / track / 'test.jsonl') != source['sha256']:
            raise ValueError('dataset changed after prepare')
    return c, read(plan / 'tasks.json')


def request_body(c, messages):
    return dict(c['request'], model=c['model'], messages=messages, stream=False)


def call(c, body):
    started = time.time()
    record = {'request': body, 'started_unix': started, 'status': 'transport_error'}
    headers = {'Content-Type': 'application/json'}
    if c.get('api_key_env'):
        key = os.environ.get(c['api_key_env'])
        if not key:
            raise ValueError('API key environment variable is unset')
        headers['Authorization'] = 'Bearer ' + key
    req = urllib.request.Request(c['base_url'] + '/v1/chat/completions', data=json.dumps(body).encode(), headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    tic = time.monotonic()
    try:
        with opener.open(req, timeout=c['timeout']) as r:
            record['http_status'] = r.status
            record['raw_response'] = r.read().decode('utf8')
        obj = json.loads(record['raw_response'])
        record['response'] = obj
        choice = obj['choices'][0]
        record['finish_reason'] = choice.get('finish_reason')
        if record['finish_reason'] not in ('stop', 'length'):
            raise ValueError('unexpected finish_reason')
        record['content'] = choice['message'].get('content')
        if not isinstance(record['content'], str) or not record['content'].strip():
            raise ValueError('empty or non-string answer content')
        record['usage'] = obj.get('usage', {})
        record['status'] = 'ok'
    except urllib.error.HTTPError as e:
        record.update(http_status=e.code, raw_response=e.read().decode('utf8', errors='replace'), error=str(e))
    except Exception as e:
        record['error'] = str(e)
    record['elapsed_s'] = time.monotonic() - tic
    return record


def audit_reason(reason, score):
    """Conservative diagnostics only: never rewrite the official final verdict."""
    from decimal import Decimal
    flags = []
    plain = reason.replace('**', '')
    pattern = r'(?im)^\s*(?:final score|total score|total points|total points earned|total)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/\s*10|out of 10)?\s*[.!]?\s*$'
    for match in re.finditer(pattern, plain):
        if Decimal(match.group(1)) != Decimal(str(score)):
            flags.append({'kind': 'explicit_total_differs_from_verdict', 'excerpt': match.group(0).strip()})
    expression = r'(?<![\w.])([0-9]+(?:\.[0-9]+)?(?:\s*\+\s*[0-9]+(?:\.[0-9]+)?)+)\s*=\s*([0-9]+(?:\.[0-9]+)?)(?![\w.])'
    for match in re.finditer(expression, plain):
        left = sum(Decimal(x.strip()) for x in match.group(1).split('+'))
        if left != Decimal(match.group(2)):
            flags.append({'kind': 'arithmetic_in_reason_inconsistent', 'excerpt': match.group(0)})
    return flags


def parse_grade(track, text, finish_reason, policy='strict_final'):
    if policy == 'mmlu_extract_first':
        if finish_reason not in ('stop', 'length'):
            raise ValueError('unexpected judge finish_reason')
        value = r'CORRECT|INCORRECT' if track == 'olympiad' else r'[0-9]+(?:\.[0-9]+)?'
        pattern = re.compile(r'(?m)^[ \t]*VERDICT:[ \t]*(' + value + r')[ \t]*\r?$')
        matches = list(pattern.finditer(text))
        # A numeric suffix at truncated EOF may be only part of a number.
        if finish_reason == 'length' and not text.endswith('\n'):
            matches = [m for m in matches if m.end() < len(text)]
        if not matches:
            raise ValueError('no complete explicit VERDICT')
        selected = matches[0]
        token = selected.group(1)
        score = None if track == 'olympiad' else float(token)
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 10):
            raise ValueError('first verdict score outside [0,10]')
        flags = [] if score is None else audit_reason(text, score)
        values = [m.group(1) for m in matches]
        if len(set(values)) > 1:
            flags.append({'kind': 'multiple_different_verdicts', 'values': values})
        if finish_reason == 'length':
            flags.append({'kind': 'truncated_judge_verdict_recovered'})
        return {'correct': token == 'CORRECT' if score is None else score >= 7,
                'score': score, 'reason': text, 'audit_flags': flags,
                'extraction': 'first_complete_explicit_verdict', 'verdict_count': len(matches),
                'selected_verdict': selected.group(0).strip()}
    if policy != 'strict_final':
        raise ValueError('unknown scoring_policy')
    if finish_reason != 'stop':
        raise ValueError('judge must finish normally (stop), not be truncated')
    lines = text.strip().splitlines()
    if not lines:
        raise ValueError('empty judge reply')
    last = lines[-1].strip()
    reason = '\n'.join(lines[:-1])
    if track == 'olympiad':
        if last not in ('VERDICT: CORRECT', 'VERDICT: INCORRECT'):
            raise ValueError('last line must be the official Olympiad VERDICT')
        return {'correct': last == 'VERDICT: CORRECT', 'score': None, 'reason': reason, 'audit_flags': []}
    match = re.fullmatch(r'VERDICT: ([0-9]+(?:\.[0-9]+)?)', last)
    if match is None:
        raise ValueError('last line must be VERDICT: <numeric points>')
    score = float(match.group(1))
    if not math.isfinite(score) or not 0 <= score <= 10:
        raise ValueError('score must be a finite number in [0,10]')
    return {'correct': score >= 7, 'score': score, 'reason': reason, 'audit_flags': audit_reason(reason, score)}


def verify_run(directory, contract_hash, stage):
    directory = Path(directory)
    meta = read(directory / 'run.json')
    if meta['contract_sha256'] != contract_hash or meta['stage'] != stage:
        raise ValueError('run belongs to a different plan/stage')
    index = read(directory / 'index.json')
    actual = {p.name: sha256_file(p) for p in sorted((directory / 'requests').glob('*.json'))}
    if actual != index['files'] or set(meta['case_ids']) != {read(directory/'requests'/n)['case_id'] for n in actual}:
        raise ValueError('run is incomplete or request records changed')
    if index['run_sha256'] != sha256_file(directory/'run.json'):
        raise ValueError('run metadata changed')
    return meta, {read(directory/'requests'/n)['case_id']: read(directory/'requests'/n) for n in actual}


def execute(a, stage):
    c, tasks = load_plan(a.plan)
    ch = sha256_file(Path(a.plan) / 'contract.json')
    inputs = None
    upstream = None
    if stage == 'judge':
        sm, inputs = verify_run(a.responses, ch, 'solver')
        upstream = {'run_sha256': sha256_file(Path(a.responses)/'run.json'), 'index_sha256': sha256_file(Path(a.responses)/'index.json')}
        tasks = [t for t in tasks if t['case_id'] in inputs]
    if getattr(a, 'case_ids', None):
        wanted = a.case_ids
        if len(wanted) != len(set(wanted)) or set(wanted) - {t['case_id'] for t in tasks}:
            raise ValueError('duplicate or unknown case ID')
        tasks = [t for t in tasks if t['case_id'] in wanted]
    if getattr(a, 'track', None):
        tasks = [t for t in tasks if t['track'] == a.track]
    if a.limit is not None:
        if a.limit <= 0:
            raise ValueError('limit must be positive')
        tasks = tasks[:a.limit]
    if not tasks:
        raise ValueError('selection contains no tasks')
    cfg = c[stage]
    if cfg.get('api_key_env') and not os.environ.get(cfg['api_key_env']):
        raise ValueError('API key environment variable is unset')
    run = Path(a.output)
    meta = {'stage': stage, 'contract_sha256': ch, 'case_ids': [t['case_id'] for t in tasks],
            'config': cfg, 'upstream': upstream, 'client_code': code_hashes(), 'retries': 0}
    if run.exists():
        if not a.resume or read(run/'run.json') != meta:
            raise ValueError('existing output: --resume with identical configuration required')
    else:
        run.mkdir(parents=True)
        (run/'requests').mkdir()
        dump(run/'run.json', meta)
    lock = run/'.active.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError('run is locked; verify no client is running before removing stale .active.lock')
    os.close(fd)
    def one(t):
        path = run/'requests'/(t['case_id']+'.json')
        src = inputs[t['case_id']] if inputs is not None else None
        if stage == 'solver':
            messages = [{'role':'user', 'content':t['problem']}]
        else:
            prompt = c['judge_prompts'][t['track']].format_map({
                'problem': t['problem'], 'answer': src.get('content') or '',
                'reference answer': t['answer'], 'rubric': t['answer']})
            messages = [{'role': 'user', 'content': prompt}]
        body = request_body(cfg,messages)
        binding = {'case_id':t['case_id'], 'track':t['track'], 'subject':t['subject'],
                   'request_sha256':digest(body), 'contract_sha256':ch,
                   'solver_record_sha256':digest(src) if src is not None else None}
        if path.exists():
            previous = read(path)
            payload_hash = previous.pop('record_sha256')
            if digest(previous) != payload_hash or any(previous.get(k)!=v for k,v in binding.items()):
                raise ValueError('resume record altered or belongs to another request')
            return
        if src is not None and src['status'] != 'ok':
            result = {'status':'solver_failed','request':None}
        else:
            result = call(cfg,body)
            if stage == 'judge' and result['status']=='ok':
                try:
                    result['grade'] = parse_grade(t['track'],result['content'],result.get('finish_reason'),c.get('scoring_policy','strict_final'))
                except (ValueError, TypeError) as e:
                    result.update(status='invalid_judge',error=str(e))
        result.update(binding)
        result['record_sha256'] = digest(result)
        dump(path,result)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['concurrency']) as pool:
            for i, _ in enumerate(pool.map(one,tasks),1):
                print(f'{stage}: {i}/{len(tasks)}', flush=True)
        dump(run/'index.json',{'run_sha256':sha256_file(run/'run.json'),
                              'files':{p.name:sha256_file(p) for p in sorted((run/'requests').glob('*.json'))}})
    finally:
        lock.unlink()
    return {'stage':stage,'completed':len(tasks),'output':str(run)}


def summary(a):
    c, tasks = load_plan(a.plan)
    ch = sha256_file(Path(a.plan)/'contract.json')
    sm, sr = verify_run(a.responses,ch,'solver')
    jm, jr = verify_run(a.grades,ch,'judge')
    expected = {'run_sha256':sha256_file(Path(a.responses)/'run.json'), 'index_sha256':sha256_file(Path(a.responses)/'index.json')}
    if jm['upstream'] != expected or set(sr) != set(jr):
        raise ValueError('judge and solver results do not correspond exactly')
    rows=[]
    for t in tasks:
        if t['case_id'] not in sr:
            continue
        s,j=sr[t['case_id']],jr[t['case_id']]
        if j['solver_record_sha256']!=digest(s):
            raise ValueError('judge linked to different solver record')
        g=j.get('grade',{})
        rows.append({'case_id':t['case_id'],'track':t['track'],'subject':t['subject'],
                     'solver_status':s['status'],'judge_status':j['status'],
                     'solver_truncated':s.get('finish_reason')=='length',
                     'judge_truncated':j.get('finish_reason')=='length',
                     'correct':g.get('correct'),'score':g.get('score'),
                     'audit_flag_count':len(g.get('audit_flags',[]))})
    def stats(rs):
        good=[r for r in rs if r['judge_status']=='ok']
        successes=sum(r['correct'] is True for r in good)
        unresolved=sum(r['solver_status']=='ok' and r['judge_status']!='ok' for r in rs)
        points=[r['score'] for r in good if r['score'] is not None]
        return {'total':len(rs),'graded':len(good),'successes':successes,
                'solver_failures':sum(r['solver_status']!='ok' for r in rs),
                'judge_unresolved':unresolved,
                'judge_invalid':sum(r['judge_status']=='invalid_judge' for r in rs),
                'judge_transport_failures':sum(r['judge_status'] not in ('ok','invalid_judge','solver_failed') for r in rs),
                'judge_truncated_with_grade':sum(r['judge_truncated'] and r['judge_status']=='ok' for r in rs),
                'judge_audit_flagged':sum(r['audit_flag_count']>0 for r in rs),'solver_truncated':sum(r['solver_truncated'] for r in rs),
                'judge_truncated':sum(r['judge_truncated'] for r in rs),
                'success_rate':successes/len(rs) if rs and (not unresolved or c.get('scoring_policy')=='mmlu_extract_first') else None,
                'observed_success_lower_bound':successes/len(rs) if rs else None,
                'mean_research_score_valid_only':sum(points)/len(points) if points else None}
    report={'protocol':c['protocol'],'scoring_policy':c.get('scoring_policy','strict_final'),
            'denominator_policy':'all_cases_invalid_as_nonsuccess' if c.get('scoring_policy')=='mmlu_extract_first' else 'unresolved_nulls_rate','judge_model':c['judge']['model'],'solver_model':c['solver']['model'],
            'same_model_name':c['judge']['model']==c['solver']['model'],
            'scope':'full_gold' if len(rows)==160 else 'subset_smoke',
            'contract_sha256':ch, 'solver_index_sha256':expected['index_sha256'],
            'judge_index_sha256':sha256_file(Path(a.grades)/'index.json'),
            'tracks':{tr:stats([r for r in rows if r['track']==tr]) for tr in COUNTS},
            'subjects':{f'{tr}/{sub}':stats([r for r in rows if r['track']==tr and r['subject']==sub])
                        for tr,sub in sorted({(r['track'],r['subject']) for r in rows})}}
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'summary.json',report)
    with (out/'per_case.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return report


def materialize(a):
    c,tasks=load_plan(a.plan)
    if a.sample is not None:
        if not 0 < a.sample <= len(tasks):
            raise ValueError('sample must be 1..160')
        tasks=[tasks[i*len(tasks)//a.sample] for i in range(a.sample)]
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as f:
        for t in tasks:
            f.write(json.dumps({'prompt':t['problem']},ensure_ascii=False)+'\n')
    dump(str(out)+'.manifest.json',{'case_ids':[t['case_id'] for t in tasks], 'sha256':sha256_file(out),
                                 'contract_sha256':sha256_file(Path(a.plan)/'contract.json'),
                                 'note':'bench uses server default chat template; does not imply identical solver wire requests'})
    return {'prompts':len(tasks),'output':str(out)}


def add_commands(sub):
    def cmd(name, fn):
        p=sub.add_parser(name)
        p.set_defaults(fn=lambda a: print(json.dumps(fn(a),ensure_ascii=False,indent=2)))
        return p
    s=cmd('fetch',fetch); s.add_argument('--output',required=True)
    s=cmd('prepare',prepare)
    for name in ('assets','solver-config','judge-config','output'):
        s.add_argument('--'+name,required=True)
    s.add_argument('--solver-runtime-contract'); s.add_argument('--judge-runtime-contract')
    for stage, name in [('solver','run'),('judge','judge')]:
        s=cmd(name,lambda a,stage=stage:execute(a,stage))
        for arg in ('plan','output'):
            s.add_argument('--'+arg,required=True)
        if stage=='judge': s.add_argument('--responses',required=True)
        s.add_argument('--resume',action='store_true'); s.add_argument('--limit',type=int)
        s.add_argument('--track', choices=list(COUNTS))
        s.add_argument('--case-id', dest='case_ids', action='append', help='repeat for explicit smoke cases')
    s=cmd('summary',summary)
    for name in ('plan','responses','grades','output'): s.add_argument('--'+name,required=True)
    s=cmd('materialize',materialize)
    s.add_argument('--plan',required=True); s.add_argument('--output',required=True); s.add_argument('--sample',type=int)
    from .bench import bench
    s=cmd('bench',lambda a:bench(a.run_dir,a.name,a.dataset,a.base_url,a.served_model,a.tokenizer,a.num_prompts,a.output_len,a.vllm_bin,a.dry_run))
    for name in ('run-dir','name','dataset','base-url','served-model','tokenizer'): s.add_argument('--'+name,required=True)
    s.add_argument('--num-prompts',type=int,default=160); s.add_argument('--output-len',type=int,default=256)
    s.add_argument('--vllm-bin',default='vllm'); s.add_argument('--dry-run',action='store_true')
