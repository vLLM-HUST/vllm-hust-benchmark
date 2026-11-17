"""Frozen, resumable HLE-Verified evaluation. Only talks to existing HTTP services."""
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

REVISION = 'b705e0fb541c025a1532ce0d60d70ae2f53b00e0'
DATA = {
 'gold': {'filename':'Gold_subset.jsonl','rows':668,'bytes':97945135,'sha256':'f03c2632400970ffe5167d5ecde169ad38d34ce4dceb2eb8cfb54ece9c5fae00'},
 'revision': {'filename':'Revision_subset.jsonl','rows':1143,'bytes':84866749,'sha256':'a5a1b1d84edfb90844dfd2f506f331e69d58cb8ef9488ef5f4ebd420fb30cec6'},
 'uncertain': {'filename':'Uncertain_subset.jsonl','rows':689,'bytes':160325825,'sha256':'ef6979faa200d8106de1d31567ec7a5c1243eb6bb0b299a2367b1fb4014f1e5a'}
}
COUNTS = {k:v['rows'] for k,v in DATA.items()}
from .official_prompts import SYSTEM_PROMPT, JUDGE_PROMPT, RESPONSE_FORMAT, SOURCE as PROMPT_SOURCE


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
    if c.get('scoring_policy', 'strict_json') not in ('strict_json', 'complete_json'):
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
    sources = {}
    for track in a.subset or ['gold', 'revision']:
        info = DATA[track]
        url = f"https://media.githubusercontent.com/media/SKYLENAGE-AI/HLE-Verified/{REVISION}/data/{info['filename']}"
        p = out / track / 'test.jsonl'
        p.parent.mkdir()
        with urllib.request.urlopen(url, timeout=180) as response, p.open('wb') as f:
            shutil.copyfileobj(response, f)
        if p.stat().st_size != info['bytes'] or sha256_file(p) != info['sha256']:
            raise ValueError('download differs from pinned LFS hash: ' + track)
        sources[track] = dict(info, url=url)
    dump(out / 'SOURCE.json', {'revision': REVISION, 'files': sources})
    return inventory(argparse.Namespace(assets=str(out), subset=a.subset))


def inventory(a):
    result = {'dataset': 'SKYLENAGE-AI/HLE-Verified', 'revision': REVISION, 'subsets': {}}
    seen = set()
    for track in a.subset or ['gold', 'revision']:
        p = Path(a.assets) / track / 'test.jsonl'
        info = DATA[track]
        if p.stat().st_size != info['bytes'] or sha256_file(p) != info['sha256']:
            raise ValueError('dataset bytes differ from pinned LFS object: ' + track)
        with p.open(encoding='utf8') as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if len(rows) != info['rows']:
            raise ValueError('unexpected row count')
        for row in rows:
            if not isinstance(row.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']):
                raise ValueError('unsafe or missing case ID')
            if row['id'] in seen:
                raise ValueError('duplicate dataset ID')
            seen.add(row['id'])
            if not isinstance(row.get('question'), str) or not row['question'].strip():
                raise ValueError('missing question')
            if not isinstance(row.get('answer'), (str,int,float)) or not str(row['answer']).strip():
                raise ValueError('missing answer')
            if row.get('Verified_Classes') != track.title() + ' subset':
                raise ValueError('subset label mismatch')
            if row.get('answer_type') not in ('exactMatch', 'multipleChoice'):
                raise ValueError('unsupported answer_type')
            img = row.get('image')
            if img and (not isinstance(img, str) or not img.startswith('data:image/')):
                raise ValueError('image must be embedded, immutable data URL')
        result['subsets'][track] = {
            'total':len(rows), 'text':sum(not r.get('image') for r in rows),
            'image':sum(bool(r.get('image')) for r in rows),
            'answer_types':{k:sum(r['answer_type']==k for r in rows) for k in ('exactMatch','multipleChoice')},
            'sha256':info['sha256']}
    return result


def prepare(a):
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    subsets = a.subset or ['gold', 'revision']
    if len(set(subsets)) != len(subsets):
        raise ValueError('duplicate subset')
    inv = inventory(a)
    solver, judge = config(a.solver_config), config(a.judge_config)
    tasks, sources, excluded = [], {}, []
    for track in subsets:
        p = Path(a.assets) / track / 'test.jsonl'
        sources[track] = dict(DATA[track])
        with p.open(encoding='utf8') as f:
            raw = [json.loads(line) for line in f if line.strip()]
        for i, row in enumerate(raw):
            if a.modality == 'text-only' and row.get('image'):
                excluded.append({'case_id':track+'-'+row['id'], 'reason':'image excluded by explicit text-only selection'})
                continue
            tasks.append({'case_id':track+'-'+row['id'], 'id':row['id'], 'track':track,
                          'row_index':i, 'problem':row['question'], 'answer':str(row['answer']),
                          'image':row.get('image') or '', 'answer_type':row['answer_type'],
                          'subject':row.get('category') or row.get('raw_subject') or 'unknown'})
    out.mkdir(parents=True)
    for track in subsets:
        dest=out/'assets'/track; dest.mkdir(parents=True)
        shutil.copyfile(Path(a.assets)/track/'test.jsonl',dest/'test.jsonl')
    dump(out/'tasks.json',tasks)
    dump(out/'inventory.json',inv)
    dump(out/'excluded.json',excluded)
    contract={'schema':1,'dataset':'SKYLENAGE-AI/HLE-Verified','declared_revision':REVISION,
              'sources':sources,'tasks_sha256':sha256_file(out/'tasks.json'),
              'inventory_sha256':sha256_file(out/'inventory.json'),'excluded_sha256':sha256_file(out/'excluded.json'),
              'selection':{'subsets':subsets,'modality':a.modality,'excluded':len(excluded)},
              'solver':solver,'judge':judge,'solver_prompt':SYSTEM_PROMPT,'judge_prompt':JUDGE_PROMPT,
              'judge_response_format':RESPONSE_FORMAT,'prompt_source':PROMPT_SOURCE,
              'protocol':'HLE original evaluation prompts/schema on HLE-Verified; Qwen substitute; single trial; not official leaderboard replication',
              'scoring_policy':judge.get('scoring_policy','strict_json'),
              'denominator_policy':'selected_cases_including_failures; unresolved separately reported',
              'client_code':code_hashes(),'runtime_contracts':{}}
    for role in ('solver','judge'):
        source=getattr(a,role+'_runtime_contract',None)
        if source:
            dest=out/(role+'-runtime-contract.json');shutil.copyfile(source,dest)
            contract['runtime_contracts'][dest.name]=sha256_file(dest)
    (out/'client_source').mkdir()
    for name in contract['client_code']:
        shutil.copyfile(Path(__file__).parent/name,out/'client_source'/name)
    dump(out/'contract.json',contract)
    (out/'contract.sha256').write_text(sha256_file(out/'contract.json')+'\n')
    return {'tasks':len(tasks),'selection':contract['selection'],'inventory':inv,'contract_sha256':sha256_file(out/'contract.json')}


def load_plan(plan):
    plan = Path(plan)
    c = read(plan / 'contract.json')
    if (plan/'contract.sha256').read_text().strip() != sha256_file(plan/'contract.json'):
        raise ValueError('contract changed after prepare')
    for name in ('inventory','excluded'):
        if c[name+'_sha256'] != sha256_file(plan/(name+'.json')): raise ValueError(name+' changed')
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


def parse_grade(track, text, finish_reason, policy='strict_json'):
    if policy not in ('strict_json','complete_json'):
        raise ValueError('unknown scoring policy')
    if finish_reason not in ('stop','length') or (policy=='strict_json' and finish_reason!='stop'):
        raise ValueError('judge must finish normally under strict_json')
    def pairs(items):
        obj={}
        for key,value in items:
            if key in obj: raise ValueError('duplicate JSON field')
            obj[key]=value
        return obj
    def invalid_constant(value):
        raise ValueError('non-finite JSON value')
    obj=json.loads(text,object_pairs_hook=pairs,parse_constant=invalid_constant)
    required={'extracted_final_answer','reasoning','correct','confidence','strict'}
    if not isinstance(obj,dict) or set(obj)!=required:
        raise ValueError('judge response does not match official schema')
    if not all(isinstance(obj[k],str) for k in ('extracted_final_answer','reasoning')):
        raise ValueError('answer/reasoning must be strings')
    if obj['correct'] not in ('yes','no') or obj['strict'] is not True:
        raise ValueError('invalid correctness/strict fields')
    if type(obj['confidence']) is not int or not 0<=obj['confidence']<=100:
        raise ValueError('confidence must be integer 0..100')
    return {'correct':obj['correct']=='yes','score':None,'reason':obj['reasoning'],
            'extracted_final_answer':obj['extracted_final_answer'],'confidence':obj['confidence'],
            'audit_flags':[{'kind':'complete_json_recovered_from_length'}] if finish_reason=='length' else [],
            'extraction':'complete_structured_json'}


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
            content=[{'type':'text','text':t['problem']}]
            if t['image']: content.append({'type':'image_url','image_url':{'url':t['image']}})
            messages = [{'role':'system','content':c['solver_prompt']},{'role':'user','content':content}]
        else:
            prompt = c['judge_prompt'].format(question=t['problem'],response=src.get('content') or '',correct_answer=t['answer'])
            messages = [{'role': 'user', 'content': prompt}]
        body = request_body(cfg,messages)
        if stage=='judge': body['response_format']=c['judge_response_format']
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
                    result['grade'] = parse_grade(t['track'],result['content'],result.get('finish_reason'),c.get('scoring_policy','strict_json'))
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
    c,tasks=load_plan(a.plan);ch=sha256_file(Path(a.plan)/'contract.json')
    sm,sr=verify_run(a.responses,ch,'solver');jm,jr=verify_run(a.grades,ch,'judge')
    expected={'run_sha256':sha256_file(Path(a.responses)/'run.json'),'index_sha256':sha256_file(Path(a.responses)/'index.json')}
    if jm['upstream']!=expected or set(sr)!=set(jr):
        raise ValueError('judge and solver results do not correspond exactly')
    rows=[]
    for t in tasks:
        if t['case_id'] not in sr:continue
        s,j=sr[t['case_id']],jr[t['case_id']]
        if j['solver_record_sha256']!=digest(s):raise ValueError('judge linked to another solver record')
        g=j.get('grade',{})
        rows.append({'case_id':t['case_id'],'track':t['track'],'subject':t['subject'],
                     'modality':'image' if t['image'] else 'text','answer_type':t['answer_type'],
                     'solver_status':s['status'],'judge_status':j['status'],
                     'solver_truncated':s.get('finish_reason')=='length','judge_truncated':j.get('finish_reason')=='length',
                     'correct':g.get('correct'),'confidence':g.get('confidence'),
                     'audit_flag_count':len(g.get('audit_flags',[]))})
    def stats(rs):
        n=len(rs);good=[r for r in rs if r['judge_status']=='ok'];successes=sum(r['correct'] is True for r in good)
        acc=successes/n if n else None
        return {'total':n,'graded':len(good),'successes':successes,'accuracy':acc,
                'unresolved':n-len(good),'complete_grading':len(good)==n,
                'accuracy_note':'all selected cases in denominator; unresolved are nonsuccess, not confirmed incorrect',
                'wald_95_half_width':1.96*math.sqrt(acc*(1-acc)/n) if n else None,
                'solver_failures':sum(r['solver_status']!='ok' for r in rs),
                'judge_invalid':sum(r['judge_status']=='invalid_judge' for r in rs),
                'judge_transport_failures':sum(r['judge_status'] not in ('ok','invalid_judge','solver_failed') for r in rs),
                'solver_truncated':sum(r['solver_truncated'] for r in rs),'judge_truncated':sum(r['judge_truncated'] for r in rs)}
    report={'protocol':c['protocol'],'scoring_policy':c['scoring_policy'],'selection':c['selection'],
            'scope':'complete_selected_plan' if set(sr)=={t['case_id'] for t in tasks} else 'subset_smoke',
            'judge_model':c['judge']['model'],'solver_model':c['solver']['model'],
            'contract_sha256':ch,'solver_index_sha256':expected['index_sha256'],
            'judge_index_sha256':sha256_file(Path(a.grades)/'index.json'),
            'tracks':{tr:stats([r for r in rows if r['track']==tr]) for tr in c['selection']['subsets']},
            'groups':{tr+'/'+mode:stats([r for r in rows if r['track']==tr and r['modality']==mode])
                      for tr in c['selection']['subsets'] for mode in ('text','image')},
            'subjects':{tr+'/'+subject:stats([r for r in rows if r['track']==tr and r['subject']==subject])
                        for tr,subject in sorted({(r['track'],r['subject']) for r in rows})},
            'calibration_error':None,
            'calibration_note':'not implemented: pinned upstream binning omits final bin and crashes below 100 grades; raw confidence preserved; no silent replacement metric'}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);dump(out/'summary.json',report)
    with (out/'per_case.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(out/'per_case.json',rows)
    return report


def materialize(a):
    c,tasks=load_plan(a.plan)
    # Text-only load probe is explicitly separate from multimodal accuracy evaluation.
    eligible=[t for t in tasks if not t['image']]
    if a.sample is not None:
        if not 0<a.sample<=len(eligible):raise ValueError('sample exceeds eligible text rows')
        eligible=[eligible[i*len(eligible)//a.sample] for i in range(a.sample)]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as f:
        for t in eligible:f.write(json.dumps({'prompt':t['problem']},ensure_ascii=False)+'\n')
    dump(str(out)+'.manifest.json',{'case_ids':[t['case_id'] for t in eligible],
         'sha256':sha256_file(out),'contract_sha256':sha256_file(Path(a.plan)/'contract.json'),
         'excluded_images':sum(bool(t['image']) for t in tasks),
         'note':'TEXT-ONLY LOAD PROBE. No official solver system prompt; default server template. Not accuracy or equivalent multimodal workload.'})
    return {'prompts':len(eligible),'output':str(out)}


def table(a):
    from . import seal
    run=Path(a.run_dir).resolve()
    report=Path(a.summary).resolve()
    relative=report.relative_to(run).as_posix()
    bundle=read(run/'EVIDENCE_BUNDLE.json')
    if relative not in bundle['files'] or bundle['files'][relative]['sha256']!=sha256_file(report):
        raise ValueError('summary must be part of frozen evidence')
    if not seal.seal_check(run)['ok']:raise ValueError('evidence check failed')
    data=read(report)
    result=run/'results'/'hle-verified-results.md'
    result.parent.mkdir(exist_ok=True)
    lines=['# HLE-Verified evaluation','',
           'Protocol: '+data['protocol'],'','Scope: '+data['scope'],'',
           'Solver: '+data['solver_model']+'; Judge: '+data['judge_model'],'',
           'Scoring policy: '+data['scoring_policy'],'',
           '| Subset | Selected | Graded | Correct | Accuracy | Unresolved | Solver length | Judge length |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for track, stats in data['tracks'].items():
        rate='N/A' if stats['accuracy'] is None else f"{100*stats['accuracy']:.2f}%"
        lines.append(f"| {track} | {stats['total']} | {stats['graded']} | {stats['successes']} | {rate} | {stats['unresolved']} | {stats['solver_truncated']} | {stats['judge_truncated']} |")
    lines += ['', 'Gold and Revision are reported separately. This is not an official leaderboard replication.',
              'Unresolved cases stay in the denominator; their correctness is unknown. A smoke result is not a full score.',
              '', 'evidence_bundle_sha256: '+bundle['evidence_bundle_sha256'],
              'summary_sha256: '+sha256_file(report),'contract_sha256: '+data['contract_sha256'],
              '', 'See MANIFEST.json for the table SHA-256. Anchor the manifest hash outside this directory.','']
    with result.open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    return {'table':str(result),'table_sha256':sha256_file(result),'evidence_bundle_sha256':bundle['evidence_bundle_sha256']}


def add_commands(sub):
    def cmd(name, fn):
        p=sub.add_parser(name)
        p.set_defaults(fn=lambda a: print(json.dumps(fn(a),ensure_ascii=False,indent=2)))
        return p
    s=cmd('table',table); s.add_argument('--run-dir',required=True); s.add_argument('--summary',required=True)
    s=cmd('fetch',fetch); s.add_argument('--output',required=True); s.add_argument('--subset',action='append',choices=list(DATA))
    s=cmd('inventory',inventory); s.add_argument('--assets',required=True); s.add_argument('--subset',action='append',choices=list(DATA))
    s=cmd('prepare',prepare)
    s.add_argument('--subset',action='append',choices=list(DATA))
    s.add_argument('--modality',choices=['all','text-only'],default='all')
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
    s=cmd('bench',lambda a:bench(a.run_dir,a.name,a.dataset,a.base_url,a.served_model,a.tokenizer,a.num_prompts,a.output_len,a.vllm_bin,a.dry_run,a.modality,a.input_manifest))
    for name in ('run-dir','name','dataset','base-url','served-model','tokenizer'): s.add_argument('--'+name,required=True)
    s.add_argument('--num-prompts',type=int,default=200); s.add_argument('--output-len',type=int,default=256)
    s.add_argument('--vllm-bin',default='vllm'); s.add_argument('--dry-run',action='store_true')

    s.add_argument('--modality', choices=['text','image'], default='text')
    s.add_argument('--input-manifest')
    from .bench_inputs import materialize_sealed
    s=cmd('bench-materialize',lambda a:materialize_sealed(a.accuracy_run,a.output,a.modality,a.sample))
    s.add_argument('--accuracy-run',required=True)
    s.add_argument('--output',required=True)
    s.add_argument('--modality',choices=['text','image'],required=True)
    s.add_argument('--sample',type=int)
