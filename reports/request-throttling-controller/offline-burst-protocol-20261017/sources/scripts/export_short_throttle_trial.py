#!/usr/bin/env python3
"""Export a terminal short-trial capsule without endpoints or token payloads."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from benchmark_artifacts import ROUND_FIELDS


def export_window(archive, destination, expected_sha256):
    def read(name):return archive.extractfile(name).read()
    def write(name,value):
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2)+'\n')
    terminal=json.loads(read('collection-terminal.json'))
    assert terminal['ray_status'] in ['SUCCEEDED','FAILED','STOPPED']
    trial=json.loads(read('window/window-report.json'));plan=trial['plan']
    assert trial['status']=='MEASURED_WINDOWS', 'Window collection is not complete'
    config={k:plan[k] for k in ['name','model','seed','programs','concurrency','window_seconds','outer_budget_seconds','expected_sessions_sha256','source_sessions_sha256','scheduler_interval_seconds']}
    config.update(arrival_mode=plan.get('arrival_mode','closed-loop'),client_concurrency_limit=plan.get('client_concurrency_limit',plan['concurrency']))
    write('config.json',config)
    data=read('workload/sessions.csv');assert hashlib.sha256(data).hexdigest()==config['expected_sessions_sha256']
    (destination/'sessions.csv').write_bytes(data)
    write('workload-manifest.json',json.loads(read('workload/manifest.json')))
    points=[]
    for point in trial['points']:
        label=point['label'];assert re.fullmatch('[a-z0-9-]+',label)
        points.append({k:v for k,v in point.items() if k not in ['command','error']})
        body=read('window/'+label+'/window-rounds.jsonl')
        rr=[json.loads(l) for l in body.splitlines() if l.strip()]
        sanitized=b''.join((json.dumps({k:r[k] for k in ROUND_FIELDS if k in r})+'\n').encode() for r in rr)
        directory=destination/label;directory.mkdir()
        (directory/'rounds.jsonl.gz').write_bytes(gzip.compress(sanitized,mtime=0))
        mm=[json.loads(l) for l in read('window/'+label+'/metrics.jsonl').splitlines() if l.strip()]
        mm=[{k:v for k,v in m.items() if k=='t' or (k.startswith('vllm:') and isinstance(v,(int,float)))} for m in mm if m['t']<=plan['window_seconds']]
        (directory/'metrics.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in mm))
    report=dict(status=trial['status'],evidence_type='fixed-window-replay',finite_makespan_claim=False,
        points=points,overlapping_input_rounds=trial['overlapping_input_rounds'],overlapping_inputs_equal=trial['overlapping_inputs_equal'],
        private_archive_sha256=expected_sha256,qualified_for_matrix=False,ray_terminal=terminal,
        limitations=['Single pair, single seed; no stable gain or best-fixed-concurrency claim.',
            'Completed-output throughput excludes unfinished requests; engine rates use their own recorded sample interval.',
            'Window-boundary cancellations do not count as completed programs or as normal request errors.',
            'Token-length replay is not task repair accuracy.',
            'Source-audited experimental hybrid reservation, APCon/512MiB, not the original matrix runtime profile.'])
    write('report.json',report)
    (destination/'verify.py').write_bytes(Path(__file__).with_name('verify_throttle_window_export.py').read_bytes())
    files=sorted(p for p in destination.rglob('*') if p.is_file())
    (destination/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(destination))+'\n' for p in files))
    return report


def export(archive_path, expected_sha256, destination):
    archive_bytes=archive_path.read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest()!=expected_sha256:
        raise ValueError('Private archive SHA256 mismatch')
    if destination.exists():
        raise ValueError('Never overwrite a published capsule')
    destination.mkdir(parents=True)
    originals={}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
        if 'window/window-report.json' in archive.getnames():
            return export_window(archive,destination,expected_sha256)
        def read(name):
            body=archive.extractfile(name).read()
            originals[name]=hashlib.sha256(body).hexdigest()
            return body
        def write(name, value):
            path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value,indent=2)+'\n')
        trial=json.loads(read('trial/trial.json'));plan=trial['plan']
        terminal=json.loads(read('collection-terminal.json'))
        if terminal['ray_status'] not in ['SUCCEEDED','FAILED','STOPPED']:
            raise ValueError('A running trial is not a terminal publication')
        sessions=read('workload/sessions.csv')
        if hashlib.sha256(sessions).hexdigest()!=plan['expected_sessions_sha256']:
            raise ValueError('Input hash mismatch')
        (destination/'sessions.csv').write_bytes(sessions)
        write('workload-manifest.json',json.loads(read('workload/manifest.json')))
        write('capacity-evidence.json',json.loads(read('capacity-evidence.json')))
        write('hybrid-audit.json',json.loads(read('hybrid-audit.json')))
        write('config.json',{k:plan[k] for k in ['name','model','programs','seed','point_budget_seconds','request_timeout_seconds','expected_sessions_sha256','source_sessions_sha256','scheduler_interval_seconds']})
        public=[];signatures=[]
        for point in trial['points']:
            label=point['label']
            if not re.fullmatch('[a-z0-9-]+',label):
                raise ValueError('Invalid point label')
            item={k:point[k] for k in ['label','status','scheduling_enabled','makespan_seconds','max_waiting','max_paused','request_errors','token_mismatches'] if k in point}
            if item['status']=='RUNNING':item['status']='INTERRUPTED'
            item['concurrency']=next(p['concurrency'] for p in plan['points'] if p['label']==label)
            if point['status']=='PASS':
                summary=json.loads(read('trial/'+label+'/summary.json'))
                write(label+'/summary.json',{k:summary[k] for k in ['started_at_wall','sessions_sha256','result']})
                body=read('trial/'+label+'/rounds.jsonl')
                rows=[json.loads(line) for line in body.splitlines() if line.strip()]
                signatures.append({(r['session_id'],r['round_idx']):(r['prompt_token_ids_sha256'],r['engine_prompt_tokens'],r['engine_completion_tokens']) for r in rows})
                sanitized=b''.join((json.dumps({k:r[k] for k in ROUND_FIELDS if k in r})+'\n').encode() for r in rows)
                (destination/label/'rounds.jsonl.gz').write_bytes(gzip.compress(sanitized,mtime=0))
                metrics=[json.loads(line) for line in read('trial/'+label+'/metrics.jsonl').splitlines() if line.strip()]
                item['kv_peak_percent']=100*max((m.get('vllm:kv_cache_usage_perc',0) for m in metrics),default=0)
            if 'after' in point:
                for field,counter in [('cache_hit_tokens','vllm:prefix_cache_hits_total'),('preemptions','vllm:num_preemptions_total')]:
                    item[field]=point['after'][counter]-point['before'][counter]
            public.append(item)
        if signatures and not all(v==signatures[0] for v in signatures):
            raise ValueError('Completed point prompt/usage signatures differ')
        report=dict(completed_points_inputs_equal=bool(signatures),status=('INTERRUPTED' if trial['status']=='RUNNING' else trial['status']),ray_terminal=terminal,evidence_type='complete-program-token-length-replay',
            points=public,paired_inputs_equal=trial.get('paired_inputs_equal'),comparisons=trial.get('comparisons'),
            seeds=[plan['seed']],qualified_for_matrix=False,
            limitations=['Single seed; no stable gain claim.','Token-length replay does not measure task repair accuracy.',
                'Explicit hybrid buffer adaptation; approximate token estimation, acting decay and inferred sharing remain boundaries.',
                'Finished-request prefill statistics do not measure all recomputation work.'],
            private_archive_sha256=expected_sha256,source_file_sha256=originals)
        write('report.json',report)
    # Independent reader has no imports from the controller or private trial.
    verifier=Path(__file__).with_name('verify_short_throttle_export.py')
    (destination/'verify.py').write_bytes(verifier.read_bytes())
    files=sorted(p for p in destination.rglob('*') if p.is_file())
    (destination/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(destination))+'\n' for p in files))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=export(args.archive,args.expected_sha256,args.output)
    print(json.dumps(result,indent=2))
