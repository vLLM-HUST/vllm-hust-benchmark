#!/usr/bin/env python3
"""Bounded paired trials through the existing replayer and pinned Controller."""
import argparse
import importlib.util
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request,urlopen
from probe_prefix_cache import counters


def http(base,path,body=None,timeout=5):
    request=Request(base+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=timeout) as r:
        text=r.read().decode();return json.loads(text) if 'json' in r.headers.get('Content-Type','') else text


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text())
    expected_hash=plan.get('expected_sessions_sha256')
    if expected_hash is not None:
        actual_hash=hashlib.sha256((Path(plan['workload'])/'sessions.csv').read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError('Frozen workload hash mismatch: '+actual_hash)
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(Path(plan['canonical_root'])/'scripts'))
    from run_swe_smith_benchmark import validate_result
    report={'status':'RUNNING','plan':plan,'points':[]}
    def save():(output/'trial.json').write_text(json.dumps(report,indent=2)+'\n')
    save();signatures={}
    for point in plan['points']:
        label=point['label'];endpoint=point['endpoint'];directory=output/label
        assert http(endpoint,'/programs')=={},'Unreleased programs before point'
        identity=http(endpoint,'/migration-identity');assert identity['scheduling_enabled'] is point['scheduling_enabled']
        if plan.get('expected_capacity_evidence'):
            assert identity['capacity_evidence'] == plan['expected_capacity_evidence'], 'Unexpected capacity contract'
        if plan.get('warmup_workload'):
            warmup_output=output/('warmup-'+label)
            warmup_command=[sys.executable,plan['canonical_root']+'/experiments/workload/replay_sessions.py',
                '--workload',plan['warmup_workload'],'--base-url',endpoint+'/v1','--model',plan['model'],
                '--out',str(warmup_output),'--api','chat','--tokenizer',plan['model_path'],
                '--concurrency','1','--seed',str(plan['seed']),'--timeout',str(plan.get('request_timeout_seconds',60)),
                '--engine-url',plan['backend']+'/v1','--fixed-program-content',
                '--chat-template-kwargs','{"enable_thinking": false}','--evidence-type','replay']
            warmup_start=time.monotonic()
            with (output/('warmup-'+label+'.log')).open('w') as warmup_log:
                subprocess.run(warmup_command,stdout=warmup_log,stderr=subprocess.STDOUT,check=True,timeout=120)
            validate_result(warmup_output,1)
            assert http(endpoint,'/programs')=={},'Warmup release failure'
            report.setdefault('warmups',[]).append({'label':label,'command':warmup_command,
                'wall_seconds':time.monotonic()-warmup_start,'included_in_measurement':False})
            save()
        http(plan['backend'],'/reset_prefix_cache',{})
        before=counters(http(plan['backend'],'/metrics'));start=time.monotonic()
        command=[sys.executable,plan['canonical_root']+'/experiments/workload/replay_sessions.py',
            '--workload',plan['workload'],'--base-url',endpoint+'/v1','--model',plan['model'],
            '--out',str(directory),'--api','chat','--tokenizer',plan['model_path'],
            '--concurrency',str(point['concurrency']),'--seed',str(plan['seed']),
            '--timeout',str(plan.get('request_timeout_seconds',60)),'--engine-url',plan['backend']+'/v1','--fixed-program-content',
            '--chat-template-kwargs','{"enable_thinking": false}','--evidence-type','replay']
        record={'label':label,'status':'RUNNING','command':command,'scheduling_enabled':point['scheduling_enabled'],'max_waiting':0,'max_paused':0,'before':before};report['points'].append(record);save()
        directory.mkdir();log=(directory/'replayer.log').open('w');child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            with (directory/'controller-observations.jsonl').open('w') as stream:
                while child.poll() is None:
                    if time.monotonic()-start>plan['point_budget_seconds']:
                        record['status']='INCOMPLETE_TIMEOUT';child.terminate()
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    try:
                        states=http(endpoint,'/programs');diag=http(endpoint,'/scheduler-diagnostics')
                        record['max_waiting']=max(record['max_waiting'],len(diag['waiting']))
                        record['max_paused']=max(record['max_paused'],sum(str(v.get('state','')).lower()=='paused' for v in states.values()))
                        stream.write(json.dumps({'at':time.time(),'programs':states,'scheduler':diag})+'\n');stream.flush()
                    except Exception as error:record.setdefault('observation_errors',[]).append(str(error))
                    save();time.sleep(.5)
            record['job_wall_seconds_including_initialization']=time.monotonic()-start
            record['exit_code']=child.returncode;record['after']=counters(http(plan['backend'],'/metrics'))
            if child.returncode!=0:raise RuntimeError('Replayer failed or timed out: '+label)
            summary,rows=validate_result(directory,plan['programs']);r=summary['result']
            record.update(status='PASS',makespan_seconds=r['makespan_seconds'],rounds=r['rounds'],programs_completed=r['sessions_completed'],request_errors=r['rounds_failed'],token_mismatches=r['mismatches'])
            signatures[label]={(row['session_id'],row['round_idx']):(row['prompt_token_ids_sha256'],row['engine_prompt_tokens'],row['engine_completion_tokens']) for row in rows}
            assert http(endpoint,'/programs')=={},'Program release failure'
            save();print(json.dumps(record),flush=True)
        except Exception as error:
            if record['status']=='RUNNING':record['status']='FAIL'
            record['error']=str(error);report['status']='FAIL';save();raise
        finally:
            log.close()
            if child.poll() is None:child.terminate()
    labels=[x['label'] for x in report['points']]
    assert all(signatures[label]==signatures[labels[0]] for label in labels[1:]),'Input/usage mismatch across strategies'
    comparisons=[]
    for off_point in plan['points']:
        if off_point['scheduling_enabled']:
            continue
        for on_point in plan['points']:
            if not on_point['scheduling_enabled'] or on_point['concurrency'] != off_point['concurrency']:
                continue
            off=next(x for x in report['points'] if x['label']==off_point['label'])
            on=next(x for x in report['points'] if x['label']==on_point['label'])
            comparisons.append({'off':off['label'],'on':on['label'],
                'concurrency':off_point['concurrency'],
                'rate_ratio_on_over_off':off['makespan_seconds']/on['makespan_seconds']})
    report.update(status='PASS',paired_inputs_equal=True,comparisons=comparisons,
        throttling_observed=any(x['scheduling_enabled'] and (x['max_paused']>0 or x['max_waiting']>0) for x in report['points']))
    if len(comparisons)==1:
        report['rate_ratio_on_over_off']=comparisons[0]['rate_ratio_on_over_off']
    report['interpretation']='Exploratory paired real-model replay; no task quality or stability claim; fixed-concurrency baseline and reversed order still required.'
    save();print(json.dumps({k:v for k,v in report.items() if k not in ('plan','points')}),flush=True)

if __name__=='__main__':main()
