#!/usr/bin/env python3
"""Offline-batch or explicitly declared historical closed-loop window diagnostic using the canonical trajectory replayer.

Only completed requests within the window count toward client throughput. Engine
counter rates use their own recorded sample interval. No finite makespan claim.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from run_short_throttle_trial import http
from engine_metrics import read_once,metrics_url


def rows(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass  # A live append may be in progress.
    return result


def validate_arrival_contract(plan, inputs):
    arrival_mode=plan.get('arrival_mode','offline-burst')
    if arrival_mode=='offline-burst':
        if plan.get('concurrency')!=0 or plan.get('client_concurrency_limit') is not None:
            raise ValueError('Offline batch requires no client concurrency cap; use concurrency=0')
        if not inputs or any(float(r['arrival_time'])!=0 for r in inputs):
            raise ValueError('Offline batch requires all recorded arrivals at time0')
    elif arrival_mode=='closed-loop':
        if plan.get('concurrency',0)<=0:
            raise ValueError('Explicit historical closed-loop mode requires a positive client cap')
    else:
        raise ValueError('Unknown arrival contract')
    return arrival_mode


def run(plan):
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    workload=Path(plan['workload']);data=(workload/'sessions.csv').read_bytes()
    assert hashlib.sha256(data).hexdigest()==plan['expected_sessions_sha256']
    with (workload/'sessions.csv').open() as stream:
        inputs=list(csv.DictReader(stream))
    owned={r['session_id'] for r in inputs}
    arrival_mode=validate_arrival_contract(plan,inputs)
    report=dict(status='RUNNING',plan=plan,points=[],evidence_type='fixed-window-replay',finite_makespan_claim=False)
    def save():
        tmp=output/'window-report.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(output/'window-report.json')
    save()
    for point in plan['points']:
        endpoint=point['endpoint'];label=point['label'];directory=output/label
        assert http(endpoint,'/programs')=={}
        identity=http(endpoint,'/migration-identity')
        assert identity['scheduling_enabled'] is point['scheduling_enabled']
        assert identity['capacity_evidence']['tokens']==49152 and identity['capacity_evidence']['per_program_buffer_tokens']==14335
        assert not any(read_once(metrics_url(plan['backend'])).get(k,0) for k in ['vllm:num_requests_running','vllm:num_requests_waiting'])
        http(plan['backend'],'/reset_prefix_cache',{})
        command=[sys.executable,plan['replayer'],'--workload',str(workload),'--base-url',endpoint+'/v1',
            '--model',plan['model'],'--out',str(directory),'--api','chat','--tokenizer',plan['model_path'],
            '--concurrency',str(plan['concurrency']),'--duration-seconds',str(plan['window_seconds']),
            '--seed',str(plan['seed']),'--timeout','0','--engine-url',plan['backend']+'/v1',
            '--fixed-program-content','--chat-template-kwargs','{"enable_thinking": false}','--evidence-type','replay']
        item=dict(label=label,status='RUNNING',command=command,scheduling_enabled=point['scheduling_enabled'],concurrency=plan['concurrency'],arrival_mode=arrival_mode,client_concurrency_limit=plan.get('client_concurrency_limit',plan['concurrency']),submitted_programs=len(owned),max_waiting=0,max_paused=0)
        report['points'].append(item);save();directory.mkdir()
        log=(directory/'replayer.log').open('w');child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        started=time.monotonic();window_reached=False
        try:
            with (directory/'controller-observations.jsonl').open('w') as observation:
                while child.poll() is None:
                    metrics=rows(directory/'metrics.jsonl')
                    if metrics and metrics[-1]['t']>=plan['window_seconds']:
                        window_reached=True;break
                    if time.monotonic()-started>plan['outer_budget_seconds']:
                        raise RuntimeError('Outer budget expired before complete measurement window')
                    states=http(endpoint,'/programs');diag=http(endpoint,'/scheduler-diagnostics')
                    item['max_waiting']=max(item['max_waiting'],len(diag['waiting']))
                    item['max_paused']=max(item['max_paused'],sum(s['state']=='paused' for s in states.values()))
                    observation.write(json.dumps(dict(at=time.time(),programs=states,scheduler=diag))+'\n');observation.flush()
                    item['latest_replay_seconds']=metrics[-1]['t'] if metrics else None
                    item['completed_calls_so_far']=len(rows(directory/'rounds-live.jsonl'));save();time.sleep(.5)
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            metrics=rows(directory/'metrics.jsonl');finished=rows(directory/'rounds-live.jsonl')
            if not window_reached:raise RuntimeError('Replayer ended before full window')
            completed=[r for r in finished if r['finished_seconds']<=plan['window_seconds']]
            assert len({(r['session_id'],r['round_idx']) for r in completed})==len(completed)
            for r in completed:
                assert r['previous_history_sha256']==r['inherited_history_sha256']
                if r['status']=='ok':
                    assert r['engine_prompt_tokens']==r['expected_engine_prompt_tokens']
                    assert r['engine_completion_tokens']==r['output_len']
            expected_by_program={}
            for row in inputs:expected_by_program.setdefault(row['session_id'],set()).add(int(row['round_idx']))
            done_by_program={}
            for row in completed:
                if row['status']=='ok':done_by_program.setdefault(row['session_id'],set()).add(row['round_idx'])
            item['programs_fully_completed_within_window']=sum(done_by_program.get(pid,set())==expected for pid,expected in expected_by_program.items())
            samples=[m for m in metrics if m['t']<=plan['window_seconds']]
            assert samples and samples[0]['t']<2 and samples[-1]['t']>=plan['window_seconds']-2
            first,last=samples[0],samples[-1];elapsed=last['t']-first['t']
            item.update(status='MEASURED_WINDOW',completed_calls=len(completed),request_errors=sum(r['status']!='ok' for r in completed),
                completed_output_tokens=sum(r.get('engine_completion_tokens') or 0 for r in completed),
                kv_peak_percent=100*max(m.get('vllm:kv_cache_usage_perc',0) for m in samples),
                engine_counter_interval_seconds=elapsed,actual_input_rounds=len(inputs),finite_makespan_claim=False)
            for field,counter in [('engine_output_tokens_per_second','vllm:generation_tokens_total'),('engine_input_tokens_per_second','vllm:prompt_tokens_total')]:
                item[field]=(last[counter]-first[counter])/elapsed
            for field,counter in [('preemptions','vllm:num_preemptions_total'),('cache_hit_tokens','vllm:prefix_cache_hits_total')]:
                item[field]=last[counter]-first[counter]
            item['completed_output_tokens_per_second']=item['completed_output_tokens']/plan['window_seconds']
            (directory/'window-rounds.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in completed))
            save()
        except Exception as error:
            item.update(status='FAIL',error=str(error));report['status']='FAIL';save();raise
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            log.close()
            states=http(endpoint,'/programs');assert set(states)<=owned,'Unrelated program; refuse cleanup'
            for program in states:http(endpoint,'/programs/release',{'program_id':program})
            assert http(endpoint,'/programs')=={}
            drain_start=time.monotonic()
            while True:
                now=read_once(metrics_url(plan['backend']))
                if not any(now.get(k,0) for k in ['vllm:num_requests_running','vllm:num_requests_waiting']):break
                if time.monotonic()-drain_start>60:
                    report['status']='FAIL_DRAIN';save();raise RuntimeError('Owned in-flight requests did not drain; no next point')
                time.sleep(1)
        item['job_wall_seconds_including_initialization_and_cleanup']=time.monotonic()-started;save()
        print(json.dumps(item),flush=True)
    signatures=[]
    for point in report['points']:
        rr=rows(output/point['label']/'window-rounds.jsonl')
        signatures.append({(r['session_id'],r['round_idx']):(r['prompt_token_ids_sha256'],r['engine_prompt_tokens'],r['engine_completion_tokens']) for r in rr if r['status']=='ok'})
    common=set(signatures[0]).intersection(*[set(s) for s in signatures[1:]])
    assert common and all(all(s[k]==signatures[0][k] for k in common) for s in signatures[1:]), 'Overlapping prompt/usage mismatch'
    report.update(status='MEASURED_WINDOWS',overlapping_input_rounds=len(common),overlapping_inputs_equal=True);save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args();run(json.loads(args.plan.read_text()))
