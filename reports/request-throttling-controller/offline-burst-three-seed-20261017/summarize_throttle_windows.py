#!/usr/bin/env python3
"""Verify public replay capsules and summarize paired window rates by seed."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def summarize(capsules, order_repeats=()):
    records=[]
    binding=None
    seen=set()
    for repeat,paths in [(False,capsules),(True,order_repeats)]:
        for path in paths:
            config=json.loads((path/'config.json').read_text())
            report=json.loads((path/'report.json').read_text())
            expected={key:config[key] for key in ['name','model','programs','concurrency','window_seconds','expected_sessions_sha256','arrival_mode','client_concurrency_limit']}
            if expected['arrival_mode']!='offline-burst' or expected['client_concurrency_limit'] is not None or expected['concurrency']!=0:
                raise ValueError('Only uncapped offline-burst capsules belong in this summary')
            if binding is None:binding=expected
            if expected!=binding:raise ValueError('Input, model or measurement protocol differs')
            if not repeat:
                if config['seed'] in seen:raise ValueError('Duplicate primary seed; label order repeats separately')
                seen.add(config['seed'])
            if report['status']!='MEASURED_WINDOWS' or report['ray_terminal']['ray_status']!='SUCCEEDED' or not report['overlapping_inputs_equal']:
                raise ValueError('Incomplete or inconsistent window pair')
            points=report['points']
            if len(points)!=2 or {p['scheduling_enabled'] for p in points}!={False,True}:
                raise ValueError('Exactly one ON/OFF pair required')
            on=next(p for p in points if p['scheduling_enabled']);off=next(p for p in points if not p['scheduling_enabled'])
            if any(p['request_errors'] for p in points):raise ValueError('Request errors invalidate aggregation')
            if off['engine_output_tokens_per_second']<=0:raise ValueError('Non-positive OFF throughput')
            records.append(dict(run=report['ray_terminal']['job_id'].split('--')[0],seed=config['seed'],order_repeat=repeat,
                order=['ON' if p['scheduling_enabled'] else 'OFF' for p in points],
                off=off,on=on,paired_output_rate_ratio=on['engine_output_tokens_per_second']/off['engine_output_tokens_per_second'],
                source_report_sha256=hashlib.sha256((path/'report.json').read_bytes()).hexdigest(),
                source_config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest()))
    primary=[r for r in records if not r['order_repeat']]
    if not primary:raise ValueError('No primary pairs')
    return dict(schema=1,status='VERIFIED_WINDOW_PAIRS',evidence_type='fixed-window-replay',protocol=binding,
        finite_makespan_claim=False,quality=None,qualified_for_original_matrix=False,
        aggregate=dict(seeds=sorted(seen),repetitions=len(primary),
            median_paired_output_rate_ratio=statistics.median(r['paired_output_rate_ratio'] for r in primary),
            median_off_output_tokens_per_second=statistics.median(r['off']['engine_output_tokens_per_second'] for r in primary),
            median_on_output_tokens_per_second=statistics.median(r['on']['engine_output_tokens_per_second'] for r in primary)),
        records=records,
        limitations=['Fixed windows do not establish complete-batch makespan or task repair accuracy.',
            'Runtime profile must be read alongside the attached deployment evidence; original matrix qualification is false.',
            'Order repeat is separate and excluded from the primary seed median.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capsules',type=Path,nargs='+',required=True)
    p.add_argument('--order-repeat',type=Path,nargs='*',default=[])
    p.add_argument('--runtime-profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Never overwrite a frozen summary; choose a new output')
    for path in args.capsules+args.order_repeat:
        subprocess.run([sys.executable,str((path/'verify.py').resolve())],check=True)
    report=summarize(args.capsules,args.order_repeat)
    body=args.runtime_profile.read_bytes();report['runtime_profile']=json.loads(body);report['runtime_profile_sha256']=hashlib.sha256(body).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['aggregate']))


if __name__=='__main__':main()
