#!/usr/bin/env python3
"""Frozen offline replay -> Ray Jobs -> private collection -> public verification.

A config supplies an existing ready RIDE deployment, or an explicit argv invoking
its standard provisioner. This runner does not implement a scheduling policy.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
TERMINAL={'SUCCEEDED','FAILED','STOPPED'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write_once(path,value):
    body=json.dumps(value,indent=2)+'\n'
    if path.exists():
        if path.read_text()!=body:raise ValueError('Frozen artifact differs: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:stream.write(body)


def command(argv,**kwargs):
    return subprocess.run([str(x) for x in argv],check=True,**kwargs)


def remote(host,code):
    result=command(['ssh','-o','ConnectTimeout=10',host,'python3','-'],input=code,text=True,capture_output=True,timeout=180)
    return json.loads(result.stdout)


def load_config(path):
    config=read(path)
    for key in ['deployment_plan','window_template','workload','output','ride_root','runtime_profile']:
        value=Path(config[key]);config[key]=value if value.is_absolute() else (path.parent/value).resolve()
    runs=config['runs']
    if not runs or len({r['run_id'] for r in runs})!=len(runs):raise ValueError('Unique explicit run IDs required')
    for run in runs:
        if not re.fullmatch('[a-z0-9][a-z0-9-]{0,100}',run['run_id']):raise ValueError('Invalid run ID')
        if run['order'] not in ['OFF_ON','ON_OFF']:raise ValueError('Explicit ON/OFF order required')
        if run.get('job_id') and not run['job_id'].startswith(run['run_id']+'--'):raise ValueError('Job/run identity mismatch')
    primary=[r['seed'] for r in runs if not r.get('order_repeat',False)]
    if len(set(primary))!=len(primary):raise ValueError('Duplicate primary seeds; mark order repeats explicitly')
    if config.get('benchmark_repo'):
        value=Path(config['benchmark_repo']);config['benchmark_repo']=value if value.is_absolute() else (path.parent/value).resolve()
    if config.get('summary_output'):
        value=Path(config['summary_output']);config['summary_output']=value if value.is_absolute() else (path.parent/value).resolve()
    return config


def prepare(config,spec):
    deployment=read(config['deployment_plan']);template=read(config['window_template'])
    workload=config['workload'];data=workload/'sessions.csv'
    with data.open() as stream:inputs=list(csv.DictReader(stream))
    if not inputs or any(float(row['arrival_time'])!=0 for row in inputs):raise ValueError('Offline arrivals must all be time0')
    if template.get('arrival_mode')!='offline-burst' or template.get('concurrency')!=0 or template.get('client_concurrency_limit') is not None:
        raise ValueError('Offline template must have no client concurrency cap')
    if sha(data)!=template['expected_sessions_sha256']:raise ValueError('Input SHA differs from template')
    if len({row['session_id'] for row in inputs})!=template['programs']:raise ValueError('Batch size differs from input')
    run=spec['run_id'];directory=config['output']/run
    if directory.exists():raise ValueError('Run exists; use status/finish or all to resume it')
    plan=copy.deepcopy(deployment);plan.update(run_id=run,operator_root=config.get('operator_parent','/tmp')+'/'+run+'-operator',remote_root=config.get('remote_parent','/workspace/home')+'/'+run)
    if not plan['remote_root'].startswith('/workspace/home/') or '..' in Path(plan['remote_root']).parts:raise ValueError('Unsafe remote root')
    window=copy.deepcopy(template);window.update(seed=spec['seed'],workload=plan['remote_root']+'/workload',output=plan['remote_root']+'/window',replayer=plan['remote_root']+'/experiments/workload/replay_sessions.py')
    points={p['scheduling_enabled']:p for p in window['points']}
    if set(points)!={False,True} or len(window['points'])!=2:raise ValueError('Exactly one ON/OFF pair required')
    window['points']=[points[False],points[True]] if spec['order']=='OFF_ON' else [points[True],points[False]]
    directory.mkdir(parents=True);write_once(directory/'plan.json',plan);write_once(directory/'window-plan.json',window)
    files={'plan.json':directory/'plan.json','window-plan.json':directory/'window-plan.json'}
    for name in ['sessions.csv','manifest.json']:files['workload/'+name]=workload/name
    files.update({'experiments/workload/'+p.name:p for p in (ROOT/'experiments/workload').glob('*.py')})
    files['ride-scripts/deployment_payload.py']=config['ride_root']/'scripts/deployment_payload.py'
    snapshot=directory/'payload';metadata=[]
    for name,path in files.items():
        destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        metadata.append(dict(path=name,sha256=sha(destination)))
    write_once(directory/'source-manifest.json',metadata)
    return directory


IDENTITY='''
i=p.get('payload_identity',p.get('payload'))
for kind,name,uid in [('pod',i['pod'],i['pod_uid']),('raycluster',i['cluster'],i['cluster_uid'])]:
 d=json.loads(subprocess.check_output(['kubectl','-n',i['namespace'],'get',kind,name,'-o','json']))
 assert d['metadata']['uid']==uid,'Pod/cluster identity changed'
 if kind=='pod':assert urllib.parse.urlparse(p['ray_address']).hostname==d['status']['podIP'],'Ray address differs from current owned Pod'
'''


def status(host,directory,job_id=None):
    plan=read(directory/'plan.json');run=plan['run_id'];job=job_id or run+'--offline-window-v1'
    receipt=directory/'submission.json'
    if receipt.exists():job=read(receipt)['job_id']
    code='import json,subprocess,urllib.request\np='+repr(plan)+'\n'+IDENTITY+'''\ntry:
 job=json.load(urllib.request.urlopen(p['ray_address']+'/api/jobs/'+JOB,timeout=15))
except urllib.error.HTTPError as e:
 if e.code!=404:raise
 job={'status':'NOT_SUBMITTED'}
print(json.dumps({'job_id':JOB,'status':job['status']}))
'''.replace('JOB',repr(job))
    result=remote(host,code);(directory/'pipeline-status.json').write_text(json.dumps(result,indent=2)+'\n');return result


def launch(host,directory):
    plan=read(directory/'plan.json');run=plan['run_id'];job=run+'--offline-window-v1'
    if (directory/'submission.json').exists():raise ValueError('Submission intent exists; inspect status instead of resubmitting')
    manifest=read(directory/'source-manifest.json');buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for item in manifest:
            path=directory/'payload'/item['path']
            if sha(path)!=item['sha256']:raise ValueError('Frozen payload changed')
            archive.add(path,arcname=item['path'])
    # Exact operator path, refusing existing directories. RIDE validates Pod/UID
    # and immutable payload contents before writing into workspace storage.
    operator=plan['operator_root']
    command(['ssh','-o','ConnectTimeout=10',host,'mkdir -- '+shlex.quote(operator)],capture_output=True)
    command(['ssh',host,'tar -xzf - -C '+shlex.quote(operator)],input=buffer.getvalue(),capture_output=True)
    request=dict(job_id=job,run_id=run)
    write_once(directory/'submission.json',request)  # Before POST: unknown outcomes must not be blindly retried.
    code='import json,pathlib,subprocess,sys,urllib.request,fcntl\np='+repr(plan)+'\n'+IDENTITY+'''
lock=open('/tmp/offline-throttle-'+i['cluster_uid']+'.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
jobs=json.load(urllib.request.urlopen(p['ray_address']+'/api/jobs/',timeout=15))
assert not any(j['status'] in ['PENDING','RUNNING'] and 'run_throttle_window_trial.py' in j.get('entrypoint','') for j in jobs),'Another window is active'
sys.path.insert(0,p['operator_root']+'/ride-scripts')
from deployment_payload import prepare_payload
files=[dict(source=p['operator_root']+'/'+x['path'],path=x['path'],sha256=x['sha256']) for x in MANIFEST if not x['path'].startswith('ride-scripts/')]
prepare_payload(dict(i,destination=p['remote_root'],files=files))
body=dict(submission_id=JOB,entrypoint='python '+p['remote_root']+'/experiments/workload/run_throttle_window_trial.py --plan '+p['remote_root']+'/window-plan.json',runtime_env=p['runtime_env'],entrypoint_num_cpus=.1,entrypoint_resources=p['execution_resources'])
result=json.load(urllib.request.urlopen(urllib.request.Request(p['ray_address']+'/api/jobs/',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=30))
print(json.dumps(result))
'''.replace('MANIFEST',repr(manifest)).replace('JOB',repr(job))
    try:return remote(host,code)
    except subprocess.CalledProcessError as error:
        (directory/'launch-error.log').write_text((error.stdout or '')+(error.stderr or ''))
        raise RuntimeError('Launch failed or has unknown outcome; inspect launch-error.log and Ray status before retry') from error


def finish(host,directory,job_id):
    frozen=read(directory/'window-plan.json');plan=read(directory/'plan.json')
    archive=directory/'raw-and-source.tar.gz'
    if not archive.exists():
        command([sys.executable,ROOT/'scripts/collect_short_throttle_trial.py','--run-directory',directory,'--ssh-host',host,'--job-id',job_id,'--remote-root',str(Path(frozen['output']).parent)])
    digest=sha(archive)
    with tarfile.open(archive) as source:
        terminal=json.load(source.extractfile('collection-terminal.json'))
        measured=json.load(source.extractfile('window/window-report.json'))
    if terminal['job_id']!=job_id or terminal['ray_status']!='SUCCEEDED':raise ValueError('Wrong/non-successful terminal archive; preserve without qualification')
    if measured['plan']['seed']!=frozen['seed'] or measured['plan']['output']!=frozen['output']:raise ValueError('Archive seed/output mismatch')
    public=directory/'public'
    if not public.exists():command([sys.executable,ROOT/'scripts/export_short_throttle_trial.py','--archive',archive,'--expected-sha256',digest,'--output',public])
    config=read(public/'config.json');report=read(public/'report.json')
    if config['seed']!=frozen['seed'] or config['expected_sessions_sha256']!=frozen['expected_sessions_sha256'] or report['private_archive_sha256']!=digest or report['ray_terminal']['job_id']!=job_id:raise ValueError('Public capsule belongs to another plan/archive')
    command([sys.executable,public/'verify.py'])
    receipt=dict(status='VERIFIED',run_id=plan['run_id'],job_id=job_id,archive_sha256=digest,report_sha256=sha(public/'report.json'),finite_makespan_claim=False)
    write_once(directory/'pipeline-receipt.json',receipt);return receipt


def run(config,action):
    deployment=config['deployment_plan']
    if not deployment.exists():
        if action!='all' or not config.get('provision_command'):raise ValueError('Ready deployment plan missing')
        # An explicit argv delegates Coder/model/Controller setup to its existing
        # platform entrypoint; no shell interpolation or shared-template changes.
        command(config['provision_command'])
        if not deployment.exists():raise ValueError('Provisioner did not produce its promised ready plan')
    directories=[]
    for spec in config['runs']:
        directory=config['output']/spec['run_id'];directories.append(directory)
        if action=='prepare' or (action=='all' and not directory.exists()):prepare(config,spec)
        if action=='prepare':continue
        frozen=read(directory/'window-plan.json')
        order=['ON' if p['scheduling_enabled'] else 'OFF' for p in frozen['points']]
        if frozen['seed']!=spec['seed'] or order!=spec['order'].split('_'):raise ValueError('Resume spec differs from frozen seed/order')
        if sha(config['workload']/'sessions.csv')!=frozen['expected_sessions_sha256']:raise ValueError('Resume input differs from frozen input')
        if action=='launch':print(json.dumps(launch(config['ssh_host'],directory)));continue
        current=status(config['ssh_host'],directory,spec.get('job_id'))
        print(json.dumps(current),flush=True)
        if action=='status':continue
        if action=='all' and current['status']=='NOT_SUBMITTED' and not (directory/'submission.json').exists():
            launch(config['ssh_host'],directory)
            current=status(config['ssh_host'],directory,spec.get('job_id'))
        if action=='all':
            deadline=time.monotonic()+config.get('wait_budget_seconds',900)
            while current['status'] not in TERMINAL:
                if current['status']=='NOT_SUBMITTED':raise ValueError('Unknown submission outcome; reconcile Ray before retry')
                if time.monotonic()>deadline:raise TimeoutError('Watch budget expired; task remains remote, use status/resume')
                time.sleep(min(15,config.get('poll_seconds',15)));current=status(config['ssh_host'],directory,spec.get('job_id'))
        if current['status']!='SUCCEEDED':raise ValueError('Measurement not successful; preserve evidence and diagnose: '+current['status'])
        print(json.dumps(finish(config['ssh_host'],directory,current['job_id'])),flush=True)
    if action in ['all','finish']:
        primary=[d/'public' for d,s in zip(directories,config['runs']) if not s.get('order_repeat',False)]
        repeats=[d/'public' for d,s in zip(directories,config['runs']) if s.get('order_repeat',False)]
        summary=config.get('summary_output',config['output']/'summary.json')
        if not summary.exists():
            argv=[sys.executable,ROOT/'scripts/summarize_throttle_windows.py','--capsules',*primary,'--runtime-profile',config['runtime_profile'],'--output',summary]
            if repeats:argv+=['--order-repeat',*repeats]
            command(argv)
        else:
            from summarize_throttle_windows import summarize
            expected=read(summary);actual=summarize(primary,repeats)
            if actual!={k:v for k,v in expected.items() if k not in ['runtime_profile','runtime_profile_sha256']}:raise ValueError('Existing summary differs')
            if sha(config['runtime_profile'])!=expected['runtime_profile_sha256']:raise ValueError('Runtime profile differs from frozen summary')
        if config.get('benchmark_repo'):
            base=Path(config['benchmark_repo'])/'reports/request-throttling-controller'
            for directory in directories:
                target=base/directory.name
                if target.exists():
                    if (target/'SHA256SUMS').read_bytes()!=(directory/'public/SHA256SUMS').read_bytes():raise ValueError('Backup differs; never overwrite it')
                else:shutil.copytree(directory/'public',target,ignore=shutil.ignore_patterns('__pycache__'))
            write_once(base/config.get('summary_name','offline-window-summary')/'report.json',read(summary))
            print('Public backup staged locally; Git push and website publication are separate review steps')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','launch','status','finish','all'])
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();run(load_config(args.config.resolve()),args.action)


if __name__=='__main__':main()
