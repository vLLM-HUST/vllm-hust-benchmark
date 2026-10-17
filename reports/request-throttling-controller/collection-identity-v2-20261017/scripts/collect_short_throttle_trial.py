#!/usr/bin/env python3
"""Collect terminal short-trial evidence after verifying Ray and Pod ownership.

Private archive only: deployment addresses and token payloads may be present.
Never use this archive as the public export.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REMOTE = r'''
import json,pathlib,subprocess,sys,urllib.request
p=json.loads(pathlib.Path(PLAN).read_text());i=p['payload_identity']
assert p['run_id']==RUN_ID and JOB.startswith(RUN_ID+'--'), 'Run/job identity mismatch'
root=pathlib.Path(ROOT_OVERRIDE or p['remote_root'])
assert root.name==RUN_ID, 'Collection root does not match run ID'
for kind,name,uid in [('pod',i['pod'],i['pod_uid']),('raycluster',i['cluster'],i['cluster_uid'])]:
    d=json.loads(subprocess.check_output(['kubectl','-n',i['namespace'],'get',kind,name,'-o','json']))
    assert d['metadata']['uid']==uid, 'Ownership identity mismatch'
j=json.load(urllib.request.urlopen(p['ray_address']+'/api/jobs/'+JOB,timeout=15))
assert j['status'] in ['SUCCEEDED','FAILED','STOPPED'], 'Refuse inconsistent archive of running measurement'
terminal={"job_id":JOB,"ray_status":j["status"],"run_id":RUN_ID}
probe = """import pathlib,sys,tarfile,json,io
root=pathlib.Path(ROOT)
for report_name in ['window/window-report.json','trial/trial.json']:
    path=root/report_name
    if path.exists():
        report=json.loads(path.read_text())
        assert pathlib.Path(report['plan']['output']).parent==root, 'Report belongs to a different run'
        assert report['plan']['seed']==EXPECTED_SEED, 'Collected seed differs from frozen local plan'
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    body=json.dumps(TERMINAL).encode();info=tarfile.TarInfo('collection-terminal.json');info.size=len(body);archive.addfile(info,io.BytesIO(body))
    for name in ['trial','window','workload','warmup-workload','experiments','src','capacity-evidence.json','hybrid-audit.json']:
        path=root/name
        if path.exists(): archive.add(path,arcname=name)
""".replace('ROOT',repr(str(root))).replace('EXPECTED_SEED',repr(SEED)).replace('TERMINAL',repr(terminal))
subprocess.run(['kubectl','-n',i['namespace'],'exec',i['pod'],'-c','ray-head','--','python3','-c',probe],check=True)
'''


def validate_collection_identity(plan, run_id, job_id, remote_root=None):
    if plan['run_id'] != run_id or not job_id.startswith(run_id + '--'):
        raise ValueError('Run/job identity mismatch')
    root = remote_root or plan['remote_root']
    if Path(root).name != run_id:
        raise ValueError('Collection root does not match run ID')
    return root


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--ssh-host',required=True)
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--remote-root',help='Explicit collection-only correction; must end with this run ID')
    args=parser.parse_args()
    plan=json.loads((args.run_directory/'plan.json').read_text())
    validate_collection_identity(plan,args.run_directory.name,args.job_id,args.remote_root)
    frozen=json.loads((args.run_directory/('window-plan.json' if (args.run_directory/'window-plan.json').exists() else 'trial-plan.json')).read_text())
    code=REMOTE.replace('ROOT_OVERRIDE',repr(args.remote_root)).replace('RUN_ID',repr(args.run_directory.name)).replace('SEED',repr(frozen['seed'])).replace('PLAN',repr(plan['operator_root']+'/plan.json')).replace('JOB',repr(args.job_id))
    archive=args.run_directory/'raw-and-source.tar.gz'
    temporary=archive.with_suffix('.partial')
    if archive.exists() or temporary.exists():
        raise ValueError('Never overwrite an existing archive or partial collection')
    with temporary.open('xb') as output:
        result=subprocess.run(['ssh','-o','ConnectTimeout=10',args.ssh_host,'python3','-'],
            input=code.encode(),stdout=output,stderr=subprocess.PIPE,timeout=180)
    (args.run_directory/'collection.log').write_bytes(result.stderr)
    result.check_returncode()
    temporary.rename(archive)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.run_directory/'raw-and-source.sha256').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps(dict(archive=str(archive),sha256=digest,scope='private terminal evidence; public export requires sanitization')))


if __name__=='__main__':
    main()
