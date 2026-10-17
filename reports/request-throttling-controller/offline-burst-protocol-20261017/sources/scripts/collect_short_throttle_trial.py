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
for kind,name,uid in [('pod',i['pod'],i['pod_uid']),('raycluster',i['cluster'],i['cluster_uid'])]:
    d=json.loads(subprocess.check_output(['kubectl','-n',i['namespace'],'get',kind,name,'-o','json']))
    assert d['metadata']['uid']==uid, 'Ownership identity mismatch'
j=json.load(urllib.request.urlopen(p['ray_address']+'/api/jobs/'+JOB,timeout=15))
assert j['status'] in ['SUCCEEDED','FAILED','STOPPED'], 'Refuse inconsistent archive of running measurement'
terminal={"job_id":JOB,"ray_status":j["status"]}
probe = """import pathlib,sys,tarfile,json,io
root=pathlib.Path(ROOT)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    body=json.dumps(TERMINAL).encode();info=tarfile.TarInfo('collection-terminal.json');info.size=len(body);archive.addfile(info,io.BytesIO(body))
    for name in ['trial','window','workload','warmup-workload','experiments','src','capacity-evidence.json','hybrid-audit.json']:
        path=root/name
        if path.exists(): archive.add(path,arcname=name)
""".replace('ROOT',repr(p['remote_root'])).replace('TERMINAL',repr(terminal))
subprocess.run(['kubectl','-n',i['namespace'],'exec',i['pod'],'-c','ray-head','--','python3','-c',probe],check=True)
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--ssh-host',required=True)
    parser.add_argument('--job-id',required=True)
    args=parser.parse_args()
    plan=json.loads((args.run_directory/'plan.json').read_text())
    code=REMOTE.replace('PLAN',repr(plan['operator_root']+'/plan.json')).replace('JOB',repr(args.job_id))
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
