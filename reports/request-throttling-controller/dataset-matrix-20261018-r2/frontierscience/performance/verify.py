"""Verify published performance arithmetic, optionally bound to the private archive."""
import argparse,hashlib,json,math,pathlib,statistics,tarfile

def sha(data):return hashlib.sha256(data).hexdigest()
def verify(root,archive=None):
 root=pathlib.Path(root)
 if (root/'SHA256SUMS').exists():
  for line in (root/'SHA256SUMS').read_text().splitlines():
   h,n=line.split('  ',1);assert pathlib.Path(n).name==n;assert sha((root/n).read_bytes())==h
 p=json.loads((root/'checkpoint.json').read_text());assert p['state']=='THREE_REPEAT_PERFORMANCE_COMPLETE';rows=p['records'];assert len(rows)==8
 assert len({r['stage'] for r in rows})==8 and {r['input_sha256'] for r in rows}=={p['input_sha256']}
 for arm in ['native','controller']:
  selected=[r for r in rows if r['arm']==arm];assert {r['repetition'] for r in selected}=={'warmup',1,2,3}
  for k,v in p['aggregate'][arm].items():assert v==statistics.median(r[k] for r in selected if r['repetition']!='warmup')
 for r in rows:
  assert r['completed']==160 and r['failed']==0 and r['duration']>0;assert r['total_output_tokens']==40960
  for k,total in [('output_throughput',r['total_output_tokens']),('request_throughput',r['completed']),('total_token_throughput',r['total_input_tokens']+r['total_output_tokens'])]:assert math.isclose(r[k],total/r['duration'],rel_tol=1e-9)
 if archive:
  assert sha(pathlib.Path(archive).read_bytes())==p['private_raw_sha256']
  with tarfile.open(archive) as tar:
   def raw(n):return tar.extractfile(n).read()
   queue=json.loads(raw('queue-plan.json'));assert sha(raw('queue-plan.json'))==p['source_queue_sha256'];assert queue['run_id']==p['run_id']
   for r in rows:
    base='performance/'+r['stage'];data=raw(base+'/outputs/frontier.json');assert sha(data)==r['raw_sha256'];d=json.loads(data)
    assert d['num_prompts']==160 and d['max_concurrency'] is None and d['request_rate']=='inf'
    assert sha(raw(base+'/inputs/frontier.materialized.jsonl'))==r['input_sha256'];assert len(raw(base+'/inputs/frontier.materialized.jsonl').splitlines())==160
    cmd=raw(base+'/outputs/frontier.cmd.txt').decode();assert '--custom-output-len 256' in cmd and '--temperature 0' in cmd and '--seed 0' in cmd and '--disable-shuffle' in cmd
    for k,v in r.items():
     if k in d:assert d[k]==v,k
    summary=json.loads(raw(base+'/outputs/frontier.summary.json'))
    for k in ['completed','failed','duration','output_throughput','mean_ttft_ms','mean_tpot_ms']:assert summary[k]==d[k]
 return {'performance':'PASS','private_archive_verified':bool(archive),'formal_points':6,'warmups_excluded':2,'delay_provenance':'vllm client aggregate statistics; per-request timing records not retained, so delay means cannot be independently recalculated','aggregate':p['aggregate']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive');a=p.parse_args();print(json.dumps(verify(pathlib.Path(__file__).parent,a.archive)))
