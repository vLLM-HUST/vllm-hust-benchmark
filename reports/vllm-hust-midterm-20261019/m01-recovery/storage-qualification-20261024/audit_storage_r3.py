"""Independent derived audit; preserves all sealed raw records and verdicts."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,'/root/m01-storage-runtime-20261023')
from audit_wave_capture import audit
from audit_draft_inputs import validate as validate_inputs
from audit_draft_lengths import validate as validate_lengths
from classify_capture_skips import classify
from run_capture_b1 import check_client
from worker_wave_identity import request_key

root=Path('/root/m01-capture-b1-storage-r3'); base=Path(__file__).resolve().parent
run='capture-b1-storage-r3'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
verified=[]
for line in (root/'SHA256SUMS').read_text().splitlines():
 expected,name=line.split(maxsplit=1);name=name.lstrip('*');path=root/name
 assert path.resolve().is_relative_to(root.resolve())
 assert digest(path)==expected,name
 verified.append(name)
receipt=json.loads((root/'receipt.json').read_text())
assert receipt['status']=='diagnostic_capture_joined'
assert receipt['sessions_drained'] and receipt['all_devices_released']
s=json.loads((root/'scheduler-53873.json').read_text())
workers=[json.loads(p.read_text()) for p in sorted(root.glob('worker-*.json'))]
joined=audit(s,workers);assert joined['status']=='joined',joined
workload=check_client(root,run)
worker_checks=[]
for w in workers:
 worker_checks.append({'rank':w['rank'],'lengths':validate_lengths(w),'inputs':validate_inputs(w),'progress_source_protection':w['progress_source_protection']})
classification=classify(s['events']);assert classification['unclassified']==0
counts=Counter(); retained=Counter()
for e in s['events']:
 if e['event']=='output_update':
  obs=e['observation'];counts[e['request_key']]+=obs['appended_count']
  for key in ('raw_accepted_drafts','retained_drafts','retained_first_draft','retained_second_draft'):
   assert type(obs[key]) is int and obs[key]>=0
   retained[key]+=obs[key]
assert len(counts)==200 and set(counts.values())=={256}
engine=[json.loads(x) for x in (root/'kvmat-engine.jsonl').read_text().splitlines()]
policy=[json.loads(x) for x in (root/'kvmat-policy.jsonl').read_text().splitlines()]
assert len(engine)==len(policy)==200
assert {request_key(run,e['request_id']) for e in engine}==set(counts)
a={hashlib.sha256(p['runtime_cache_salt'].encode()).hexdigest():p['prompt_tokens'] for p in policy}
b={e['cache_salt_sha256']:e['prompt_tokens'] for e in engine}
assert len(a)==len(b)==200 and a==b
assert all(p['runtime_effective_decision']=='recompute' and p['reused_tokens']==0 for p in policy)
micro=[]
for path in sorted(root.glob('microflow-*.jsonl')):
 rows=[json.loads(x) for x in path.read_text().splitlines()]
 c=Counter(x['event'] for x in rows);assert c['graph_replay']>0
 micro.append({'file':path.name,'event_counts':dict(c)})
live=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  try:
   fields=(p/'stat').read_text().rsplit(')',1)[1].split()
   if int(fields[3]) in {50081,50314,82668}:live.append(int(p.name))
  except FileNotFoundError:pass
assert not live,live
npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=20,check=True).stdout
assert all(f'No running processes found in NPU {i}' in npu for i in (3,6))
(base/'storage-r3-final-npu.txt').write_text(npu)
bench=json.loads((root/'mmlu-pro.json').read_text())
result={'run_id':run,'observed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'status':'diagnostic_capture_and_derived_host_counts_verified','raw_evidence_directory':str(root),'verified_sha_entries':len(verified),'raw_manifest_sha256':digest(root/'SHA256SUMS'),'workload':workload,'join':joined,'workers':worker_checks,'original_retained_count_complete':s['retained_count_complete'],'original_unclassified_event_count':s['unclassified_requests'],'derived_nonupdate_classification':classification,'derived_output_counts':{'requests':len(counts),'per_request':256,'total':sum(counts.values()),'matches_engine_request_identities':True},'derived_host_draft_counts':dict(retained),'kv_materialization':{'engine_admissions':200,'policy_records':200,'salt_bindings_verified':True,'decisions':'recompute','reuse_tokens':0,'scope':'actual request admission and recompute control; no cache reuse benefit claimed'},'microflow':micro,'client_exit_code':receipt['client_exit_code'],'service_exit_codes':receipt['exit_codes'],'remaining_sessions':live,'physical_devices_released':[3,6],'memory_events':Path('/sys/fs/cgroup/memory.events').read_text(),'diagnostic_only_metrics':{'client_seconds':bench['duration'],'output_tokens_per_second':bench['output_throughput'],'scope':'includes instrumentation and cold request kernel work; not formal B1 performance'},'effective_flops':None,'reference_time':None,'m01_attainment':None,'remaining_work':['Same-run device arithmetic and reference-time accounting has not been established by this transport/host audit.','Host nonupdate classification does not classify useful prefill device work.','Draft device stream ordering remains unqualified; do not transfer these counts to another run timing.'],'audit_source_sha256':digest(Path(__file__))}
with (base/'storage-r3-final-audit.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
