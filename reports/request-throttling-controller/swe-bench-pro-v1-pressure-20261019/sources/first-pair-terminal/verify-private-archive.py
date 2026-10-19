from pathlib import Path
import collections,hashlib,json,re,tarfile,io
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--evidence-directory',type=Path,required=True);args=parser.parse_args();root=args.evidence_directory;archive=root/'first-pair-private.tar.gz';assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()=='42dee5e748425410f1bd5d3259a4c6ea487cab8f00f107901cdbd99e02377846'
extract=root/'first-pair-terminal-extract';extract.mkdir(exist_ok=False)
def metric(text,name):
 rows=re.findall(r'^'+re.escape(name)+r'(?:\{[^\n]*?\})? ([^\n ]+)',text,re.M)
 return sum(float(x) for x in rows) if rows else None
with tarfile.open(archive) as t:
 members={x.name:x for x in t.getmembers()};payload={m.name:t.extractfile(m).read() for m in members.values() if m.isfile()};t.extractfile=lambda name:io.BytesIO(payload[name]);manifest=json.load(t.extractfile('file-manifest.json'))
 for name,entry in manifest.items():
  data=t.extractfile(name).read();assert len(data)==entry['bytes'];assert hashlib.sha256(data).hexdigest()==entry['sha256'],name
 def read(name):return json.load(t.extractfile(name))
 lock=read('execution/repository/config/swe-bench-pro-v1-lock.json')
 for name,sha in lock['agent_source_files'].items():assert hashlib.sha256(t.extractfile('agent/'+name).read()).hexdigest()==sha,name
 summary={'status':'INDEPENDENT_PRIVATE_ARCHIVE_VERIFIED_FAILURE','archive_sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),'archive_bytes':archive.stat().st_size,'checked_files':len(manifest),'arms':{},'benefit_claim':False,'valid_successful_work_comparison':False,'grading':'512 NOT_RUN_DEFERRED; not zero scores','observability_gap':'Start and terminal screenshots exist; not every intermediate 10-minute screenshot was captured. Raw per-second observations and logs retained.'};all_ids=[];all_programs=[]
 for arm in ['native','controller']:
  prefix='execution/first-pair/pair-1-'+arm+'/';b=read(prefix+'batch-report.json');reports=[];usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0};calls=0;arrival=[];response_hashes=collections.defaultdict(list)
  for i in range(256):
   d=prefix+f'task-{i:03d}/';r=read(d+'report-private.json');reports.append(r);all_programs.append(r['program_id']);all_ids.append(t.extractfile(d+'container-id.txt').read().decode().strip());arrival.append(read(d+'arrival.json')['actual_start_monotonic']);tr=read(d+'trajectory-private.json');assert r['program_release']['released'] is True;assert r['status']=='RetryError' and 'raised Timeout' in r['error'];assert r['agent_submitted'] is False
   assistant=[m for m in tr['messages'] if m['role']=='assistant']
   for step,m in enumerate(assistant):
    calls+=1
    for k in usage:usage[k]+=m['extra']['response']['usage'][k]
    response_hashes[(r['instance_id'],step)].append(hashlib.sha256(m['content'].encode()).hexdigest())
  elapsed=max(r['finished_monotonic'] for r in reports)-min(r['started_monotonic'] for r in reports)
  assert abs(elapsed-b['agent_batch_elapsed_seconds'])<1e-9 and usage==b['usage'] and calls==b['model_calls'];assert len(b['unresolved_grading'])==256
  for name in ['batch-report.json','observations-private.jsonl','engine-metrics-before.json','engine-metrics-after.json','batch-start.json']:
   p=extract/arm/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(prefix+name).read())
  obs=[json.loads(s) for s in t.extractfile(prefix+'observations-private.jsonl').read().decode().splitlines()];good=[s for s in obs if 'engine' in s];ratios=[metric(s['engine'],'vllm:kv_cache_usage_perc') for s in good];waiting=[metric(s['engine'],'vllm:num_requests_waiting') for s in good];capacities=[v['remaining_capacity'] for s in obs for v in s.get('scheduler',{}).get('backends',{}).values()];before=read(prefix+'engine-metrics-before.json');after=read(prefix+'engine-metrics-after.json')
  queries=metric(after,'vllm:prefix_cache_queries_total')-metric(before,'vllm:prefix_cache_queries_total');hits=metric(after,'vllm:prefix_cache_hits_total')-metric(before,'vllm:prefix_cache_hits_total')
  summary['arms'][arm]={'terminal_tasks':256,'terminal_statuses':dict(collections.Counter(x['status'] for x in reports)),'agent_submitted':0,'timeout_tasks':256,'elapsed_until_error_seconds':elapsed,'saved_response_usage':usage,'successful_model_responses':calls,'arrival_spread_seconds':max(arrival)-min(arrival),'observations':len(obs),'observer_error_samples':sum(any(k.endswith('_error') for k in s) for s in obs),'kv_occupancy_max':max(ratios),'kv_samples_at_or_above_90pct':sum(x>=.9 for x in ratios),'engine_waiting_max':max(waiting),'scheduler_remaining_min':min(capacities),'sampled_scheduler_overflow_count':sum(x<0 for x in capacities),'preemptions_counter_delta':metric(after,'vllm:num_preemptions_total')-metric(before,'vllm:num_preemptions_total'),'prefix_cache_query_tokens_delta':queries,'prefix_cache_hit_tokens_delta':hits,'prefix_cache_hit_ratio':hits/queries if queries else None,'after_batch_running_requests':metric(after,'vllm:num_requests_running'),'after_batch_waiting_requests':metric(after,'vllm:num_requests_waiting'),'response_hash_groups':len(response_hashes),'groups_with_identical_responses_across_all_observed_replicas':sum(len(v)>1 and len(set(v))==1 for v in response_hashes.values()),'unresolved_grades':256,'program_release_receipts':256}
 assert len(set(all_ids))==len(set(all_programs))==512
logs=json.loads((root/'first-pair-npu-terminal-private.json').read_text())['logs'];summary['logged_program_events']={}
for arm in ['native','controller']:
 text=next(v['log'] for k,v in logs.items() if k.endswith('-'+arm));events={}
 for event,pattern in [('paused','Paused program '),('marked_for_pause','Marked program '),('resumed','Resumed program ')]:
  ids=re.findall(r'^INFO:ThunderAgent.scheduler.router:'+re.escape(pattern)+r'(\S+)',text,re.M);events[event]={'events':len(ids),'unique_programs':len(set(ids))}
 summary['logged_program_events'][arm]=events
summary['elapsed_error_window_change_percent']=100*(summary['arms']['controller']['elapsed_until_error_seconds']/summary['arms']['native']['elapsed_until_error_seconds']-1)
(root/'first-pair-independent-verification-private.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
