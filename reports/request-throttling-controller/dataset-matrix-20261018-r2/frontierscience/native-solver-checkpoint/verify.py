"""Check a solver checkpoint, optionally against its private archive; no scoring."""
import argparse,collections,hashlib,json,pathlib,tarfile

def sha(data):return hashlib.sha256(data).hexdigest()
def digest(obj):return sha(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode())
def verify(root,archive=None):
 root=pathlib.Path(root)
 if (root/'SHA256SUMS').exists():
  for line in (root/'SHA256SUMS').read_text().splitlines():
   h,name=line.split('  ',1);relative=pathlib.Path(name);assert not relative.is_absolute() and '..' not in relative.parts;assert sha((root/relative).read_bytes())==h,name
 p=json.loads((root/'checkpoint.json').read_text());assert p['state']=='SOLVER_COMPLETE_GRADING_PENDING' and p['stage']=='quality-frontier-all-native'
 assert set(p['tracks'])=={'olympiad','research'};assert len(p['request_record_hashes'])==160
 for track,n in [('olympiad',100),('research',60)]:
  d=p['tracks'][track];assert d['planned']==n and d['responses_completed']+d['request_failures']==n;assert sum(d['finish_reasons'].values())==n
 if archive:
  archive=pathlib.Path(archive);assert sha(archive.read_bytes())==p['private_raw_sha256']
  with tarfile.open(archive) as tar:
   def raw(name):return tar.extractfile(name).read()
   def read(name):return json.loads(raw(name))
   c=read('plan/contract.json');assert sha(raw('plan/contract.json'))==p['contract_sha256'];assert sha(raw('plan/tasks.json'))==p['tasks_sha256'];assert c['client_code']==p['client_code_sha256']
   for name,h in c['client_code'].items():assert sha(raw('plan/client_source/'+name))==h,name
   tasks=read('plan/tasks.json');by_id={t['case_id']:t for t in tasks};assert len(by_id)==len(tasks)==160
   meta=read('solver/run.json');index=read('solver/index.json');assert sha(raw('solver/run.json'))==index['run_sha256'];assert sha(raw('solver/index.json'))==p['request_index_sha256'];assert index['files']==p['request_record_hashes'];assert set(meta['case_ids'])==set(by_id) and meta['contract_sha256']==p['contract_sha256'] and meta['stage']=='solver'
   stats={t:{'planned':n,'responses_completed':0,'request_failures':0,'finish_reasons':collections.Counter(),'prompt_tokens':0,'output_tokens':0} for t,n in [('olympiad',100),('research',60)]}
   for name,h in index['files'].items():
    data=raw('solver/requests/'+name);assert sha(data)==h;r=json.loads(data);record_hash=r.pop('record_sha256');assert digest(r)==record_hash;task=by_id[r['case_id']];assert r['contract_sha256']==p['contract_sha256'] and r['track']==task['track'];assert digest(r['request'])==r['request_sha256']
    expected=dict(c['solver']['request'],model=c['solver']['model'],messages=[{'role':'user','content':task['problem']}],stream=False);assert r['request']==expected
    d=stats[task['track']];d['responses_completed' if r['status']=='ok' else 'request_failures']+=1;d['finish_reasons'][r.get('finish_reason')]+=1
    if r['status']=='ok':
     response=json.loads(r['raw_response']);assert response==r['response'];assert response['choices'][0]['message']['content']==r['content'];assert response['choices'][0]['finish_reason']==r['finish_reason'];assert response['usage']==r['usage']
     u=r['usage'];assert all(type(u[k]) is int and u[k]>=0 for k in ['prompt_tokens','completion_tokens','total_tokens']);assert u['prompt_tokens']+u['completion_tokens']==u['total_tokens'];d['prompt_tokens']+=u['prompt_tokens'];d['output_tokens']+=u['completion_tokens']
   for d in stats.values():d['finish_reasons']=dict(d['finish_reasons'])
   assert stats==p['tracks']
 return {'checkpoint':'PASS','private_archive_verified':bool(archive),'responses':sum(t['responses_completed'] for t in p['tracks'].values()),'quality_score':'PENDING','comparison':'PENDING'}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--archive');a=parser.parse_args();print(json.dumps(verify(pathlib.Path(__file__).parent,a.archive)))
