"""One run and cluster; model, router, client are independently visible Ray jobs."""
import json,os,shlex,time,urllib.request
from pathlib import Path
import ray
from ray.job_submission import JobSubmissionClient
root=Path(os.environ['REPO_ROOT']);name=os.environ['EXPERIMENT_NAME'];out=root/'data/results/workload'/name;out.mkdir(parents=True,exist_ok=False)
ray.init(address='auto')
@ray.remote
def validate_environment():
 import acl
 return dict(node=ray.get_runtime_context().get_node_id(),acl=acl.__file__)
nodes=[n for n in ray.nodes() if n['Alive']];proof=ray.get([validate_environment.options(resources={'node:'+n['NodeManagerAddress']:0.001}).remote() for n in nodes],timeout=120);assert len(proof)==4
(out/'ascend-environment-validation.json').write_text(json.dumps(proof,indent=2));ray.shutdown()
c=JobSubmissionClient('http://127.0.0.1:8265');env={k:os.environ[k] for k in ['REPO_ROOT','EXPERIMENT_NAME','PYTHONPATH','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','VLLM_SERVER_DEV_MODE','PYTHONUNBUFFERED','RAY_DEDUP_LOGS','GROWING_CONTEXT_PILOT','GROWING_CONTEXT_PROGRAMS','GROWING_CONTEXT_WORKLOAD']};resource={'node:'+os.environ['EXPERIMENT_HEAD_IP']:0.001};ids={role:role+'--'+name for role in ['vllm-model-serving','thunderagent-program-router','program-workload-client']};submitted=[]
(out/'service-jobs.json').write_text(json.dumps(ids,indent=2))
def submit(role,args):
 identity=ids[role];c.submit_job(submission_id=identity,entrypoint=shlex.join(args),runtime_env={'env_vars':env},entrypoint_resources=resource);submitted.append(role);print('[submitted] '+identity,flush=True)
def health(role,url,limit):
 deadline=time.monotonic()+limit
 while time.monotonic()<deadline:
  status=c.get_job_status(ids[role]).value
  if status in ['FAILED','STOPPED','SUCCEEDED']:raise RuntimeError(f'{role} exited before health: {status}')
  try:
   with urllib.request.urlopen(url,timeout=5) as r:
    if r.status==200:print('[healthy] '+ids[role],flush=True);return
  except Exception:pass
  time.sleep(5)
 raise TimeoutError(role+' health timeout')
def logs():
 for role,file in [('vllm-model-serving','engine.log'),('thunderagent-program-router','thunderagent.log'),('program-workload-client','client.log')]:
  if role in submitted:(out/file).write_text(c.get_job_logs(ids[role]))
try:
 configuration={'programs':int(os.environ['GROWING_CONTEXT_PROGRAMS']),'rounds':int(os.environ['GROWING_CONTEXT_PROGRAMS'])*20,'initial_input_tokens':4096,'growth_tokens_per_round':640,'output_tokens':128,'tool_wait_seconds':3,'arrival_rate':1,'tensor_parallel_size':4,'acting_token_decay':False,'scheduler_interval_seconds':5,'warmup':'smoke then prefix reset; router calibration retained','service_jobs':ids,'head_ip':os.environ['EXPERIMENT_HEAD_IP'],'input_semantics':'synthetic accumulated history, actual outputs excluded'}
 (out/'effective-experiment-configuration.json').write_text(json.dumps(configuration,indent=2))
 submit('vllm-model-serving',['vllm','serve','/models/Qwen2.5-14B-Instruct','--served-model-name','qwen2.5-14b','--host','0.0.0.0','--port','8100','--distributed-executor-backend','ray','--tensor-parallel-size','4','--enforce-eager','--enable-prefix-caching']);health('vllm-model-serving','http://127.0.0.1:8100/health',1800)
 submit('thunderagent-program-router',['python',str(root/'experiments/control_plane/thunderagent_router.py'),'--backends','http://127.0.0.1:8100','--port','8000','--router','tr','--disable-backend-connection-reuse','--disable-request-timeouts','--profile-dir',str(out/'thunderagent-profile')]);health('thunderagent-program-router','http://127.0.0.1:8000/health',120)
 submit('program-workload-client',['python',str(root/'experiments/workload/run_growing_context_client.py')])
 last=''
 while True:
  state=c.get_job_status(ids['program-workload-client']).value
  text=c.get_job_logs(ids['program-workload-client'])
  if text!=last:print(text[len(last):],end='',flush=True);last=text
  logs()
  if state=='SUCCEEDED':break
  if state in ['FAILED','STOPPED']:raise RuntimeError('client '+state)
  for role in ['vllm-model-serving','thunderagent-program-router']:
   assert c.get_job_status(ids[role]).value=='RUNNING',role+' exited during client'
  time.sleep(15)
 assert json.loads((out/'validation.json').read_text())['valid']
 print('[validated] all service jobs, smoke, workload and history inheritance',flush=True)
finally:
 logs()
 (out/'service-job-final-status.json').write_text(json.dumps({role:c.get_job_status(ids[role]).value for role in submitted},indent=2))
 for role in reversed(submitted):
  if c.get_job_status(ids[role]).value in ['PENDING','RUNNING']:c.stop_job(ids[role])
 logs()
