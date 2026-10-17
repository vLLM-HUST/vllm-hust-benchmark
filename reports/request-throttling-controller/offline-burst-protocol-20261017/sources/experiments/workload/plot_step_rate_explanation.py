"""Explain completion-rate changes with raw requests, tool waits and token counters."""
import json,re,csv,argparse,datetime,textwrap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();fig,axs=plt.subplots(5,2,figsize=(15,17),sharex=True);axs=axs.ravel();export=[]
for folder,label,color,ls in [('vllm-direct-service','Direct vLLM','#D55E00','--'),('thunderagent-program-throttling','ThunderAgent program throttling (fresh backend connections)','#0072B2','-')]:
 d=a.root/folder;s=json.loads((d/'summary.json').read_text());end=s['result']['makespan_seconds'];assert s['result']['rounds_succeeded']==6294
 r=[json.loads(x) for x in (d/'rounds.jsonl').read_text().splitlines()];m=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()];t=np.array([x['t'] for x in m]);grid=np.arange(0,end,10);ends=np.array([x['finished_seconds'] for x in r]);starts=np.array([x['sent_seconds'] for x in r]);toolends=ends+np.array([x['tool_wait_after_ms']/1000 for x in r]);pc={}
 for x,te in zip(r,toolends):pc[x['session_id']]=max(pc.get(x['session_id'],0),te)
 outstanding=np.array([np.sum((starts<=v)&(ends>v)) for v in grid]);tools=np.array([np.sum((ends<=v)&(toolends>v)) for v in grid]);remaining=np.array([np.sum(np.array(list(pc.values()))>v) for v in grid]);arr={x['session_id']:x['program_arrival_seconds'] for x in r};remaining-=np.array([sum(z>v for z in arr.values()) for v in grid])
 axs[1].plot(grid/60,remaining,color=color,ls=ls,label=label);axs[2].plot(t/60,[x['vllm:num_requests_running'] for x in m],color=color,ls=ls);axs[3].plot(t/60,[x['vllm:num_requests_waiting'] for x in m],color=color,ls=ls);axs[4].plot(grid/60,outstanding,color=color,ls=ls);axs[5].plot(grid/60,tools,color=color,ls=ls)
 bounds=np.r_[np.arange(0,end,60),end];q=np.array([x['vllm:prompt_tokens_total'] for x in m]);h=np.array([x['vllm:prefix_cache_hits_total'] for x in m]);g=np.array([x['vllm:generation_tokens_total'] for x in m]);centers=[];rates=[];inputs=[];outputs=[]
 for left,right in zip(bounds[:-1],bounds[1:]):
  i=min(np.searchsorted(t,left),len(t)-1);j=min(np.searchsorted(t,right),len(t)-1);dt=t[j]-t[i];rate=np.sum((ends>left)&(ends<=right))/(right-left)*60;inp=(q[j]-q[i]-(h[j]-h[i]))/dt if dt else np.nan;out=(g[j]-g[i])/dt if dt else np.nan;centers.append((left+right)/120);rates.append(rate);inputs.append(inp);outputs.append(out);export.append([folder,left,right,rate,inp,out])
 paused=set(); events=[]
 log=d/'thunderagent.log'
 if log.exists():
  for line in log.read_text().splitlines():
   event=re.search(r'(Paused program |Resumed program |Released and removed program: )(\S+)',line)
   if not event:continue
   try:moment=datetime.datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=datetime.timezone.utc).timestamp()-s['started_at_wall']
   except ValueError:continue
   events.append((moment,event.group(1).startswith('Paused'),event.group(2)))
 pos=0;counts=[]
 for moment in grid:
  while pos<len(events) and events[pos][0]<=moment:
   _,pause,program=events[pos]
   if pause:paused.add(program)
   else:paused.discard(program)
   pos+=1
  counts.append(len(paused))
 axs[8].plot(grid/60,counts,color=color,ls=ls)
 mean_output=[]
 for left,right in zip(bounds[:-1],bounds[1:]):
  values=[x['engine_completion_tokens'] for x in r if left<x['finished_seconds']<=right];mean_output.append(np.mean(values) if values else np.nan)
 axs[9].plot(centers,mean_output,color=color,ls=ls)
 axs[0].plot(centers,rates,color=color,ls=ls);axs[6].plot(centers,inputs,color=color,ls=ls);axs[7].plot(centers,outputs,color=color,ls=ls)
 for ax in axs:ax.axvline(end/60,color=color,ls=':',alpha=.4)
labels=['Step completion rate (steps/min; 60-second windows)','Arrived programs not yet completed','Engine running requests','Engine waiting requests','Outstanding client calls (includes router / engine waiting)','Programs in recorded tool waits','Uncached prompt token rate (tokens/s; counter proxy)','Generated output token rate (tokens/s)','Programs marked paused in router logs','Mean output tokens per completed call']
for ax,label in zip(axs,labels):ax.set_ylabel(textwrap.fill(label,32),fontsize=11);ax.set_ylim(bottom=0);ax.set_xlim(left=0);ax.grid(alpha=.25)
for ax in axs[-2:]:ax.set_xlabel('Elapsed time since workload start (minutes)')
fig.suptitle('Why does step completion rate fall? Fixed 200-program workload',y=.995);fig.legend(*axs[1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.965),fontsize=11);fig.tight_layout(rect=(0,0,1,.91),h_pad=2,w_pad=3);out=a.root/'figures';fig.savefig(out/'step_rate_explanation.png',dpi=150);fig.savefig(out/'step_rate_explanation.pdf')
with (out/'step_rate_explanation.csv').open('w') as f:w=csv.writer(f);w.writerow(['configuration','start_seconds','end_seconds','steps_per_minute','uncached_prompt_tokens_per_second','output_tokens_per_second']);w.writerows(export)
for k in (10,20,30):
 v=k*60;j=min(np.searchsorted(t,v),len(t)-1);print(folder,k,'min: running',m[j]['vllm:num_requests_running'],'waiting',m[j]['vllm:num_requests_waiting'])
print(out)
