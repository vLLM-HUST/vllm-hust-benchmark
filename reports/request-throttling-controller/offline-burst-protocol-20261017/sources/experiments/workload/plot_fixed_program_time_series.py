"""Validated paired experiment time series from raw rounds and engine counters."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
fig,axes=plt.subplots(3,2,figsize=(14,12),sharex=True);axes=axes.ravel()
labels=['Completed model calls','Completed programs','Model calls per minute (60-second windows)','Prefix cache hit rate (%, 60-second windows)','KV cache usage (%)','Engine waiting requests']
export=[]
for folder,name,color,style in [('vllm-direct-service','Direct vLLM','#D55E00','--'),('thunderagent-program-throttling','ThunderAgent program throttling (fresh backend connections)','#0072B2','-')]:
 d=a.root/folder;r=json.loads((d/'summary.json').read_text())['result'];assert r['sessions_completed']==200 and r['rounds_succeeded']==6294 and r['rounds_failed']==0
 rounds=[json.loads(x) for x in (d/'rounds.jsonl').read_text().splitlines()];m=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()];end=r['makespan_seconds'];finish=np.sort([x['finished_seconds'] for x in rounds]);programs={}
 for x in rounds:programs[x['session_id']]=max(programs.get(x['session_id'],0),x['finished_seconds']+x['tool_wait_after_ms']/1000)
 assert abs(max(programs.values())-end)<2
 axes[0].step(np.r_[0,finish,end]/60,np.r_[0,np.arange(1,len(finish)+1),len(finish)],where='post',label=name,color=color,ls=style)
 pf=np.sort(list(programs.values()));axes[1].step(np.r_[0,pf,end]/60,np.r_[0,np.arange(1,len(pf)+1),len(pf)],where='post',color=color,ls=style)
 t=np.array([x['t'] for x in m]);q=np.array([x['vllm:prefix_cache_queries_total'] for x in m]);h=np.array([x['vllm:prefix_cache_hits_total'] for x in m]);bounds=np.r_[np.arange(0,end,60),end]
 for start,stop in zip(bounds[:-1],bounds[1:]):
  count=np.sum((finish>start)&(finish<=stop));rate=count/(stop-start)*60;i=np.searchsorted(t,start);j=min(np.searchsorted(t,stop),len(t)-1);dq=q[j]-q[i];dh=h[j]-h[i];hit=dh/dq*100 if dq>0 else np.nan;mid=(start+stop)/120
  axes[2].plot([start/60,stop/60],[rate,rate],color=color,ls=style);axes[3].plot(mid,hit,'.',color=color)
  export.append([folder,start,stop,int(count),rate,hit])
 axes[4].plot(t/60,[x['vllm:kv_cache_usage_perc']*100 for x in m],color=color,ls=style,lw=1)
 axes[5].plot(t/60,[x['vllm:num_requests_waiting'] for x in m],color=color,ls=style,lw=1)
 for ax in axes:ax.axvline(end/60,color=color,ls=':',alpha=.5)
for ax,label in zip(axes,labels):ax.set_ylabel(label);ax.set_ylim(bottom=0);ax.set_xlim(left=0);ax.grid(alpha=.25)
axes[3].set_ylim(0,100);axes[4].set_ylim(0,100)
for ax in axes[-2:]:ax.set_xlabel('Elapsed time since workload start (minutes)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.965),ncol=1,fontsize=11)
fig.suptitle('200 programs / 6,294 model calls: completion and cache behavior',y=.995)
fig.tight_layout(rect=(0,0,1,.92));out=a.root/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'fixed_program_time_series.png',dpi=160);fig.savefig(out/'fixed_program_time_series.pdf')
with (out/'fixed_program_time_series.csv').open('w') as f:w=csv.writer(f);w.writerow(['configuration','window_start_seconds','window_end_seconds','completed_calls','calls_per_minute','prefix_hit_percent']);w.writerows(export)
print(out)
