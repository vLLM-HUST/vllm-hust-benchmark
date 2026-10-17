"""Time-weighted per-minute engine/request occupancy, not completion counts."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();d=a.root/'thunderagent-program-throttling';s=json.loads((d/'summary.json').read_text());r=[json.loads(x) for x in (d/'rounds.jsonl').read_text().splitlines()];m=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()];end=s['result']['makespan_seconds'];t=np.array([x['t'] for x in m]);starts=np.array([x['sent_seconds'] for x in r]);finishes=np.array([x['finished_seconds'] for x in r]);tools=finishes+np.array([x['tool_wait_after_ms']/1000 for x in r]);bounds=list(np.arange(0,end,60))+[end];rows=[]
def metric_average(key,left,right):
 values=np.array([x[key] for x in m]);weights=np.maximum(0,np.minimum(np.r_[t[1:],end],right)-np.maximum(t,left));covered=weights.sum();assert covered>0
 selected=values[weights>0];return float(np.dot(weights,values)/covered),float(selected.min()),float(selected.max()),float(covered/(right-left))
for left,right in zip(bounds[:-1],bounds[1:]):
 active,lo,hi,coverage=metric_average('vllm:num_requests_running',left,right);waiting,_,_,_=metric_average('vllm:num_requests_waiting',left,right)
 overlap=lambda begin,endpoints:float(np.maximum(0,np.minimum(endpoints,right)-np.maximum(begin,left)).sum()/(right-left))
 rows.append(dict(start_seconds=left,end_seconds=right,engine_running_programs_average=active,engine_running_programs_minimum=lo,engine_running_programs_maximum=hi,engine_waiting_programs_average=waiting,client_outstanding_programs_average=overlap(starts,finishes),tool_waiting_programs_average=overlap(finishes,tools),metrics_coverage_fraction=coverage))
out=a.root/'figures';out.mkdir(exist_ok=True)
with (out/'program_serving_occupancy.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
data=np.genfromtxt(out/'program_serving_occupancy.csv',delimiter=',',names=True);x=(data['start_seconds']+data['end_seconds'])/120
fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
axes[0].plot(x,data['engine_running_programs_average'],label='Engine running programs: time-weighted mean',color='#0072B2');axes[0].fill_between(x,data['engine_running_programs_minimum'],data['engine_running_programs_maximum'],alpha=.18,color='#0072B2',label='Observed minimum–maximum within each minute');axes[0].plot(x,data['engine_waiting_programs_average'],label='Engine waiting programs: time-weighted mean',color='#D55E00')
axes[1].plot(x,data['client_outstanding_programs_average'],label='Programs with an unfinished client call (includes router waiting)',color='#0072B2');axes[1].plot(x,data['tool_waiting_programs_average'],label='Programs in recorded tool waits',color='#009E73')
for ax in axes:ax.set_ylim(bottom=0);ax.set_xlim(left=0);ax.set_ylabel('Concurrent programs');ax.axvline(max(x['program_arrival_seconds'] for x in r)/60,color='gray',ls='--',label='End of arrivals');ax.legend(fontsize=8, loc='upper right', framealpha=0.9);ax.grid(alpha=.25)
axes[1].set_xlabel('Elapsed time (minutes)');fig.suptitle('Programs being served or waiting: 60-second occupancy windows');fig.tight_layout();fig.savefig(out/'program_serving_occupancy.png',dpi=160);fig.savefig(out/'program_serving_occupancy.pdf');print(out)
