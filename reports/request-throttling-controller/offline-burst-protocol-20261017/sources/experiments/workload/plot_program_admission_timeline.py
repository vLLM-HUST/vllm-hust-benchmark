"""Plot actual per-program events, without assigning programs to assumed batches."""
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();d=a.root/'thunderagent-program-throttling';r=[json.loads(x) for x in (d/'rounds.jsonl').read_text().splitlines()];groups={}
for x in r:groups.setdefault(x['session_id'],[]).append(x)
order=sorted(groups,key=lambda k:groups[k][0]['program_arrival_seconds']);out=a.root/'figures';out.mkdir(exist_ok=True)
with (out/'program_timeline_events.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['program','round','arrival_seconds','sent_seconds','first_token_seconds','finished_seconds','tool_end_seconds'])
 for k in order:
  for x in sorted(groups[k],key=lambda v:v['round_idx']):w.writerow([k,x['round_idx'],x['program_arrival_seconds'],x['sent_seconds'],x['first_token_seconds'],x['finished_seconds'],x['finished_seconds']+x['tool_wait_after_ms']/1000])
fig,(ax,bx)=plt.subplots(2,1,figsize=(14,10),gridspec_kw={'height_ratios':[3,1]},sharex=True)
for y,k in enumerate(order):
 rows=sorted(groups[k],key=lambda x:x['round_idx']);first=rows[0]['first_token_seconds']/60;last=rows[-1]['finished_seconds']/60
 ax.plot([first,last],[y,y],color='#b8d3e8',lw=.65);ax.scatter([first],[y],color='#D55E00',s=6);ax.scatter([last],[y],color='black',s=6)
ax.invert_yaxis();ax.set_ylabel('Program arrival order (all 600 programs)');ax.set_title('Actual first-token and final-completion times; no assumed batches')
legend=[Line2D([0],[0],marker='o',color='#D55E00',lw=0,label='First token of first call'),Line2D([0],[0],marker='o',color='black',lw=0,label='Final call completed'),Line2D([0],[0],color='#b8d3e8',label='Span between those events (includes waits)')];ax.legend(handles=legend,loc='upper right',fontsize=9)
ends=np.array([x['finished_seconds'] for x in r]);firsts=np.array([x['finished_seconds'] for x in r if x['round_idx']==0]);lasts=np.array([x['finished_seconds'] for x in r if x['round_idx']==19]);b=np.arange(0,ends.max()+60,60)
for values,name,color in [(ends,'All completed steps/min','#0072B2'),(firsts,'First calls completed/min','#D55E00'),(lasts,'Final calls completed/min','#009E73')]:bx.step(b[1:]/60,np.histogram(values,bins=b)[0],where='pre',label=name,color=color)
bx.legend(fontsize=9);bx.set_ylabel('Completed calls/min');bx.set_xlabel('Elapsed time (minutes)');bx.set_ylim(bottom=0)
for axis in (ax,bx):axis.set_xlim(left=0);axis.axvline(599/60,color='gray',ls='--');axis.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'program_admission_overview.png',dpi=160);fig.savefig(out/'program_admission_overview.pdf');plt.close(fig)
selected=order[:24]+order[64:72]
fig,ax=plt.subplots(figsize=(15,11))
for y,k in enumerate(selected):
 for x in groups[k]:
  start=x['sent_seconds']/60;first=x['first_token_seconds']/60;end=x['finished_seconds']/60;tool=x['tool_wait_after_ms']/60000
  ax.plot([start,first],[y,y],color='#bbb',lw=2)
  ax.plot([first,end],[y,y],color='#D55E00' if x['round_idx']==0 else '#0072B2',lw=3)
  if tool:ax.plot([end,end+tool],[y,y],color='#009E73',lw=3)
  ax.scatter(end,y,color='black',s=5)
 ax.scatter(max(x['finished_seconds'] for x in groups[k])/60,y,marker='x',color='black',s=45)
ax.set_yticks(range(len(selected)),[k.rsplit('-',1)[-1] for k in selected]);ax.invert_yaxis();ax.set_xlim(0,24);ax.set_xlabel('Elapsed time (minutes)');ax.set_ylabel('Program identifier suffix (first 24 arrivals + arrivals 64–71)');ax.set_title('Measured call timelines: pause does not imply synchronized starts or finishes')
ax.legend(handles=[Line2D([0],[0],color='#bbb',lw=3,label='Request sent → first token (router + engine + network wait)'),Line2D([0],[0],color='#D55E00',lw=3,label='First call: first token → completion'),Line2D([0],[0],color='#0072B2',lw=3,label='Later calls: first token → completion'),Line2D([0],[0],color='#009E73',lw=3,label='Recorded tool wait (3 seconds)'),Line2D([0],[0],marker='x',color='black',lw=0,label='Program final completion')],loc='upper center',bbox_to_anchor=(.5,-.065),ncol=2,fontsize=9);ax.grid(axis='x',alpha=.25);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'program_call_timeline_detail.png',dpi=160);fig.savefig(out/'program_call_timeline_detail.pdf');plt.close(fig)
print(out)
