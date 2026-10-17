"""Four aligned panels from the existing canonical occupancy and router CSV files."""
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
for name in ['plot_program_serving_occupancy.py','plot_router_active_programs.py']:
 subprocess.run([sys.executable,str(Path(__file__).with_name(name)),str(a.root)],check=True)
out=a.root/'figures'
d=np.genfromtxt(out/'program_serving_occupancy.csv',delimiter=',',names=True)
r=np.genfromtxt(out/'router_active_programs.csv',delimiter=',',names=True)
x=(d['start_seconds']+d['end_seconds'])/120;t=r['elapsed_seconds']/60
end=json.loads((a.root/'thunderagent-program-throttling/summary.json').read_text())['result']['makespan_seconds']/60
rounds=[json.loads(line) for line in (a.root/'thunderagent-program-throttling/rounds.jsonl').read_text().splitlines()];arrival_end=max(row['program_arrival_seconds'] for row in rounds)
fig,axes=plt.subplots(4,1,figsize=(13,14),sharex=True,layout='constrained')
axes[0].plot(x,d['engine_running_programs_average'],color='#0072B2',label='Engine running programs: 60-second mean')
axes[0].fill_between(x,d['engine_running_programs_minimum'],d['engine_running_programs_maximum'],color='#0072B2',alpha=.18,label='Sampled minimum–maximum within each minute')
axes[0].plot(x,d['engine_waiting_programs_average'],color='#D55E00',label='Engine waiting programs: 60-second mean')
axes[1].plot(x,d['client_outstanding_programs_average'],color='#0072B2',label='Programs with an unfinished client call (includes router waiting)')
axes[1].plot(x,d['tool_waiting_programs_average'],color='#009E73',label='Programs in recorded tool waits')
for key,label,color in [('router_active_programs','Router active programs (reasoning + acting)','#0072B2'),('router_reasoning_programs','Router reasoning programs','#009E73'),('router_acting_programs','Router acting programs','#CC79A7')]:
 axes[2].step(t,r[key],where='post',label=label,color=color,linewidth=1.2)
axes[3].step(t,r['router_queued_or_paused_programs'],where='post',color='#D55E00',label='Router queued or paused programs (excluded from active)')
titles=['(a) Engine service and waiting — time-weighted minute averages','(b) Client calls and tool waits — time-weighted minute averages','(c) Router active programs — snapshots after scheduler checks','(d) Router queued or paused programs — snapshots after scheduler checks']
for ax,title in zip(axes,titles):
 ax.set_title(title,loc='left',fontsize=11);ax.set_xlim(0,end);ax.set_ylim(bottom=0);ax.set_ylabel('Concurrent programs');ax.axvline(arrival_end/60,ls='--',color='gray',label='End of arrivals');ax.grid(alpha=.25)
 ax.legend(loc='lower left',bbox_to_anchor=(0,1.13),fontsize=8,ncol=2,frameon=False)
axes[-1].set_xlabel('Elapsed time (minutes)')
fig.savefig(out/'program_occupancy_overview.png',dpi=160);fig.savefig(out/'program_occupancy_overview.pdf');print(out/'program_occupancy_overview.png')
