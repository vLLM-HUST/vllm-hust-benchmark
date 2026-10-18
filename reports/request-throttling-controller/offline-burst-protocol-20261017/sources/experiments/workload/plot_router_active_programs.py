"""Plot sampled router membership; queued/paused programs are excluded from active."""
import argparse,csv,json,re
from pathlib import Path
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
s=json.loads((a.root/'thunderagent-program-throttling/summary.json').read_text());start=s['started_at_wall'];end=s['result']['makespan_seconds'];rows=[]
rounds=[json.loads(line) for line in (a.root/'thunderagent-program-throttling/rounds.jsonl').read_text().splitlines()];arrival_end=max(row['program_arrival_seconds'] for row in rounds)
pattern=re.compile(r'scheduler after t=([\d.]+).*?programs=(\d+) reasoning=(\d+) acting=(\d+) \| engine: running=([\d.]+) waiting=([\d.]+).*? \| waiting=(\d+) known_programs=(\d+)')
for n,line in enumerate((a.root/'thunderagent.log').read_text().splitlines(),1):
 m=pattern.search(line)
 if not m:continue
 t=float(m[1])-start
 if not 0<=t<=end:continue
 active,reasoning,acting=int(m[2]),int(m[3]),int(m[4]);assert active==reasoning+acting
 rows.append(dict(elapsed_seconds=t,router_active_programs=active,router_reasoning_programs=reasoning,router_acting_programs=acting,engine_running_requests=float(m[5]),engine_waiting_requests=float(m[6]),router_queued_or_paused_programs=int(m[7]),router_known_programs=int(m[8]),source_log_line=n))
assert rows
out=a.root/'figures';out.mkdir(exist_ok=True)
with (out/'router_active_programs.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
with (out/'router_active_programs.csv').open() as f:rows=list(csv.DictReader(f))
x=[float(r['elapsed_seconds'])/60 for r in rows]
fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
for key,label,color in [('router_active_programs','Router active programs (reasoning + acting)','#0072B2'),('router_reasoning_programs','Router reasoning programs','#009E73'),('router_acting_programs','Router acting programs','#CC79A7')]:axes[0].step(x,[float(r[key]) for r in rows],where='post',label=label,color=color)
axes[1].step(x,[float(r['router_queued_or_paused_programs']) for r in rows],where='post',label='Router queued or paused programs (excluded from active)',color='#D55E00')
for ax in axes:ax.set_xlim(0,end/60);ax.set_ylim(bottom=0);ax.set_ylabel('Programs');ax.grid(alpha=.25);ax.axvline(arrival_end/60,ls='--',color='gray',label='End of arrivals');ax.legend(loc='upper right',fontsize=9)
axes[1].set_xlabel('Elapsed time (minutes)');fig.suptitle('ThunderAgent router active programs: snapshots after scheduler checks');fig.tight_layout();fig.savefig(out/'router_active_programs.png',dpi=160);fig.savefig(out/'router_active_programs.pdf');print('samples',len(rows),'active range',min(int(r['router_active_programs']) for r in rows),max(int(r['router_active_programs']) for r in rows))
