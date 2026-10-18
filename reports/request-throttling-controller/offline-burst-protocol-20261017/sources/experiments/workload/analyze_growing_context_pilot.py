"""Validate pilot history semantics and derive growth, pressure and completion curves."""
import argparse,csv,json,html,re,datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();d=a.root/'thunderagent-program-throttling';s=json.loads((d/'summary.json').read_text());result=s['result'];r=[json.loads(x) for x in (d/'rounds.jsonl').read_text().splitlines()];m=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()]
configuration=json.loads((a.root/'effective-experiment-configuration.json').read_text());programs=configuration['programs'];steps=configuration['rounds']
assert result['sessions_completed']==programs and result['rounds_succeeded']==steps and result['rounds_failed']==0
assert all(x['count']==0 for x in result['mismatches'].values())
for row in r:
 assert row['status']=='ok' and row['engine_prompt_tokens']==4096+row['round_idx']*640+s['chat_template_tokens']
 if row['round_idx']>0:assert row['previous_history_token_count']==row['prefix_len'] and row['previous_history_sha256']==row['inherited_history_sha256']
logs=(a.root/'thunderagent.log').read_text().splitlines();events=[]
for line in logs:
 if 'Paused program ' not in line and 'Resumed program ' not in line:continue
 try:stamp=datetime.datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=datetime.timezone.utc).timestamp()
 except ValueError:continue
 elapsed=stamp-s['started_at_wall']
 if elapsed>=0:events.append((elapsed,'pause' if 'Paused program ' in line else 'resume'))
end=result['makespan_seconds'];arrival_end=max(x['program_arrival_seconds'] for x in r);bounds=list(np.arange(0,end,60))+[end];rows=[]
for left,right in zip(bounds[:-1],bounds[1:]):
 finished=[x for x in r if left<x['finished_seconds']<=right];sent=[x for x in r if left<=x['sent_seconds']<right];sample=[x for x in m if left<=x['t']<right]
 rows.append(dict(start_seconds=left,end_seconds=right,successful_steps_per_minute=len(finished)/(right-left)*60,mean_input_tokens_of_calls_sent=np.mean([x['engine_prompt_tokens'] for x in sent]) if sent else float('nan'),maximum_input_tokens_of_calls_sent=max([x['engine_prompt_tokens'] for x in sent],default=float('nan')),sample_mean_kv_percent=np.mean([x['vllm:kv_cache_usage_perc']*100 for x in sample]) if sample else float('nan')))
previous_queries=previous_hits=None
for row in rows:
 left,right=row['start_seconds'],row['end_seconds'];sample=[v for v in m if left<=v['t']<right]
 if sample:
  latest=sample[-1];queries=latest['vllm:prefix_cache_queries_total'];hits=latest['vllm:prefix_cache_hits_total']
  dq=queries-previous_queries if previous_queries is not None else float('nan');dh=hits-previous_hits if previous_hits is not None else float('nan');previous_queries,previous_hits=queries,hits
 else:dq=dh=float('nan')
 row.update(complete_60_second_window=int(right-left>=59.999),prefix_query_tokens_increment=dq,prefix_hit_tokens_increment=dh,prefix_increment_hit_percent=dh/dq*100 if dq>0 else float('nan'),pause_events=sum(left<=stamp<right and kind=='pause' for stamp,kind in events),resume_events=sum(left<=stamp<right and kind=='resume' for stamp,kind in events))
out=a.root/'figures';out.mkdir(exist_ok=True)
with (out/'growing_context_windows.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
table=np.genfromtxt(out/'growing_context_windows.csv',delimiter=',',names=True);x=(table['start_seconds']+table['end_seconds'])/120
fig,axes=plt.subplots(5,1,figsize=(12,16),sharex=True,layout='constrained')
for ax,key,label in zip(axes,['mean_input_tokens_of_calls_sent','sample_mean_kv_percent','successful_steps_per_minute'],['Mean input length of calls sent (tokens)','Sample mean KV cache usage (%)','Successfully completed steps/min']):
 ax.plot(x,table[key],color='#0072B2',label=label);ax.set_ylabel(label);ax.set_xlim(0,end/60);ax.set_ylim(bottom=0);ax.axvline(arrival_end/60,color='gray',ls='--',label='End of arrivals');ax.legend(fontsize=8);ax.grid(alpha=.25)
axes[3].plot(x,table['prefix_increment_hit_percent'],label='Prefix cache hit ratio from counter increments (%)');axes[3].set_ylabel('Prefix hit ratio (%)');axes[3].set_ylim(0,100)
axes[4].step(x,table['pause_events'],where='mid',label='Router pause events per window');axes[4].step(x,table['resume_events'],where='mid',label='Router resume events per window');axes[4].set_ylabel('Events per window');axes[4].set_ylim(bottom=0)
for ax in axes[3:]:ax.set_xlim(0,end/60);ax.axvline(arrival_end/60,color='gray',ls='--',label='End of arrivals');ax.grid(alpha=.25);ax.legend(fontsize=8)
for ax in axes:ax.axvspan(table['start_seconds'][-1]/60,end/60,color='gray',alpha=.15,label='Final partial window')
axes[2].scatter([x[-1]],[table['successful_steps_per_minute'][-1]],marker='x',color='black',label='Final partial window (normalized)');axes[2].legend(fontsize=8)
axes[0].plot(x,table['maximum_input_tokens_of_calls_sent'],color='#D55E00',label='Maximum input length of calls sent');axes[0].legend(fontsize=8);axes[-1].set_xlabel('Elapsed time (minutes)');fig.savefig(out/'growing_context_pressure.png',dpi=160);fig.savefig(out/'growing_context_pressure.pdf')
assessment={'execution_valid':True,'history_growth_valid':True,'purpose':'Growing-context stream; periodicity and capacity require time-series evidence, not assumed from run size','arrival_seconds':arrival_end,'makespan_seconds':end,'programs':programs,'successful_steps':steps,'failures':0,'mean_steps_per_minute':steps/end*60,'pause_events':sum(kind=='pause' for _,kind in events),'resume_events':sum(kind=='resume' for _,kind in events),'future_main_run_sizing':'Determine sufficient arrival windows using observed service and pressure; do not infer monotone active decline from aggregate population.'}
(out/'analysis.json').write_text(json.dumps(assessment,indent=2))
(out/'report-draft.xml').write_text('<h3>逐轮上下文增长实验</h3><p>初始4096合成输入，每轮完整追加128预定响应历史和512工具观察；20轮最终16256加模板。真实模型输出不进入后续历史。'+html.escape(json.dumps(assessment,ensure_ascii=False))+'</p><p>历史继承hash逐轮一致、零输入输出token偏差；仅用于校准，不构成持续到达稳态容量结论，不报告baseline加速比。</p><img path="@./'+str(out/'growing_context_pressure.png')+'" caption="输入长度、KV使用及成功完成steps/min；arrival结束后为drain。"/><img path="@./'+str(out/'program_occupancy_overview.png')+'" caption="四排状态对照；分钟均值与router快照明确分开。"/>')
print(json.dumps(assessment))
