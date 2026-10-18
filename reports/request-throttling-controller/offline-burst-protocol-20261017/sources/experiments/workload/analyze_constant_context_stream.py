#!/usr/bin/env python3
"""Validated raw rounds -> one window table -> figures and report draft."""
import argparse
import csv
import datetime
import html
import json
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
d=a.root/'thunderagent-program-throttling'
s=json.loads((d/'summary.json').read_text());result=s['result']
assert result['sessions_completed']==600 and result['rounds_succeeded']==12000 and result['rounds_failed']==0
assert all(x['count']==0 for x in result['mismatches'].values())
r=[json.loads(line) for line in (d/'rounds.jsonl').read_text().splitlines()]
assert all(x['status']=='ok' for x in r) and len({x['engine_prompt_tokens'] for x in r})==1
m=[json.loads(line) for line in (d/'metrics.jsonl').read_text().splitlines()]
t=np.array([x['t'] for x in m]);finish=np.array([x['finished_seconds'] for x in r]);done={};arrivals={}
for x in r:
    done[x['session_id']]=max(done.get(x['session_id'],0),x['finished_seconds']+x['tool_wait_after_ms']/1000)
    arrivals[x['session_id']]=x['program_arrival_seconds']
end=result['makespan_seconds'];bounds=list(np.arange(0,end,60))+[end]
events=[]
for line in (a.root/'thunderagent.log').read_text().splitlines():
    match=re.search(r'(Paused program |Queued new program |Resumed program |Released and removed program: )(\S+)',line)
    if match:
        stamp=datetime.datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=datetime.timezone.utc).timestamp()-s['started_at_wall']
        if stamp >= 0:
            events.append((stamp,match.group(1).startswith(('Paused','Queued')),match.group(2)))
events.sort();paused=set();position=0;rows=[]
for left,right in zip(bounds[:-1],bounds[1:]):
    while position<len(events) and events[position][0]<=right:
        _,pause,program=events[position];position+=1
        if pause:paused.add(program)
        else:paused.discard(program)
    i=min(np.searchsorted(t,left),len(m)-1);j=min(np.searchsorted(t,right),len(m)-1)
    q=m[j]['vllm:prefix_cache_queries_total']-m[i]['vllm:prefix_cache_queries_total'];h=m[j]['vllm:prefix_cache_hits_total']-m[i]['vllm:prefix_cache_hits_total']
    arrived=sum(x<=right for x in arrivals.values());completed=sum(x<=right for x in done.values())
    rows.append(dict(start_seconds=left,end_seconds=right,steps_per_minute=float(sum((finish>left)&(finish<=right))/(right-left)*60),programs_arrived=arrived,programs_completed=completed,programs_unfinished=arrived-completed,paused_programs=len(paused),kv_usage_percent=m[j]['vllm:kv_cache_usage_perc']*100,prefix_hit_percent=h/q*100 if q>0 else float('nan'),running_requests=m[j]['vllm:num_requests_running'],waiting_requests=m[j]['vllm:num_requests_waiting']))
out=a.root/'figures';out.mkdir(exist_ok=True)
with (out/'constant_context_windows.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
# All plot inputs are read back from the single canonical table.
table=np.genfromtxt(out/'constant_context_windows.csv',delimiter=',',names=True)
fig,axes=plt.subplots(4,2,figsize=(14,13),sharex=True)
fields=['steps_per_minute','programs_unfinished','kv_usage_percent','prefix_hit_percent','running_requests','waiting_requests','paused_programs','programs_completed']
labels=['Completed steps/min (60-second windows)','Arrived unfinished programs','KV cache usage (%)','Prefix cache hit rate (%)','Engine running requests','Engine waiting requests','Programs queued or paused at router','Completed programs']
for ax,key,label in zip(axes.ravel(),fields,labels):
    ax.plot(table['end_seconds']/60,table[key],label='ThunderAgent program throttling');ax.set_ylabel(label);ax.set_ylim(bottom=0);ax.set_xlim(left=0);ax.axvline(599/60,color='black',ls='--',label='End of arrivals');ax.grid(alpha=.25)
    if key in ['kv_usage_percent','prefix_hit_percent']:ax.set_ylim(0,100)
for ax in axes[-1]:ax.set_xlabel('Elapsed time (minutes)')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2)
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/'constant_context_stream.png',dpi=160);fig.savefig(out/'constant_context_stream.pdf');plt.close(fig)
eligible=[x for x in rows if x['end_seconds']<=599 and x['end_seconds']-x['start_seconds']==60 and x['paused_programs']>0]
consecutive=len(eligible)>=5 and all(eligible[i]['start_seconds']==eligible[i-1]['end_seconds'] for i in range(1,len(eligible)))
backlog=consecutive and all(eligible[i]['programs_unfinished']>eligible[i-1]['programs_unfinished'] for i in range(1,len(eligible)))
rates=np.array([x['steps_per_minute'] for x in eligible]);cv=float(np.std(rates)/np.mean(rates)) if len(rates) and rates.mean()>0 else None
change=float(np.polyfit(np.arange(len(rates)),rates,1)[0]*(len(rates)-1)/rates.mean()) if len(rates)>=2 and rates.mean()>0 else None
assessment='未满足连续饱和窗口门禁，不能判断平台。'
if backlog:assessment='饱和窗口内速率符合预设稳定标准。' if cv<=.1 and abs(change)<=.1 else '饱和窗口内速率不符合预设稳定标准，需结合状态曲线解释。'
metrics=dict(valid=True,eligible_windows=len(eligible),backlog_gate=backlog,coefficient_of_variation=cv,fitted_relative_change=change,assessment=assessment)
(out/'analysis.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
image=out/'constant_context_stream.png'
content=f'<h3>固定上下文持续到达实验：{html.escape(a.root.name)}</h3><p>600个program，每秒到达1个，每个20轮；固定16384个合成输入token加实测模板，输出128，工具等待3秒。独立重建四卡Ray，单作业管理全部进程；取消客户端超时和router强制放行，保留整批安全期限。</p><p>验收：600/600 program、12000/12000成功步、零错误、零token偏差。{assessment}饱和窗口数{len(eligible)}，变异系数{cv}，拟合相对变化{change}。停止到达后的排空与到达期分开，不能把尾部并行度不足算容量下降。</p><img path="@{html.escape(str(image))}" caption="成功响应完成时间计步，60秒窗口；虚线为停止到达。所有面板来自同一窗口CSV。"/><p>边界：合成固定输入的真实硬件推理，第一轮需计算长前缀；输入长度固定不代表驻留KV固定；单种子没有置信区间，不宣称baseline收益或真实agent正确率。原始证据：{html.escape(str(d))}。PNG、PDF和唯一窗口CSV：{html.escape(str(out))}。</p>'
(out/'report-draft.xml').write_text(content)
print(json.dumps(metrics,ensure_ascii=False))
