"""Pure generator: keep complete previous synthetic history and append tool input."""
import argparse,csv,hashlib,json
from pathlib import Path
from build_program_workload import COLUMNS
from generate_workload import git_state
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--programs',type=int,required=True);p.add_argument('--rounds',type=int,default=20);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
rows=[]
for program in range(a.programs):
 for step in range(a.rounds):
  rows.append(dict(session_id=f'growing-context-program-{program:05d}',arrival_time=program*1000,round_idx=step,prefix_len=0 if step==0 else 4096+(step-1)*640+128,input_len=4096 if step==0 else 512,output_len=128,tool_wait_after_ms=3000 if step+1<a.rounds else 0,priority_class='ordinary',latency_objective_seconds=0))
with (a.output/'sessions.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerows(rows)
manifest=dict(**git_state(),knobs={'arrival_duration_seconds':a.programs-1,'program_rate':1,'programs':a.programs,'rounds':a.rounds},programs=a.programs,rounds_per_program=a.rounds,rate=1,initial_input_tokens=4096,synthetic_response_history_tokens=128,tool_observation_tokens=512,round_count=len(rows),sessions_sha256=hashlib.sha256((a.output/'sessions.csv').read_bytes()).hexdigest(),semantics='Complete synthetic history retained; 128 predetermined replacement response tokens plus 512 tool observation tokens appended each round. Real model output excluded for controlled inputs. No agent correctness claim.')
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'programs':a.programs,'rounds':len(rows),'input_lengths':[r['prefix_len']+r['input_len'] for r in rows[:a.rounds]]}))
