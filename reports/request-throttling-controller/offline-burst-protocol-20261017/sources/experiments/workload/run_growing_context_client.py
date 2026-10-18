#!/usr/bin/env python3
"""One submitted Ray job owns engine, router, smoke, replay and final validation."""
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

root = Path(os.environ['REPO_ROOT'])
name = os.environ['EXPERIMENT_NAME']
growing = os.environ.get('GROWING_CONTEXT_PILOT') == '1'
programs = int(os.environ.get('GROWING_CONTEXT_PROGRAMS', '100'))
workload = os.environ.get('GROWING_CONTEXT_WORKLOAD', 'four-card-growing-context-pilot-20261007')
out = root/'data/results/workload'/name
out.mkdir(parents=True, exist_ok=True)
processes = []
handles = []

def start(arguments, filename):
    handle = (out/filename).open('w')
    handles.append(handle)
    process = subprocess.Popen(arguments, stdout=handle, stderr=subprocess.STDOUT)
    processes.append(process)
    return process

def wait_health(url, process, deadline):
    start_time = time.monotonic()
    while time.monotonic()-start_time < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'process exited {process.returncode}: {url}')
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f'startup deadline: {url}')

def replay(workload, target, expected_programs, expected_steps):
    arguments = ['python', str(root/'experiments/workload/replay_sessions.py'), '--workload', str(root/'data/workloads'/workload), '--base-url', 'http://127.0.0.1:8000/v1', '--model', 'qwen2.5-14b', '--api', 'chat', '--tokenizer', '/models/Qwen2.5-14B-Instruct', '--engine-url', 'http://127.0.0.1:8100/v1', '--timeout', '0', '--drain-timeout-seconds', '21600', '--fixed-program-content', '--evidence-type', 'live', '--out', str(target)]
    print('[run] '+ ' '.join(arguments), flush=True)
    subprocess.run(arguments, check=True)
    summary = json.loads((target/'summary.json').read_text())
    result = summary['result']
    assert result['sessions_completed'] == expected_programs and result['rounds_succeeded'] == expected_steps and result['rounds_failed'] == 0, result
    assert all(value['count'] == 0 for value in result['mismatches'].values())
    rows = [json.loads(line) for line in (target/'rounds.jsonl').read_text().splitlines()]
    if growing:
        groups = {}
        for row in rows:
            step = row['round_idx']
            assert row['prefix_len'] + row['input_len'] == 4096 + step*640
            assert row['engine_prompt_tokens'] == 4096 + step*640 + summary['chat_template_tokens']
            if step > 0:
                assert row['previous_history_token_count'] == row['prefix_len']
                assert row['previous_history_sha256'] == row['inherited_history_sha256']
            groups.setdefault(row['session_id'], []).append(row)
        (target/'history-validation.json').write_text(json.dumps({'valid':True,'programs':len(groups),'rounds':len(rows),'planned_input_lengths':[4096+i*640 for i in range(max(x['round_idx'] for x in rows)+1)]},indent=2))
    else:
        assert len({row['engine_prompt_tokens'] for row in rows}) == 1
    assert all(row['engine_completion_tokens'] == 128 for row in rows)
    print('[validated] '+target.name, flush=True)

replay('four-card-growing-context-smoke-20261007' if growing else 'four-card-constant-context-smoke-20261007', out/'smoke', 2, 6 if growing else 4)
with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8100/reset_prefix_cache',method='POST'),timeout=30) as response:
    assert response.status==200
replay(workload if growing else 'four-card-constant-context-stream-20261007',out/'thunderagent-program-throttling',programs if growing else 600,programs*20 if growing else 12000)
(out/'validation.json').write_text(json.dumps({'valid':True,'programs':programs if growing else 600,'steps':programs*20 if growing else 12000,'growing_context':growing}))
