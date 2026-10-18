#!/usr/bin/env python3
"""Validate a four-call smoke before launching the fixed-context stream."""
import json
import os
import subprocess
from pathlib import Path

root = Path(os.environ['REPO_ROOT'])
smoke = 'four-card-constant-context-smoke-20261007'
experiment = 'four-card-constant-context-stream-20261007'
environment = dict(os.environ, FIXED_PROGRAM_CONTENT='1', DRAIN_AFTER_ARRIVALS='1', CONCURRENCY='0')
def run(name):
    subprocess.run(['bash', str(root/'experiments/workload/program_configurations.sh'), name, name, 'thunderagent-with-throttling'], env=environment, check=True)
run(smoke)
directory = root/'data/results/workload'/smoke/'thunderagent-program-throttling'
summary = json.loads((directory/'summary.json').read_text())
result = summary['result']
assert result['sessions_completed'] == 2 and result['rounds_succeeded'] == 4 and result['rounds_failed'] == 0, result
rounds = [json.loads(line) for line in (directory/'rounds.jsonl').read_text().splitlines()]
assert len(rounds) == 4 and len({r['engine_prompt_tokens'] for r in rounds}) == 1
assert all(r['engine_completion_tokens'] == 128 for r in rounds)
for group in summary.values():
    if isinstance(group, dict) and 'mismatches' in group:
        assert all(value['count'] == 0 for value in group['mismatches'].values()), group
print('[validation] four successful calls; constant measured prompt length; exactly 128 output tokens. Starting full stream.', flush=True)
run(experiment)
