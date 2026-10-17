#!/usr/bin/env python3
"""Generate identical-size programs with independent, reusable fixed prefixes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from build_program_workload import COLUMNS
from generate_workload import git_state

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--programs', type=int, default=600)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--prompt-tokens', type=int, default=16384)
    parser.add_argument('--new-input-tokens', type=int, default=128)
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--tool-seconds', type=float, default=3)
    parser.add_argument('--program-rate', type=float, default=1)
    a = parser.parse_args()
    if min(a.programs, a.rounds, a.new_input_tokens, a.output_tokens) <= 0 or a.prompt_tokens < a.new_input_tokens or a.program_rate <= 0 or a.tool_seconds < 0:
        parser.error('invalid workload dimensions')
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output / 'sessions.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator='\n')
        writer.writeheader()
        for program in range(a.programs):
            for step in range(a.rounds):
                writer.writerow(dict(session_id=f'constant-context-program-{program:05d}', arrival_time=round(program / a.program_rate * 1000), round_idx=step, prefix_len=a.prompt_tokens-a.new_input_tokens, input_len=a.new_input_tokens, output_len=a.output_tokens, tool_wait_after_ms=round(a.tool_seconds*1000) if step+1<a.rounds else 0, priority_class='ordinary', latency_objective_seconds=0))
    manifest = dict(generator='experiments/workload/build_constant_context_workload.py', **git_state(), knobs={**{k:v for k,v in vars(a).items() if k!='output'}, 'arrival_duration_seconds':(a.programs-1)/a.program_rate}, round_count=a.programs*a.rounds, sessions_sha256=hashlib.sha256((a.output/'sessions.csv').read_bytes()).hexdigest(), semantics='Each program has its own deterministic prefix. Every round reuses the same prefix and replaces its suffix; no accumulated conversation growth. Chat-template overhead is measured by replay. Synthetic input on real hardware; no agent correctness claim.')
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
