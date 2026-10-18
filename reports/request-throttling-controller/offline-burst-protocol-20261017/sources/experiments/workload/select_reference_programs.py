#!/usr/bin/env python3
"""Select complete programs from the reproduced source, without changing rows."""
import argparse
import csv
import hashlib
import io
import json
from collections import OrderedDict
from pathlib import Path


def select(source, destination, count=4):
    raw = source.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    programs = OrderedDict()
    if len(rows) != len(lines)-1:
        raise ValueError("Expected one physical CSV line per round")
    for row, line in zip(rows, lines[1:]):
        programs.setdefault(row['session_id'], []).append((row, line))
    selected = []
    for program_id, entries in programs.items():
        values = [row for row, _ in entries]
        maximum = max(int(r['prefix_len'])+int(r['input_len'])+int(r['output_len'])+32 for r in values)
        prompt = max(int(r['prefix_len'])+int(r['input_len']) for r in values)
        output = sum(int(r['output_len']) for r in values)
        waits = sum(int(r['tool_wait_after_ms']) for r in values)
        if len(values)<=16 and output<=1536 and waits<=45000 and maximum<=16384 and prompt>=8192:
            selected.append((program_id, entries,dict(program_id=program_id,rounds=len(values),maximum_context_with_template_reserve=maximum,output_tokens=output,tool_wait_ms=waits)))
            if len(selected)==count:
                break
    if len(selected)!=count:
        raise ValueError('Not enough complete programs within frozen input bounds')
    destination.mkdir(parents=True,exist_ok=False)
    result = lines[0]+b''.join(line for _,entries,_ in selected for _,line in entries)
    (destination/'sessions.csv').write_bytes(result)
    manifest = dict(name='SWE-smith-tranjactory',selection_kind='complete-program-subset-of-historical-source',
        source_sessions_sha256=hashlib.sha256(raw).hexdigest(),sessions_sha256=hashlib.sha256(result).hexdigest(),
        criteria=dict(order='original source order',maximum_rounds=16,maximum_output_tokens=1536,maximum_tool_wait_ms=45000,maximum_context_with_template_reserve=16384,minimum_maximum_prompt_tokens=8192),
        modified_rows=False,programs=[item[2] for item in selected],
        scope='Token-length trajectory replay; no task repair accuracy claim; subset of original full source, not the historical 128-program subset')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--programs',type=int,default=4)
    args=parser.parse_args();print(json.dumps(select(args.source,args.destination,args.programs),indent=2))
