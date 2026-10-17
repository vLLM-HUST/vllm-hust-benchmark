#!/usr/bin/env python3
"""Compose disjoint long/short source windows using the canonical window builder."""
import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from build_quick_program_workload import build


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    for name in ['long-programs','short-programs','long-minimum-prefix','long-maximum-context',
                 'short-minimum-prefix','short-maximum-context','rounds','output-cap','tool-wait-cap-ms']:
        parser.add_argument('--'+name,type=int,required=True)
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Existing output; use a new workload revision')
    with tempfile.TemporaryDirectory() as directory:
        long=Path(directory)/'long';short=Path(directory)/'short'
        common=dict(rounds=a.rounds,output_cap=a.output_cap,tool_wait_cap_ms=a.tool_wait_cap_ms)
        lm=build(a.source,long,programs=a.long_programs,minimum_prefix=a.long_minimum_prefix,
                 maximum_context=a.long_maximum_context,**common)
        long_ids={w['session_id'] for w in lm['windows']}
        sm=build(a.source,short,programs=a.short_programs,minimum_prefix=a.short_minimum_prefix,
                 maximum_context=a.short_maximum_context,exclude_session_ids=sorted(long_ids),**common)
        rows=[]
        for part in [long,short]:
            with (part/'sessions.csv').open(newline='') as f:rows.extend(csv.DictReader(f))
        ids=list(dict.fromkeys(r['session_id'] for r in rows))
        assert len(ids)==a.long_programs+a.short_programs
        for r in rows:r['arrival_time']=str(ids.index(r['session_id']))
        a.output.mkdir(parents=True)
        with (a.output/'sessions.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        manifest=dict(name=a.output.name,purpose='Heterogeneous short replay windows; not full task quality',
            source_sessions_sha256=lm['source_sessions_sha256'],
            sessions_sha256=hashlib.sha256((a.output/'sessions.csv').read_bytes()).hexdigest(),
            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            window_builder_sha256=hashlib.sha256(Path(__file__).with_name('build_quick_program_workload.py').read_bytes()).hexdigest(),
            parameters={k:v for k,v in vars(a).items() if k not in ['source','output']},
            programs=len(ids),rounds=len(rows),order='Long windows first, then short windows; no outcome filtering',
            groups={'long':lm,'short':sm},boundary='Source prefix/input lengths preserved; windows and generation/tool waits changed. All positive and negative revisions retained.')
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({k:v for k,v in manifest.items() if k!='groups'}))

if __name__=='__main__':main()
