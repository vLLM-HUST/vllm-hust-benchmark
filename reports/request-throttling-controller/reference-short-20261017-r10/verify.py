#!/usr/bin/env python3
"""Independent public-capsule verification; no controller dependencies."""
import csv
import gzip
import hashlib
import json
from pathlib import Path


def verify(root):
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        path=root/name
        if root.resolve() not in path.resolve().parents:
            raise ValueError('Invalid manifest path')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    config=json.loads((root/'config.json').read_text())
    assert hashlib.sha256((root/'sessions.csv').read_bytes()).hexdigest()==config['expected_sessions_sha256']
    with (root/'sessions.csv').open() as stream:
        inputs=list(csv.DictReader(stream))
    expected={(r['session_id'],int(r['round_idx'])) for r in inputs}
    assert len(expected)==len(inputs)
    report=json.loads((root/'report.json').read_text());signatures=[]
    for point in report['points']:
        if point['status']!='PASS':continue
        directory=root/point['label'];summary=json.loads((directory/'summary.json').read_text())
        result=summary['result']
        assert summary['sessions_sha256']==config['expected_sessions_sha256']
        assert result['sessions_completed']==config['programs'] and not result['sessions_unfinished']
        assert not result['rounds_failed'] and result['rounds_succeeded']==len(expected)
        assert all(v['count']==0 for v in result['mismatches'].values())
        with gzip.open(directory/'rounds.jsonl.gz','rt') as stream:
            rows=[json.loads(line) for line in stream]
        assert len(rows)==len(expected)
        assert {(r['session_id'],r['round_idx']) for r in rows}==expected
        assert all(r['status']=='ok' and r['prompt_token_ids_sha256'] and r['previous_history_sha256']==r['inherited_history_sha256'] for r in rows)
        signatures.append({(r['session_id'],r['round_idx']):(r['prompt_token_ids_sha256'],r['engine_prompt_tokens'],r['engine_completion_tokens']) for r in rows})
    assert signatures and all(s==signatures[0] for s in signatures)
    assert report['completed_points_inputs_equal'] is True
    if report['status']=='PASS':
        assert signatures and report['paired_inputs_equal'] is True
        assert len(signatures)==len(report['points']) and all(s==signatures[0] for s in signatures)
    assert report['qualified_for_matrix'] is False
    print(json.dumps(dict(integrity='PASS',complete_points=len(signatures),input_rounds=len(expected),performance_qualification='UNQUALIFIED_SINGLE_SEED',trial_status=report['status'])))


if __name__=='__main__':verify(Path(__file__).resolve().parent)
