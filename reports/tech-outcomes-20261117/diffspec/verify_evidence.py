"""Verify published files and frozen statistics offline; never execute archived code."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--assets', type=Path, required=True,
                    help='Directory holding the Release assets named in release-manifest.json')
args = parser.parse_args()

manifest = json.loads((ROOT / 'release-manifest.json').read_text())
for item in manifest['assets']:
    path = args.assets / item['name']
    assert path.stat().st_size == item['bytes'], item['name']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['name']

base = ROOT / 'functional/attempt02'
traces = {rank: [json.loads(line) for line in (base / 'artifacts/traces' /
          f'rank-{rank}.jsonl').read_text().splitlines()] for rank in range(4)}
by_cycle = {rank: {row['retrieval_cycle']: row for row in rows}
            for rank, rows in traces.items()}
cycles = set(by_cycle[0])
assert all(set(rows) == cycles for rows in by_cycle.values())
padded = []
for cycle in sorted(cycles):
    events = [by_cycle[rank][cycle] for rank in range(4)]
    selections = [[(row['slot'], row['committed_prefix_end'], row['selected_chunks'],
                   row['selected_tokens_sha256']) for row in event['selected']] for event in events]
    assert all(selection == selections[0] for selection in selections)
    if all(event['real_request_count'] == 2 and event['metadata_rows'] == 3 for event in events):
        assert all(event['dummy_excluded'] and event['kernel_boundaries_match']
                   and event['kernel_queries_match_last_committed'] for event in events)
        assert all(row['rejected_suffix_excluded'] for event in events for row in event['selected'])
        assert all(event['has_rejected_suffix'] for event in events)
        padded.append(cycle)
assert len(padded) == 21 and len(cycles) == 24
functional = tarfile.open(args.assets / 'diffspec-functional-20261115.tar.gz')
outputs = json.loads(functional.extractfile('functional/attempt02/artifacts/outputs.json').read())
assert len(outputs) == 2 and all(len(row['token_ids']) == 256 for row in outputs)
assert all(not row['metrics']['is_corrupted'] and row['metrics']['num_preemptions'] == 0 for row in outputs)
source = json.loads((ROOT / 'functional/pr11-source-manifest.json').read_text())
runtime_sha = next(row['sha256'] for row in source['files'] if row['path'] == 'diffspec/runtime.py')
assert all(row['runtime_sha256'] == runtime_sha for row in
           json.loads((base / 'artifacts/loaded-workers.json').read_text()))

frozen = tarfile.open(args.assets / 'diffspec-review-20261030.tar.gz')
prefix = 'diffspec-results-20261030/'
reference = tarfile.open(fileobj=io.BytesIO(frozen.extractfile(
    prefix + 'archive/reference-measurements.tar.gz').read()))
results = json.loads(frozen.extractfile(prefix + 'results.json').read())
assert (ROOT / 'historical/results.json').read_bytes() == frozen.extractfile(prefix + 'results.json').read()
matched = 0
for row in results['rows']:
    raw = []
    for folder in row['source']['measurement_folders']:
        name = folder.removeprefix('/root/') + '/summary.json'
        archive = reference if name.startswith('diffspec-fusion-eval-20261030/') else frozen
        raw.append(json.loads(archive.extractfile(name).read()))
    assert statistics.median(item['tokens_per_second'] for item in raw) == row['median_tokens_per_second']
    assert sum(item['requests'] for item in raw) == row['requests']
    matched += 1
assert matched == 9
print(json.dumps({'passed': True, 'source_commit': source['commit'],
    'two_real_plus_dummy_cycles': len(padded), 'four_rank_equal_cycles': len(cycles),
    'actual_requests': len(outputs), 'historical_cells_recomputed': matched,
    'release_assets_verified': len(manifest['assets']), 'device_experiments_started': 0}, indent=2))
