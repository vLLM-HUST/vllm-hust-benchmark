import argparse
import collections
import csv
import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--arm', choices=['A', 'B'], required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
expected = (['aclnnInplaceCopy_SliceAiCore_Slice'] * 3 +
            ['aclnnCat_ConcatD_ConcatD'] + ['l2norm_fwd_kernel2'] * 2
            if a.arm == 'A' else ['unpack_norm_kernel'])
report = {'arm': a.arm, 'root': str(a.root), 'ranks': [],
          'scope': 'T16 pure decode, post-CausalConv1d to RecurrentGatedDeltaRule; Q/K8x128 V16x128; packed row4096',
          'verdict': 'diagnostic only; full-model numerical gate unresolved',
          'limits': ['One light window per arm; not three repeated windows.',
                     'All-kernel denominator includes communication, per rank.',
                     'Compare target per-call means only when source shape and weights match; whole windows may differ.']}
for name in ['cohort.json', 'load-manifest.json']:
    report[name] = json.loads((a.root / name).read_text())
for path in sorted(a.root.glob('*/ASCEND_PROFILER_OUTPUT/kernel_details.csv')):
    rows = list(csv.DictReader(path.open()))
    streams = collections.defaultdict(list)
    for row in rows:
        streams[(row['Device_id'], row['Stream ID'])].append(row)
    calls = []
    for stream, seq in streams.items():
        seq.sort(key=lambda r: Decimal(r['Start Time(us)'].strip()))
        start = None
        for i, row in enumerate(seq):
            if row['Name'] == 'CausalConv1d':
                start = i if row['Input Shapes'].startswith('"16,4096;') else None
            if 'RecurrentGatedDeltaRule' not in row['Name']:
                continue
            if start is None:
                continue
            assert row['Input Shapes'].startswith('"16,8,128;16,8,128;16,16,128;'), row
            chain = seq[start + 1:i]
            names = [x['Name'] for x in chain]
            selected = [x for x in chain if x['Name'] in expected]
            assert [x['Name'] for x in selected] == expected, names
            assert names.count('fused_gdn_gating_kernel') == 1, names
            assert len(chain) == len(expected) + 1, names
            calls.append({'stream': stream, 'recurrent_start_us': row['Start Time(us)'].strip(),
                          'kernel_count': len(selected),
                          'kernel_active_us': sum(float(x['Duration(us)']) for x in selected),
                          'names': [x['Name'] for x in selected]})
            start = None
    assert calls, path
    total = sum(float(x['Duration(us)']) for x in rows)
    active = sum(c['kernel_active_us'] for c in calls)
    report['ranks'].append({'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'all_kernel_count': len(rows), 'all_kernel_us': total,
                            'target_calls': len(calls), 'target_kernel_count': sum(c['kernel_count'] for c in calls),
                            'target_active_us': active, 'target_fraction_all_kernel': active / total,
                            'active_us_per_call_mean': statistics.mean(c['kernel_active_us'] for c in calls),
                            'active_us_per_call_median': statistics.median(c['kernel_active_us'] for c in calls),
                            'all_fused_kernel_count': sum(x['Name'] == 'unpack_norm_kernel' for x in rows),
                            'calls': calls})
assert report['ranks']
with a.out.open('x') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
    f.write('\n')
for rank in report['ranks']:
    print({k: v for k, v in rank.items() if k not in ['calls', 'sha256']})
