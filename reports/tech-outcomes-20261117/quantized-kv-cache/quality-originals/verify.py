"""Verify published GSM8K originals without accessing the experiment machine."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute', action='store_true', help='Also run original summarize.py in a temporary directory; requires numpy')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    chunks = []
    for part in manifest['parts']:
        raw = (root / part['path']).read_bytes()
        assert len(raw) == part['bytes'] and digest(raw) == part['sha256'], part['path']
        chunks.append(raw)
    packed = b''.join(chunks)
    assert len(packed) == manifest['bytes'] and digest(packed) == manifest['sha256']
    expected = {m['path']: m for m in manifest['members']}
    originals = {}
    with tarfile.open(fileobj=io.BytesIO(packed), mode='r:gz') as tf:
        assert len(tf.getmembers()) == len(expected)
        for member in tf:
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            assert member.name in expected and member.name not in originals
            raw = tf.extractfile(member).read()
            info = expected[member.name]
            assert len(raw) == info['bytes'] and digest(raw) == info['sha256'], member.name
            originals[member.name] = raw
    assert set(originals) == set(expected)
    for line in originals['SHA256SUMS'].decode().splitlines():
        sha, name = line.split('  ', 1)
        assert digest(originals[name]) == sha, name
    summary = json.loads(originals['summary.json'])
    arms = {}
    for arm in ('bf16', 'int8'):
        raw = [json.loads(line) for line in originals[arm + '-http.jsonl'].splitlines()]
        assert len(raw) == 201 and all(r['status'] == 200 for r in raw)
        samples = json.loads(originals[arm + '/private-results.json'])['samples']['gsm8k']
        records = {}
        for sample in samples:
            key = (sample['doc_id'], sample['filter'])
            assert key not in records
            records[key] = sample
        assert len(records) == 400
        assert {key[0] for key in records} == set(range(200))
        arms[arm] = (raw[1:], records)
    def request_hashes(rows):
        return sorted(digest(json.dumps(r['request'], sort_keys=True).encode()) for r in rows)
    assert request_hashes(arms['bf16'][0]) == request_hashes(arms['int8'][0])
    for filt, paired in summary['paired'].items():
        for arm in ('bf16', 'int8'):
            correct = sum(bool(arms[arm][1][i, filt]['exact_match']) for i in range(200))
            assert correct == paired[arm + '_correct']
        for i in range(200):
            b, q = (arms[arm][1][i, filt] for arm in ('bf16', 'int8'))
            assert b['doc_hash'] == q['doc_hash'] and b['prompt_hash'] == q['prompt_hash']
            row = paired['rows'][i]
            assert row['id'] == i
            assert row['bf16_correct'] == bool(b['exact_match'])
            assert row['int8_correct'] == bool(q['exact_match'])
    if args.recompute:
        with tempfile.TemporaryDirectory(prefix='gsm8k-originals-check-') as temp:
            for name, raw in originals.items():
                dest = Path(temp) / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(raw)
            subprocess.run([sys.executable, str(Path(temp) / 'summarize.py')], check=True, stdout=subprocess.DEVNULL)
            assert json.loads((Path(temp) / 'summary.json').read_text()) == summary
    print(f'PASS: {len(expected)} original files; 200 paired questions, both filters, HTTP requests, answers and scores verified')


if __name__ == '__main__':
    main()
