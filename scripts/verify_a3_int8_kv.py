"""Verify immutable A3 evidence and card-to-observation metric identity."""
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/a3-int8-kv-20261104'
DATA = ROOT / 'website-publication/data'

def main():
    count = 0
    result = json.loads((REPORT / 'results.json').read_text())
    assert result == json.loads((DATA / 'a3-int8-kv-20261104.json').read_text())
    for archive in json.loads((REPORT / 'evidence-manifest.json').read_text()):
        path = REPORT / archive['archive']
        if archive.get('parts'):
            chunks = []
            for part in archive['parts']:
                raw = (REPORT / part['path']).read_bytes()
                assert len(raw) == part['bytes']
                assert hashlib.sha256(raw).hexdigest() == part['sha256']
                chunks.append(raw)
            packed = b''.join(chunks)
        else:
            packed = path.read_bytes()
        assert hashlib.sha256(packed).hexdigest() == archive['sha256']
        with tarfile.open(fileobj=io.BytesIO(packed), mode='r:gz') as tf:
            expected = {m['path']: m for m in archive['members']}
            assert set(tf.getnames()) == set(expected)
            summaries = {}
            requests = {}
            for member in tf:
                raw = tf.extractfile(member).read()
                info = expected[member.name]
                assert len(raw) == info['bytes']
                assert hashlib.sha256(raw).hexdigest() == info['sha256']
                if member.name.endswith('/summary.json'):
                    summaries[str(Path(member.name).parent)] = json.loads(raw)
                if member.name.endswith('/requests.jsonl'):
                    rows = [json.loads(line) for line in raw.splitlines()]
                    count += len(rows)
                    window_tokens = 0
                    for row in rows:
                        if 'token_events' in row:
                            assert sum(n for t, n in row['token_events']) == row['output_tokens']
                            window_tokens += sum(n for t, n in row['token_events'] if 0 <= t < 1800)
                    requests[str(Path(member.name).parent)] = (len(rows), window_tokens)
            for key, summary in summaries.items():
                assert requests[key][0] == summary['all_admitted_including_drain']['requests']
                if 'window_received' in summary:
                    assert requests[key][1] == summary['window_received']['output_tokens']
                    assert math.isclose(requests[key][1] / 1800, summary['window_received']['output_tokens_per_second'])
    assert count == sum(r['requests'] for r in result['runs']) == 7463
    assert all(not r['latency_pass'] for r in result['runs'])
    assert all(r['failed'] == 0 for r in result['runs'])
    card = next(e for e in json.loads((DATA / 'plugin-performance.json').read_text())['entries'] if e['id'] == result['plugin_id'])
    pairs = next(x['pairs'] for x in result['breakpoint']['results'] if x['concurrency'] == 64)
    comparisons = card['observations'][0]['comparisons']
    assert [(v['baseline'], v['candidate']) for v in comparisons] == [(p['bf16_tps'], p['int8_tps']) for p in pairs]
    assert len(pairs) == 4 and min(p['gain_pct'] for p in pairs) > 0
    assert sum(p['gain_pct'] for p in pairs) / 4 >= 3
    print(f'PASS: 3 archives, 30 runs, {count} requests; token accounting and website comparisons verified')

if __name__ == '__main__':
    main()
