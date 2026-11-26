"""Preselect a larger genuine task pool by identity, never measured performance."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'swe128-pressure'
PROTOCOL = BASE.parent.parent / 'reviews/efficientagent-portability-20261116/project-constraints/public-agent-trace-fixed32-v1.json'


def fetch(source):
    destination = ROOT / 'swe-sources' / Path(source['cache_key']).name
    original = BASE / 'swe32-final/swe-sources' / destination.name
    regional = source['url'].replace('scaleapi-results.s3.amazonaws.com',
        'scaleapi-results.s3.us-west-1.amazonaws.com')
    headers = destination.with_suffix('.headers')
    if original.exists():
        shutil.copyfile(original, destination)
    elif not destination.exists():
        subprocess.run(['curl', '--fail', '--location', '--connect-timeout', '10',
            '--max-time', '60', '--retry', '1', '--retry-all-errors', '--silent',
            '--show-error', regional, '--dump-header', str(headers), '-o', str(destination)], check=True)
    content = destination.read_bytes()
    parsed = json.loads(content)
    assert isinstance(parsed.get('trajectory'), list)
    sha = hashlib.sha256(content).hexdigest()
    if source.get('sha256'):
        assert sha == source['sha256'] and len(content) == source['size']
    frozen = dict(source, sha256=sha, size=len(content))
    frozen['sha_provenance'] = 'project frozen file lock' if source.get('sha256') else 'new source acquisition; bytes frozen before model testing'
    receipt = dict(task_id=source['task_id'], canonical_url=source['url'],
        regional_url=regional, sha256=sha, bytes=len(content), published_steps=len(parsed['trajectory']))
    return frozen, receipt


def main():
    (ROOT / 'swe-sources').mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    official = protocol['datasets']['swe']['sources']
    known = {source['task_id'] for source in official}
    extra = sorted(set(protocol['datasets']['swe']['available_published_task_ids']) - known,
        key=lambda task: hashlib.sha256(('kv-tiering-pressure-pool-v1\0' + task).encode()).hexdigest())[:96]
    sources = list(official)
    for task in extra:
        sources.append({'task_id': task, 'cache_key': 'swe/' + task + '.traj',
            'url': 'https://scaleapi-results.s3.amazonaws.com/swe-bench-pro/claude-sonnet-4-paper/traj/' + task + '/' + task + '.traj'})
    selection = {'selection': 'project fixed32 plus first96 other published V1 task IDs by SHA256(kv-tiering-pressure-pool-v1 NUL task_id); no replacements or outcome filters',
        'scope': 'new capacity-pressure diagnostic; not project fixed32 official profile', 'sources': sources}
    (ROOT / 'task-selection-before-acquisition.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, source): source for source in sources}
        for future in concurrent.futures.as_completed(futures):
            source = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({'task_id': source['task_id'], 'error': repr(exc)})
            print(json.dumps({'completed': len(results), 'failed': len(errors), 'total': 128}), flush=True)
    (ROOT / 'acquisition-errors.json').write_text(json.dumps(errors, indent=2) + '\n')
    if errors:
        raise RuntimeError('Acquisition failed; keep frozen IDs, do not substitute tasks')
    by_task = {source['task_id']: (source, receipt) for source, receipt in results}
    selection['sources'] = [by_task[source['task_id']][0] for source in sources]
    (ROOT / 'swe-selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'download-receipt.json').write_text(json.dumps([by_task[source['task_id']][1] for source in sources], indent=2) + '\n')
    shutil.copyfile(PROTOCOL, ROOT / PROTOCOL.name)
    shutil.copyfile(BASE / 'swe-agent-v110-tools.json', ROOT / 'swe-agent-v110-tools.json')


if __name__ == '__main__':
    main()
