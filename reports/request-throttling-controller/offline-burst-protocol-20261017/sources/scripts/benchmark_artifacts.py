"""Export verified replay statistics without private deployment or token payloads."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import gzip
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def public_receipt(value, source_sha256):
    if value['status'] != 'PASS' or value['evidence_type'] != 'replay' or value['quality'] is not None:
        raise ValueError('Only validated replay results may be exported')
    public = {key: value[key] for key in (
        'schema', 'name', 'cohort', 'run_id', 'status', 'evidence_type', 'records',
        'paired_comparisons', 'aggregate', 'quality', 'plan_sha256', 'limitations',
        'raw_archive_sha256', 'source_and_deployment_sha256')}
    public['packages'] = value['runtime']['packages']
    public['source_result_sha256'] = source_sha256
    return public


ROUND_FIELDS = (
    'session_id', 'round_idx', 'slot', 'program_arrival_seconds', 'priority_class',
    'latency_objective_seconds', 'sent_seconds', 'first_token_seconds', 'finished_seconds',
    'prefix_len', 'input_len', 'output_len', 'tool_wait_after_ms', 'engine_prompt_tokens',
    'engine_completion_tokens', 'engine_cached_tokens', 'status', 'prompt_token_ids_sha256',
    'previous_history_token_count', 'previous_history_sha256', 'inherited_history_sha256',
    'expected_engine_prompt_tokens', 'rendered_prompt_token_ids_sha256')


def backup(output, repository):
    output, repository = Path(output), Path(repository)
    source = output / 'matrix-result.json'
    value = json.loads(source.read_text())
    public = public_receipt(value, digest(source))
    run_id = value['run_id']
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,40}', run_id):
        raise ValueError('Invalid run ID')
    plan = json.loads((output / 'plan.json').read_text())
    if digest(output / 'plan.json') != value['plan_sha256']:
        raise ValueError('Frozen plan checksum mismatch')
    for filename, field in [('raw-data.tar.gz', 'raw_archive_sha256'),
                            ('source-and-deployment.tar.gz', 'source_and_deployment_sha256')]:
        if digest(output / filename) != value[field]:
            raise ValueError('Archive checksum mismatch')
    if not (repository / 'pyproject.toml').is_file():
        raise ValueError('Expected an existing benchmark repository checkout')
    parent = repository / 'reports/swe-smith-tranjactory'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / run_id
    if destination.exists():
        raise ValueError('Backup exists; never overwrite prior experiment')
    with tempfile.TemporaryDirectory(prefix='.replay-', dir=parent) as temporary:
        stage = Path(temporary) / run_id
        stage.mkdir()
        write(stage / 'matrix-result.json', public)
        config = {key: plan[key] for key in (
            'name', 'cohort', 'backend_count', 'programs', 'concurrencies', 'seeds',
            'transport', 'chat_template_kwargs', 'model_args', 'workload_sha256',
            'source_commit', 'source_dirty')}
        config.update(chip_count=2, node_count=1, evidence_type='replay', arrival_mode='closed-loop',
                      workload_semantics='Fixed content driven by SWE-smith trajectory lengths and tool waits')
        write(stage / 'run-config.json', config)
        runtime = value['runtime']
        runtime_public = {key: runtime[key] for key in (
            'status', 'cohort', 'packages', 'model_config_sha256', 'tokenizer_sha256',
            'round_trip_vocabulary_size', 'chat_template_tokens', 'input_rounds_checked') if key in runtime}
        runtime_public['capacity'] = {key: runtime['capacity'][key] for key in (
            'model', 'max_model_len', 'block_size', 'num_gpu_blocks', 'num_cpu_blocks',
            'kv_capacity_tokens_device', 'kv_capacity_tokens_host', 'gpu_memory_utilization')
            if key in runtime.get('capacity', {})}
        write(stage / 'runtime-identity.json', runtime_public)
        manifest = json.loads((output / 'workload/manifest.json').read_text())
        public_manifest = {key: manifest[key] for key in (
            'name', 'programs', 'program_ids', 'selection', 'sessions_sha256', 'source_sessions_sha256')}
        public_manifest['source_contract'] = {key: manifest['source_manifest'][key] for key in (
            'source', 'source_sha256', 'knobs', 'model_limit_filter', 'history_rounding_repair')
            if key in manifest.get('source_manifest', {})}
        write(stage / 'workload/manifest.json', public_manifest)
        shutil.copyfile(output / 'workload/sessions.csv', stage / 'workload/sessions.csv')
        if digest(stage / 'workload/sessions.csv') != plan['workload_sha256']:
            raise ValueError('Workload checksum mismatch')
        original_hashes = {}
        with tarfile.open(output / 'raw-data.tar.gz') as archive:
            identity_bytes = archive.extractfile('results/runtime-identity.json').read()
            original_hashes['results/runtime-identity.json'] = hashlib.sha256(identity_bytes).hexdigest()
            identity = json.loads(identity_bytes)
            controller_public = {key: identity[key] for key in (
                'implementation', 'upstream_commit', 'router_class', 'scheduler_interval_seconds',
                'metrics_interval_seconds', 'scheduling_enabled', 'acting_token_weight',
                'use_acting_token_decay', 'ordinary_request_policy', 'transport') if key in identity}
            controller_public['loaded_sources'] = [
                {'file': item['file'].split('/vendor/ThunderAgent/', 1)[1], 'sha256': item['sha256']}
                for item in identity['loaded_sources'] if '/vendor/ThunderAgent/' in item['file']]
            write(stage / 'controller-identity.json', controller_public)
            for record in value['records']:
                label = f"scan-{record['strategy']}-c{record['concurrency']}-seed{record['seed']}"
                if not re.fullmatch(r'scan-(?:vllm|Our-Controller)-c[1-9][0-9]*-seed-?[0-9]+', label):
                    raise ValueError('Invalid performance record identity')
                relative = f'results/{label}/summary.json'
                raw_summary = archive.extractfile(relative).read()
                original_hashes[relative] = hashlib.sha256(raw_summary).hexdigest()
                if original_hashes[relative] != record['summary_sha256']:
                    raise ValueError('Summary checksum mismatch')
                summary = json.loads(raw_summary)
                write(stage / 'results' / label / 'summary.json', {
                    key: summary[key] for key in ('started_at_wall', 'sessions_sha256', 'result')})
                relative = f'results/{label}/rounds.jsonl'
                rounds_hash = hashlib.sha256()
                with archive.extractfile(relative) as source_rounds, (stage / (relative + '.gz')).open('wb') as compressed, \
                        gzip.GzipFile(filename='', mode='wb', fileobj=compressed, mtime=0) as target:
                    for line in source_rounds:
                        rounds_hash.update(line)
                        row = json.loads(line)
                        target.write((json.dumps({key: row[key] for key in ROUND_FIELDS if key in row}) + '\n').encode())
                original_hashes[relative] = rounds_hash.hexdigest()
        # Use the actual frozen capsule, never source edited after measurement.
        with tarfile.open(output / 'source-and-deployment.tar.gz') as archive:
            for member in archive.getmembers():
                name = Path(member.name)
                selected = member.name in ('scripts/run_swe_smith_benchmark.py', 'scripts/benchmark_artifacts.py',
                    'scripts/collect_pod_results.py', 'src/thunderagent_controller_mod/upstream_manifest.json')
                selected |= member.name.startswith('experiments/workload/') and name.name in (
                    'replay_sessions.py', 'replay_requests.py', 'engine_capacity.py',
                    'engine_metrics.py', 'raw_observation.py', 'generate_workload.py')
                if selected and member.isfile() and not name.is_absolute() and '..' not in name.parts:
                    target = stage / 'sources' / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
        upstream = json.loads((stage / 'sources/src/thunderagent_controller_mod/upstream_manifest.json').read_text())
        loaded = {item['file']: item['sha256'] for item in controller_public['loaded_sources']}
        if identity['upstream_commit'] != upstream['commit'] or loaded != upstream['files']:
            raise ValueError('Actual loaded Controller source differs from frozen upstream manifest')
        write(stage / 'artifact-manifest.json', {
            'schema': 'request-throttling.replay-backup/v1', 'run_id': run_id,
            'archived_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Real-device replay statistics; no software task accuracy; no official baseline spec',
            'original_file_sha256': original_hashes,
            'raw_archive_sha256': value['raw_archive_sha256'],
            'source_and_deployment_sha256': value['source_and_deployment_sha256'],
            'excluded': ['Token payloads', 'Private endpoints and operational deployment archives',
                         'Smoke timings from formal performance records']})
        (stage / 'README.md').write_text(
            '# SWE-smith-tranjactory\n\n'
            'Real Ascend TP2 service replay, comparing Native vLLM with request-throttling-controller.\n'
            'This is trajectory-length and tool-wait replay; it does not execute software repair tests.\n'
            'See matrix-result.json for paired seed results and run-config.json for the fixed configuration.\n'
            'results/ contains published timing/token-count records. Token payloads and private deployment\n'
            'archives are retained locally. Published subsets have their own checksums; original hashes\n'
            'are retained separately. This directory does not register an official baseline or CI target.\n\n'
            'Run `python verify_public.py` to check file integrity, full round coverage, timing, paired inputs\n'
            'and completion-rate ratios. Hashes are integrity bindings, not signatures.\n')
        shutil.copyfile(Path(__file__).with_name('verify_swe_smith_backup.py'), stage / 'verify_public.py')
        def save_checksums():
            files = sorted(path for path in stage.rglob('*') if path.is_file() and path.name != 'SHA256SUMS')
            (stage / 'SHA256SUMS').write_text(''.join(
                f'{digest(path)}  {path.relative_to(stage)}\n' for path in files))
        save_checksums()
        spec = importlib.util.spec_from_file_location('replay_public_verifier',
            Path(__file__).with_name('verify_swe_smith_backup.py'))
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        write(stage / 'verification.json', verifier.verify(stage))
        save_checksums()
        stage.rename(destination)
    return destination
