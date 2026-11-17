"""Larger genuine task pool, fixed in advance; no HBM byte limit."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

PARENT = Path('/root/CGCL/kv-tiering-performance-recheck-20261116')
ROOT = PARENT / 'pressure-measurements-v1'
DATA = PARENT / 'swe128-pressure'
PREVIOUS = Path('/root/CGCL/kv-tiering-working-set-20261116')
OLD = Path('/root/CGCL/kv-tiering-recovery-validation-20261112')
PY = '/root/CGCL/kv-tiering-benefit-search-20261108/runtime-venv-v1/bin/python'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    with (ROOT / 'controller-start.json').open('x') as handle:
        json.dump({'pid': os.getpid()}, handle)
    campaign = module('campaign', PARENT / 'swe32-measurements-v1/run_swe_campaign.py')
    campaign.ROOT = ROOT
    event = campaign.event
    event(stage='waiting_for_early_probe')
    started = time.monotonic()
    while True:
        prior = json.loads((PARENT / 'early-admission-measurements-v1/STATUS.json').read_text())
        if prior['stage'] == 'failed':
            raise RuntimeError('Earlier experiment failed; inspect before more model runs')
        if prior['stage'] == 'complete':
            break
        if time.monotonic() - started > 14400:
            raise TimeoutError('Previous experiments have not completed')
        time.sleep(5)
    hardware = subprocess.check_output(['npu-smi', 'info'], text=True)
    (ROOT / 'npu-before.txt').write_text(hardware)
    if hardware.count('No running processes found') != 2:
        raise RuntimeError('Both NPUs must be free before capacity pressure test')
    audit = json.loads((DATA / 'swe-data-audit.json').read_text())
    assert audit['source_tasks'] == 128
    assert all(c['length_matches'] for c in audit['frozen_protocol_length_checks'])
    assert hashlib.sha256((DATA / 'swe-plan.json').read_bytes()).hexdigest() == audit['plan_sha256']
    for task in audit['tasks']:
        for request in task['requests']:
            content = json.loads(Path(request['prompt_file']).read_text())
            assert hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest() == request['prompt_sha256']
    for name in ('swe-plan.json', 'swe-data-audit.json'):
        shutil.copyfile(DATA / name, ROOT / name)
    shutil.copyfile(PREVIOUS / 'revision-v2/local-source-manifest.json', ROOT / 'local-source-manifest.json')
    campaign.save(ROOT / 'plan.json', {
        'scope': 'single exploratory pressure triple; not statistical confirmation or official score',
        'source_tasks': 128, 'eligible_tasks': audit['selected_tasks'],
        'whole_task_exclusions': audit['whole_task_exclusions'],
        'requests_each': audit['complete_published_requests'], 'data_sha256': audit['plan_sha256'],
        'order': ['pressure-native', 'pressure-default', 'pressure-early'],
        'full_hbm_utilization': .92, 'fixed_hbm_limit': False,
        'max_model_len': 262144, 'max_num_seqs': 128, 'max_num_batched_tokens': 4096,
        'client_concurrency': audit['selected_tasks'], 'added_cpu_bytes_total_candidate': 14 * 1024**3,
        'fs': False, 'mtp': False, 'output_tokens_per_request': 32,
        'request_timeout_seconds': 1800, 'client_total_timeout_seconds': 7200,
        'early_only_difference': 'tiering_hbm_pressure_aware=false',
        'timing': 'all cold and later requests; phase wall and whole client wall reported separately',
        'confirmation_rule': 'if effective recovery produces net gain, repeat independently before claiming stability',
    })
    fixtures = ROOT / 'fixtures'
    fixtures.mkdir()
    for name in ('serve_supervisor.py', 'hybrid-cpu-qual.argv.json'):
        shutil.copyfile(OLD / name, fixtures / name)
    client = (OLD / 'run_quality.py').read_text()
    assert client.count('total=600,') == 1
    (fixtures / 'run_quality.py').write_text(client.replace('total=600,', 'total=1800,'))
    source = (PREVIOUS / 'run_validation.py').read_text()
    assert source.count('timeout=2400') == 1
    (ROOT / 'validation-source.py').write_text(source.replace('timeout=2400', 'timeout=7200'))
    validation = module('validation', ROOT / 'validation-source.py')
    validation.ROOT, validation.PLUGIN, validation.event = ROOT, PREVIOUS / 'revision-v2/plugin', event
    validation.OLD = fixtures
    original_config = validation.config
    def config(label, limited, mode):
        cfg, src = original_config(label, limited, mode)
        cfg['argv'][cfg['argv'].index('--max-model-len') + 1] = '262144'
        if label == 'pressure-early':
            index = cfg['argv'].index('--kv-transfer-config') + 1
            kv = json.loads(cfg['argv'][index])
            kv['kv_connector_extra_config']['tiering_hbm_pressure_aware'] = False
            cfg['argv'][index] = json.dumps(kv)
        cfg['additional_sources'] = [str(Path(__file__)), str(ROOT / 'plan.json'),
            str(ROOT / 'swe-plan.json'), str(DATA / 'prepare_pressure_swe.py'),
            str(fixtures / 'run_quality.py')]
        assert '--kv-cache-memory-bytes' not in cfg['argv']
        assert '--num-gpu-blocks-override' not in cfg['argv']
        return cfg, src
    validation.config = config
    monitor_module = module('monitor', PARENT / 'run_recheck.py')
    monitor_module.ROOT = ROOT
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_module.metric_monitor, args=(stop,), daemon=True)
    monitor.start()
    try:
        for label, mode in [('pressure-native', 'native'), ('pressure-default', 'working-set'),
                            ('pressure-early', 'working-set')]:
            validation.arm(label, False, mode, ROOT / 'swe-plan.json')
            rows = [json.loads(line) for line in
                (ROOT / (label + '-client/requests.jsonl')).read_text().splitlines()]
            assert len(rows) == audit['complete_published_requests']
            assert all(row['usage']['prompt_tokens'] == row['prompt_tokens'] for row in rows)
        event(stage='complete')
    finally:
        stop.set()
        monitor.join(timeout=10)
        (ROOT / 'npu-after.txt').write_text(subprocess.check_output(['npu-smi', 'info'], text=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        (ROOT / 'STATUS.json').write_text(json.dumps({'stage': 'failed', 'error': repr(exc)}, indent=2) + '\n')
        raise
