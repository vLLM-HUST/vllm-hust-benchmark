"""Repeated independent restarts on frozen published SWE histories, full HBM."""
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
ROOT = PARENT / 'swe32-measurements-v1'
DATA = PARENT / 'swe32-final'
PREVIOUS = Path('/root/CGCL/kv-tiering-working-set-20261116')
PLUGIN = PREVIOUS / 'revision-v2/plugin'
PY = '/root/CGCL/kv-tiering-benefit-search-20261108/runtime-venv-v1/bin/python'
CLIENT_STARTS = {}


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def event(**value):
    if value.get('stage') == 'client_running':
        CLIENT_STARTS[value['arm']] = time.monotonic()
    if value.get('stage') == 'arm_passed' and value.get('arm') in CLIENT_STARTS:
        elapsed = time.monotonic() - CLIENT_STARTS[value['arm']]
        value['whole_client_wall_s'] = elapsed
        save(ROOT / (value['arm'] + '-client-wall.json'), {'seconds': elapsed,
            'scope': 'client process startup, every round and inter-round metrics waits; excludes server startup'})
    value['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(ROOT / 'STATUS.json', value)
    with (ROOT / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def main():
    with (ROOT / 'controller-start.json').open('x') as handle:
        json.dump({'pid': os.getpid()}, handle)
    event(stage='waiting_for_eligibility')
    started = time.monotonic()
    while True:
        prior = json.loads((PARENT / 'STATUS.json').read_text())
        if prior['stage'] == 'failed':
            raise RuntimeError('Eligibility controller failed; inspect before more model runs')
        if prior['stage'] == 'complete':
            break
        if time.monotonic() - started > 3600:
            raise TimeoutError('Eligibility controller did not finish')
        time.sleep(5)
    for name in ('native', 'default', 'forced-admission'):
        receipt = json.loads((PARENT / f'eligibility-{name}-server/exit.json').read_text())
        assert receipt['server_returncode'] == 0 and not receipt['forced_shutdown']
    hardware = subprocess.check_output(['npu-smi', 'info'], text=True)
    (ROOT / 'npu-before.txt').write_text(hardware)
    if hardware.count('No running processes found') != 2:
        raise RuntimeError('Both NPUs must be free before campaign')
    plan = json.loads((DATA / 'swe-plan.json').read_text())
    audit = json.loads((DATA / 'swe-data-audit.json').read_text())
    assert audit['selected_tasks'] == 32
    assert all(c['length_matches'] for c in audit['frozen_protocol_length_checks'])
    plan_hash = hashlib.sha256((DATA / 'swe-plan.json').read_bytes()).hexdigest()
    assert plan_hash == audit['plan_sha256']
    for task in audit['tasks']:
        for request in task['requests']:
            tokens = json.loads(Path(request['prompt_file']).read_text())
            canonical = json.dumps(tokens, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
            assert hashlib.sha256(canonical).hexdigest() == request['prompt_sha256']
    shutil.copyfile(DATA / 'swe-plan.json', ROOT / 'swe-plan.json')
    shutil.copyfile(DATA / 'swe-data-audit.json', ROOT / 'swe-data-audit.json')
    shutil.copyfile(PREVIOUS / 'revision-v2/local-source-manifest.json', ROOT / 'local-source-manifest.json')
    save(ROOT / 'plan.json', {
        'scope': plan['purpose'], 'requests_each': audit['complete_published_requests'],
        'tasks': 32, 'client_concurrency': 32, 'repetitions_independent_restarts': 3,
        'orders': [['native', 'working-set'], ['working-set', 'native'], ['native', 'working-set']],
        'gpu_memory_utilization': 0.92, 'fixed_hbm_limit': False,
        'max_model_len': 131072, 'max_num_seqs': 128, 'max_num_batched_tokens': 4096,
        'added_cpu_bytes_total': 14 * 1024**3, 'fs': False, 'mtp': False,
        'output_tokens_each_request': 32, 'ignore_eos': True,
        'data_sha256': plan_hash, 'no_live_task_quality_or_official_score': True,
        'measure_whole_finite_input_set': True, 'retain_failures_and_all_repetitions': True,
        'warmup_included_in_workload_cost': 'all cold query histories included; only model startup excluded',
        'timing': 'output tokens / sum of request-round walls; phase sampling waits excluded and separately recorded',
    })
    spec = importlib.util.spec_from_file_location('validation', PREVIOUS / 'run_validation.py')
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    validation.ROOT, validation.PLUGIN, validation.event = ROOT, PLUGIN, event
    original_config = validation.config
    def config(label, limited, mode):
        cfg, src = original_config(label, limited, mode)
        cfg['argv'][cfg['argv'].index('--max-model-len') + 1] = '131072'
        cfg['additional_sources'] = [str(Path(__file__)), str(ROOT / 'swe-plan.json'),
            str(ROOT / 'plan.json'), str(PARENT / 'prepare_swe_v3.py')]
        assert '--kv-cache-memory-bytes' not in cfg['argv']
        assert '--num-gpu-blocks-override' not in cfg['argv']
        return cfg, src
    validation.config = config
    monitor_spec = importlib.util.spec_from_file_location('monitor', PARENT / 'run_recheck.py')
    monitor_module = importlib.util.module_from_spec(monitor_spec)
    monitor_spec.loader.exec_module(monitor_module)
    monitor_module.ROOT = ROOT
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_module.metric_monitor, args=(stop,), daemon=True)
    monitor.start()
    try:
        orders = [('native', 'working-set'), ('working-set', 'native'), ('native', 'working-set')]
        for repetition, order in enumerate(orders, 1):
            for mode in order:
                label = f'r{repetition}-{mode}'
                validation.arm(label, False, mode, ROOT / 'swe-plan.json')
                rows = [json.loads(line) for line in
                    (ROOT / (label + '-client/requests.jsonl')).read_text().splitlines()]
                assert len(rows) == audit['complete_published_requests']
                assert all(row['usage']['prompt_tokens'] == row['prompt_tokens'] for row in rows)
                validation.check_sources()
            event(stage='pair_complete', repetition=repetition)
        event(stage='complete')
    finally:
        stop.set()
        monitor.join(timeout=10)
        (ROOT / 'npu-after.txt').write_text(subprocess.check_output(['npu-smi', 'info'], text=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        event(stage='failed', error=repr(exc))
        raise
