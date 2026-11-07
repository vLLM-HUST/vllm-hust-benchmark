"""Solo variant of run_quality_controlled_v3.py (adds solo driver, bounds, devices). Own a finite quality process group; preserve frozen producers, failures and teardown."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--role', choices=['native', 'raw', 'group'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--program', type=Path, required=True)
    parser.add_argument('--timeout', type=int, required=True)
    parser.add_argument('--max-steps', type=int, required=True)
    parser.add_argument('--max-wall-seconds', type=int, required=True)
    parser.add_argument('--devices', default='0,1,2,3')
    parser.add_argument('--admission', choices=['solo', 'batched'], default='solo')
    parser.add_argument('--tp-override', type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'runs' / args.label
    root.mkdir(parents=True, exist_ok=False)
    producer = Path(__file__).read_bytes()
    (root / 'producer-controller.py').write_bytes(producer)
    config = args.config.read_bytes()
    plan = (args.frozen / 'plan.json').read_bytes()
    (root / 'engine-args.json').write_bytes(config)
    (root / 'plan.json').write_bytes(plan)
    import shutil
    shutil.copytree(args.frozen, root / 'frozen')
    tool_dir = root / 'producer'
    tool_dir.mkdir()
    for name in ('execution-identity-v2.json', 'QUALITY-PLAN-V2.md', 'SHA256SUMS-v2', 'environment-decision-v2.json'):
        shutil.copy2(args.program.parent / name, root / name)
    for name in ('run_quality.py', 'score_quality.py', 'run_quality_solo.py'):
        shutil.copy2(args.program.parent / name, tool_dir / name)
    command = [sys.executable, str(tool_dir / 'run_quality_solo.py'), 'run',
               '--frozen', str(root / 'frozen'), '--plan-sha256',
               json.loads(plan)['plan_sha256'], '--engine-args', str(root / 'engine-args.json'),
               '--role', args.role, '--max-steps', str(args.max_steps), '--max-wall-seconds', str(args.max_wall_seconds), '--output', str(root / 'results'), '--admission', args.admission] + (['--tp-override', str(args.tp_override)] if args.tp_override else [])
    env = os.environ.copy()
    env['ASCEND_RT_VISIBLE_DEVICES'] = args.devices
    env['OMP_NUM_THREADS'] = '1'
    env['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    env['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    env['PYTHONPATH'] = str(args.source / 'src') + ':' + env.get('PYTHONPATH', '')
    env['VLLM_CACHE_ROOT'] = str(root / 'compiler-cache')
    record = {'label': args.label, 'started_at': now(), 'command': command,
              'producer_sha256': hashlib.sha256(producer).hexdigest(),
              'source': str(args.source), 'deadline_seconds': args.timeout,
              'source_manifest_scope': 'Copied quality identity/SHA manifests refer to the original quality collection; local frozen/plan.json and child producer digests bind this attempt snapshot.',
              'source_commit': subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_status': subprocess.check_output(['git', '-C', str(args.source), 'status', '--porcelain'], text=True),
              'config_sha256': hashlib.sha256(config).hexdigest(),
              'plan_file_sha256': hashlib.sha256(plan).hexdigest(),
              'environment': {k: env.get(k) for k in ['ASCEND_RT_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_ENABLE_V1_MULTIPROCESSING', 'PYTHONPATH', 'VLLM_CACHE_ROOT', 'HCCL_DETERMINISTIC', 'VLLM_BATCH_INVARIANT']},
              'scope': 'One fresh fixed MCQ quality diagnostic; collection success is not quality acceptance or performance qualification.',
              'passed': False}

    def save():
        (root / 'controller.json').write_text(json.dumps(record, indent=2) + '\n')

    (root / 'npu-before.txt').write_bytes(subprocess.check_output(['npu-smi', 'info']))
    proc = None
    save()
    try:
        with (root / 'process.log').open('x') as log:
            proc = subprocess.Popen(command, cwd=args.source, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = record['process_group'] = proc.pid
            save()
            started = time.monotonic()
            next_notice = 0
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > args.timeout:
                    record['timeout'] = True
                    raise TimeoutError('Finite diagnostic exceeded deadline')
                if elapsed >= next_notice:
                    print(f'{args.label}: pid={proc.pid} running {elapsed:.0f}s', flush=True)
                    next_notice = elapsed + 30
                time.sleep(1)
            record['exit_code'] = proc.returncode
            report = root / 'results/run.json'
            child_report = json.loads(report.read_bytes()) if report.exists() else {}
            record['passed'] = proc.returncode == 0 and child_report.get('collection_complete', False) and child_report.get('shutdown_returned', False) and not child_report.get('errors')
    except BaseException as exc:
        record['supervisor_error'] = repr(exc)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                record['forced_termination'] = True
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            record['exit_code'] = proc.returncode
        record['finished_at'] = now()
        (root / 'npu-after.txt').write_bytes(subprocess.check_output(['npu-smi', 'info']))
        save()
    print(json.dumps({'passed': record['passed'], 'label': args.label, 'exit_code': record.get('exit_code'), 'directory': str(root)}), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
