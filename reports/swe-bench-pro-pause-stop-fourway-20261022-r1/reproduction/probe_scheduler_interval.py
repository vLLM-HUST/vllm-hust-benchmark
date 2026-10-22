"""Finite fixed-trajectory, fixed-output real-engine scheduling ablation.

This is replay, not a new SWE-bench Agent result. All programs arrive together;
each retains its sequential request order and recorded tool delay.
"""
import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen


def call(base, path, body=None, headers=None, timeout=240):
    request = Request(base + path, data=None if body is None else json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json', **(headers or {})})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode()
        return json.loads(raw) if 'json' in response.headers.get('Content-Type', '') else raw


def idle(backend):
    text = call(backend, '/metrics')
    for key in ['num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc']:
        if not any(s.startswith('vllm:' + key + '{') for s in text.splitlines()):
            raise ValueError('Missing idle gauge: ' + key)
    gauges = {k: sum(float(s.rsplit(' ', 1)[1]) for s in text.splitlines()
                     if s.startswith('vllm:' + k + '{'))
              for k in ['num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc']}
    if any(gauges.values()):
        raise ValueError('Engine not drained: ' + repr(gauges))
    return text


def run_block(inputs, endpoint, block_id, output_tokens, request_timeout=240,
              cache_reuse='shared'):
    if cache_reuse not in ['shared', 'request-isolated']:
        raise ValueError('Unknown cache reuse mode')
    barrier = threading.Barrier(len(inputs))
    def program(row):
        pid = block_id + ':' + row['task']
        records = []
        release = None
        try:
            barrier.wait(timeout=30)
            for index, original in enumerate(row['rounds']):
                if index:
                    time.sleep(row['tool_wait_seconds'][index - 1])
                body = {**original, 'program_id': pid, 'max_tokens': output_tokens,
                        'ignore_eos': True, 'temperature': 0, 'seed': 17, 'stream': False}
                rid = pid + ':query-' + str(index + 1)
                if cache_reuse == 'request-isolated':
                    # The engine's documented salt isolates hash chains without
                    # changing prompt tokens, cache capacity or engine config.
                    body['cache_salt'] = rid
                started = time.monotonic()
                response = call(endpoint, '/v1/chat/completions', body,
                                {'X-Session-ID': pid, 'X-Delivery-Request-ID': rid}, request_timeout)
                ended = time.monotonic()
                usage = response['usage']
                records.append({'request_id': rid, 'round': index + 1, 'started': started,
                                'ended': ended, 'elapsed': ended - started, 'usage': usage})
                if usage['completion_tokens'] != output_tokens:
                    raise ValueError('Output work mismatch; no performance comparison')
        except Exception as exc:
            return {'task': row['task'], 'records': records, 'error': repr(exc)}
        finally:
            release = call(endpoint, '/programs/release', {'program_id': pid})
        return {'task': row['task'], 'records': records, 'release': release}
    with ThreadPoolExecutor(max_workers=len(inputs)) as executor:
        records = list(executor.map(program, inputs))
    flat = [q for p in records for q in p['records']]
    return {'programs': records, 'returns': len(flat),
            'errors': sum('error' in p for p in records),
            'makespan_seconds': max(q['ended'] for q in flat) - min(q['started'] for q in flat) if flat else None,
            'output_tokens': sum(q['usage']['completion_tokens'] for q in flat)}


def run(plan, root):
    request_timeout = plan.get('request_timeout_seconds', 240)
    if (isinstance(request_timeout, bool) or not isinstance(request_timeout, (int, float))
            or not math.isfinite(request_timeout) or request_timeout <= 0):
        raise ValueError('Request timeout must be finite and positive')
    raw = (root / plan['input_file']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan['input_sha256']:
        raise ValueError('Input checksum drift')
    inputs = json.loads(raw)
    if len(inputs) != plan['batch_size'] or len({p['task'] for p in inputs}) != len(inputs):
        raise ValueError('Batch membership mismatch')
    if plan['arrival_time'] != 0 or plan['client_concurrency_limit'] is not None:
        raise ValueError('Expected a finite time-zero burst without client slots')
    result = {'run_id': plan['run_id'], 'scope': 'fixed-trajectory replay, not real Agent completion', 'blocks': []}
    path = root / 'ablation-result-private.json'
    def save():path.write_text(json.dumps(result, indent=2) + '\n')
    def runtime():
        value = call(plan['backend'], '/collective_rpc', {'method': 'g7_shared_runtime_v1', 'timeout': 60})
        for row in value['results']:
            for key, expected in plan['expected_workers'][str(row['rank'])].items():
                if row[key] != expected:raise ValueError('Runtime drift: ' + key)
        return value
    result['runtime_before'] = runtime()
    for index, label in enumerate(plan['order']):
        group = plan['groups'][label]
        endpoint = group['endpoint']
        if call(endpoint, '/programs'):raise ValueError('Proxy not empty')
        before = call(endpoint, '/migration-identity')
        for key, expected in group['expected_identity'].items():
            if before.get(key) != expected:raise ValueError('Proxy identity drift: ' + key)
        metrics_before = idle(plan['backend'])
        call(plan['backend'], '/reset_prefix_cache', {})
        block = {'index': index, 'group': label, 'epoch_start': time.time(), 'identity_before': before,
                 'cache_reuse': group.get('cache_reuse', 'shared')}
        result['blocks'].append(block);save()
        observer = nullcontext()
        if plan.get('record_observations', False):
            from fixed_work_observation import FixedWorkObservation
            observer = FixedWorkObservation(root / f'block-{index}-observations',
                                           plan['backend'], endpoint,
                                           device=plan.get('device_observation'))
        with observer:
            block.update(run_block(inputs, endpoint, plan['run_id'] + ':' + str(index), plan['output_tokens'],
                                   request_timeout=request_timeout,
                                   cache_reuse=group.get('cache_reuse', 'shared')))
        for attempt in range(60):
            try:metrics_after = idle(plan['backend']);break
            except ValueError:
                if attempt == 59:raise
                time.sleep(.5)
        if call(endpoint, '/programs'):raise ValueError('Proxy failed to release programs')
        after = call(endpoint, '/migration-identity')
        block.update(epoch_end=time.time(), identity_after=after)
        (root / f'block-{index}-engine-before.txt').write_text(metrics_before)
        (root / f'block-{index}-engine-after.txt').write_text(metrics_after)
        if 'wait_aging' in before:
            block['action_delta'] = {k: after['wait_aging'][k] - before['wait_aging'][k]
                                     for k in ['aged_resumes', 'continuation_yields', 'wait_timeouts']}
        save();print('BLOCK_TERMINAL', index, label, block['returns'], block['makespan_seconds'], flush=True)
        if block['errors'] or block['returns'] != sum(len(p['rounds']) for p in inputs):
            raise ValueError('Incomplete work; stop before next block')
    result['runtime_after'] = runtime()
    result['status'] = 'COMPLETE_FIXED_WORK'
    save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    root = args.plan.parent
    (root / 'ablation-execution-intent.json').open('x').write(json.dumps({'epoch': time.time()}))
    run(json.loads(args.plan.read_text()), root)


if __name__ == '__main__':main()
