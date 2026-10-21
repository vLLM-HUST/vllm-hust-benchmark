"""Recompute the frozen lifecycle pair without discarding startup or shutdown."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def metrics(receipt, bench):
    assert receipt['status'] == 'measured' and receipt['all_devices_released']
    assert receipt['bench_exit_code'] == 0
    assert receipt['allocated_cards'] == 2 and receipt['reserved_cards'] == 4
    assert receipt['simultaneous_other_services'] == 0
    assert bench['completed'] == 200 and bench['failed'] == 0
    for name in ['output_lens', 'input_lens', 'ttfts', 'latencies', 'start_times', 'errors']:
        assert len(bench[name]) == 200, name
    assert not any(bench['errors'])
    output = sum(bench['output_lens'])
    assert output == bench['total_output_tokens'] and output > 0
    assert all(0 < count <= 256 for count in bench['output_lens'])
    assert sum(bench['input_lens']) == bench['total_input_tokens']
    wall = receipt['ended'] - receipt['started']
    mono = receipt['ended_monotonic'] - receipt['started_monotonic']
    assert abs(wall - mono) < 0.01, 'Clock discontinuity; lifecycle invalid'
    duration = bench['duration']
    assert 0 < duration < mono
    assert min(bench['start_times']) >= receipt['bench_started_monotonic'] - 1
    assert max(t + d for t, d in zip(bench['start_times'], bench['latencies'])) <= receipt['bench_ended_monotonic'] + 1
    throughput = output / duration
    assert math.isclose(throughput, bench['output_throughput'], rel_tol=1e-9)
    tpots = [(latency - first) / (count - 1) for latency, first, count in
             zip(bench['latencies'], bench['ttfts'], bench['output_lens']) if count > 1]
    assert math.isclose(statistics.mean(tpots)*1000, bench['mean_tpot_ms'], rel_tol=1e-8)
    return {
        'requests': 200, 'failed': 0, 'output_tokens': output,
        'lifecycle_s': mono, 'lifecycle_wall_s': wall,
        'pre_client_s': receipt['bench_started_monotonic'] - receipt['started_monotonic'],
        'client_s': receipt['bench_ended_monotonic'] - receipt['bench_started_monotonic'],
        'post_client_release_s': receipt['ended_monotonic'] - receipt['bench_ended_monotonic'],
        'burst_s': duration, 'output_tok_per_s': throughput,
        'mean_ttft_s': statistics.mean(bench['ttfts']), 'mean_tpot_s': statistics.mean(tpots),
        'p99_tpot_s': sorted(tpots)[math.ceil(.99*len(tpots))-1],
        'p99_sample_count': len(tpots), 'p99_nearest_rank': math.ceil(.99*len(tpots)),
        'lifecycle_reservation_card_s_per_token': 4*mono/output,
        'lifecycle_service_card_s_per_token': 2*mono/output,
        'request_reservation_card_s_per_token': 4*duration/output,
    }


def main():
    plan = json.loads((ROOT / 'm06-prechecked-pair-plan-r1.json').read_text())
    arms, inputs, receipts = {}, {}, {}
    for arm in plan['order']:
        out = ROOT / f'm01-recovery-model-m06-prechecked-{arm}-r1'
        receipt = json.loads((out / 'receipt.json').read_text())
        bench = json.loads((out / 'mmlu-pro.json').read_text())
        for name, digest in plan['source_sha256'].items():
            assert receipt['input_hashes'][name] == digest
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
        assert receipt['child_exit_codes'] == ([0, 130] if arm == 'b1' else [0, 0])
        log = (out / 'service.log').read_text()
        client_log = (out / 'bench.log').read_text()
        assert client_log.count('Initial test run completed.') == 1
        assert 'Skipping endpoint ready check.' not in client_log
        assert sum('POST /v1/chat/completions HTTP/1.1\" 200 OK' in line for line in log.splitlines()) == 201
        for line in (out / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
        if arm == 'b1':
            policy = [json.loads(line) for line in (out / 'kvmat-policy.jsonl').read_text().splitlines()]
            engine = [json.loads(line) for line in (out / 'kvmat-engine.jsonl').read_text().splitlines()]
            assert len(policy) == len(engine) == 201
            assert len({row['request_id'] for row in policy}) == 201
            assert len({row['cache_salt_sha256'] for row in engine}) == 201
            assert all(row['event'] == 'school028_request_salt_admitted' for row in engine)

        signals = [json.loads(line.split('M06_MANAGER_SIGNAL ', 1)[1])
                   for line in log.splitlines() if 'M06_MANAGER_SIGNAL ' in line]
        assert 'timeout=60s' in log
        if arm == 'b1':
            assert signals and signals[0]['signal'] == 2
        arms[arm] = metrics(receipt, bench)
        arms[arm]['shutdown'] = {
            'manager_signals': signals,
            'manager_sigkill': any(e['signal'] == 9 for e in signals),
            'engine_timeout_force_kill': 'force killing remaining process EngineCore' in log,
            'engine_dead_error_logged': 'EngineDeadError' in log,
            'api_shutdown_complete': 'Application shutdown complete.' in log,
        }
        inputs[arm] = bench['input_lens']
        receipts[arm] = receipt
    assert inputs['b0'] == inputs['b1']
    assert receipts['b1']['ended'] < receipts['b0']['started']
    b0, b1 = arms['b0'], arms['b1']
    cost = 100*(1-b1['lifecycle_reservation_card_s_per_token']/b0['lifecycle_reservation_card_s_per_token'])
    result = {'schema': 'm06-prechecked-lifecycle-result/v1', 'arms': arms,
        'lifecycle_cost_reduction_percent': cost, 'm06_lifecycle_gate_pass': cost >= 10,
        'request_cost_reduction_percent': 100*(1-b1['request_reservation_card_s_per_token']/b0['request_reservation_card_s_per_token']),
        'throughput_change_percent': 100*(b1['output_tok_per_s']/b0['output_tok_per_s']-1),
        'mean_ttft_reduction_percent': 100*(1-b1['mean_ttft_s']/b0['mean_ttft_s']),
        'mean_tpot_reduction_percent': 100*(1-b1['mean_tpot_s']/b0['mean_tpot_s']),
        'order': plan['order'], 'repetitions': 1, 'quality_scored': False,
        'original_prechecked_protocol_confirmation': True,
        'precheck_requests_per_arm': 1,
        'limitations': 'Output256 performance only; initial client precheck included in lifecycle time but not output denominator; equal-rate resource cost, not currency expense; no invented amortization or statistical certainty.'}
    with (ROOT / 'm06-prechecked-pair-result-r1.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
