"""CPU-only contextual audit of four terminal batch7 receipt streams."""

import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path('/root/workspace/muxpd-quality-layout-20261028')
CANDIDATE = BASE / 'quality-runs/prefix-quality-full-01'
OUT = Path(__file__).resolve().parent
TARGET = 'mmlu-pro-07338'
LABELS = ['native1', 'native2', 'old_raw', 'prefix']
HISTORICAL = ['native-quality-v2-hccl-true-01',
              'native-quality-v2-hccl-true-02', 'raw-quality-v2-hccl-true-01']
EXPECTED_CANDIDATE = '344f459193c04c75ad675dcd8c07d7f32eedef5a892dfe2af14271268206e192'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    sources = []

    def bind(path, expected=None):
        identity = sha(path)
        need(expected is None or identity == expected, f'Source changed: {path}')
        sources.append({'path': str(path), 'sha256': identity, 'bytes': path.stat().st_size})
        return json.loads(path.read_text())

    manifest = bind(CANDIDATE / 'producer/source-manifest-v1.json')
    plan = bind(CANDIDATE / 'producer/frozen-v2/plan.json')
    producer = CANDIDATE / 'producer/run_quality.py'
    sources.append({'path': str(producer), 'sha256': sha(producer), 'bytes': producer.stat().st_size})
    need(sha(producer) == next(row['sha256'] for row in manifest['copied_files']
                             if row['destination'] == 'run_quality.py'), 'Receipt producer differs')
    batch = next(row for row in plan['batches'] if TARGET in row['case_ids'])
    members = batch['case_ids']
    need(batch['batch_id'] == 7 and len(members) == 8, 'Unexpected batch')
    seed = members[0]
    paths = [Path(manifest['historical_runs'][key]['path']) for key in HISTORICAL]
    paths.append(CANDIDATE / 'results/run.json')
    expected = [manifest['historical_runs'][key]['sha256'] for key in HISTORICAL]
    expected.append(EXPECTED_CANDIDATE)
    results, outputs = {}, {}
    windows = {}
    for label, path, expected_sha in zip(LABELS, paths, expected, strict=True):
        run = bind(path, expected_sha)
        need(run['finished_at'] and run['collection_complete'] and run['shutdown_returned']
             and not run['errors'], 'Run is not naturally terminal')
        need(run['plan_sha256'] == plan['plan_sha256'], 'Plan mismatch')
        admissions = [row for row in run['admissions'] if row['batch_id'] == 7]
        need(len(admissions) == 2 and admissions[0]['stage'] == 'seed'
             and admissions[1]['stage'] == 'peers', 'Unexpected admission stages')
        mapping = {item['internal']: item['case_id'] for row in admissions for item in row['bindings']}
        need(list(mapping.values()) == members and admissions[1]['seed_output_tokens'] == 1
             and admissions[1]['seed_still_live'], 'Original order/seed gate changed')
        inverse = {case_id: rid for rid, case_id in mapping.items()}
        outputs[label] = {row['case_id']: row for row in run['outputs'] if row['case_id'] in members}
        need(len(outputs[label]) == 8, 'Missing batch member output')
        receipt = path.parent / 'scheduler-receipts.jsonl'
        need(sha(receipt) == run['scheduler_receipts_sha256'], 'Receipt identity mismatch')
        sources.append({'path': str(receipt), 'sha256': sha(receipt), 'bytes': receipt.stat().st_size})
        schedules, transactions = {}, []
        with receipt.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['batch_id'] != 7:
                    continue
                step = row['global_step']
                if row['event'] == 'schedule':
                    need(step not in schedules and not row['preempted_req_ids'], 'Duplicate/preempted schedule')
                    schedules[step] = row
                else:
                    need(row['event'] == 'retirement' and step in schedules, 'Unmatched retirement')
                    schedule = schedules.pop(step)
                    need(not row['cancelled_prefill_ids'], 'Cancelled P in terminal batch')
                    for rid in row['decode_retired_ids']:
                        need(schedule['scheduled_tokens'][rid] == 1
                             and schedule['phases_before'][rid] == 'D', 'Retirement is not one-token D')
                    transactions.append((schedule, row))
        need(not schedules and transactions, 'Unfinished/empty receipt stream')
        counts, retirement_steps, previous_counts = Counter(), {c: [] for c in members}, {}
        target_steps = {}
        for schedule, retired in transactions:
            step = schedule['global_step']
            previous_counts[step] = dict(counts)
            for rid in retired['decode_retired_ids']:
                c = mapping[rid]
                counts[c] += 1
                retirement_steps[c].append(step)
                if c == TARGET:
                    target_steps[counts[c] + 1] = step
        # The receipts record num_output_tokens increases only after token1.
        # Every observed D transaction schedules exactly one token, speculation
        # is disabled by the frozen engine arguments, and all final lengths
        # independently equal one + the observed D-retirement count.
        need(not run['engine_args'].get('speculative_config'), 'Speculation invalidates token alignment')
        for c in members:
            need(counts[c] + 1 == len(outputs[label][c]['token_ids']), 'Token-count alignment failed')
        target_by_step = {step: token for token, step in target_steps.items()}
        tx_by_step = {schedule['global_step']: (schedule, retired) for schedule, retired in transactions}
        member_finishes = []
        for c in members:
            rid = inverse[c]
            last = retirement_steps[c][-1]
            later = [(s, r) for s, r in transactions if s['global_step'] > last]
            need(all(rid not in s['phases_before'] and rid not in s['scheduled_tokens']
                     and rid not in r['decode_retired_ids'] for s, r in later), 'Request reappears after final retirement')
            member_finishes.append({
                'case_id': c, 'role': 'seed' if c == seed else 'peer',
                'output_tokens': len(outputs[label][c]['token_ids']),
                'finish_reason': outputs[label][c]['finish_reason'],
                'observed_decode_retirement_count': counts[c],
                'final_retirement_global_step': last,
                'target_output_token_at_same_step': target_by_step.get(last),
                'absent_from_all_later_scheduler_receipts': True,
                'next_transaction_global_step': later[0][0]['global_step'] if later else None,
            })
        selected = []
        for target_token in range(190, 221):
            step = target_steps[target_token]
            schedule, retired = tx_by_step[step]
            before = {c: previous_counts[step].get(c, 0) + 1 for c in members}
            after = dict(before)
            for rid in retired['decode_retired_ids']:
                after[mapping[rid]] += 1
            selected.append({
                'target_output_token_one_based': target_token, 'global_step': step,
                'derived_output_token_counts_before': before,
                'derived_output_token_counts_after': after,
                'scheduled_case_ids_in_saved_dict_order': [mapping[rid] for rid in schedule['scheduled_tokens']],
                'scheduler_requests_case_ids_in_saved_dict_order': [mapping[rid] for rid in schedule['phases_before']],
                'decode_retired_case_ids_in_saved_order': [mapping[rid] for rid in retired['decode_retired_ids']],
                'raw_schedule_receipt': schedule, 'raw_retirement_receipt': retired,
            })
        first_peer_step = retirement_steps[TARGET][0] - 1
        seed_count_at_first_peer_output = 1 + sum(step <= first_peer_step for step in retirement_steps[seed])
        central = [row for row in selected if 200 <= row['target_output_token_one_based'] <= 210]
        need(all(row['scheduled_case_ids_in_saved_dict_order'] == members
                 and row['decode_retired_case_ids_in_saved_order'] == members
                 and row['scheduler_requests_case_ids_in_saved_dict_order'] == members
                 for row in selected), 'Member set/order changes in expanded window')
        results[label] = {
            'admissions': admissions, 'bindings_internal_to_case': mapping,
            'member_finishes': member_finishes,
            'finish_order': [row['case_id'] for row in sorted(member_finishes, key=lambda row: row['final_retirement_global_step'])],
            'first_peer_output_global_step_derived': first_peer_step,
            'seed_output_count_at_first_peer_output_derived': seed_count_at_first_peer_output,
            'seed_lead_over_target_in_output_tokens': central[0]['derived_output_token_counts_after'][seed] - 200,
            'target207_global_step': target_steps[207],
            'target207_seed_output_count_after': central[7]['derived_output_token_counts_after'][seed],
            'all8_members_scheduled_and_retired_in_original_saved_order_target190_to220': True,
        }
        windows[label] = selected

    member_comparisons = []
    for c in members:
        row = {'case_id': c, 'role': 'seed' if c == seed else 'peer', 'outputs': {}, 'prefix_vs': {}}
        for label in LABELS:
            output = outputs[label][c]
            row['outputs'][label] = {'tokens': len(output['token_ids']), 'finish_reason': output['finish_reason'],
                                     'token_ids_sha256': hashlib.sha256(json.dumps(output['token_ids'], separators=(',', ':')).encode()).hexdigest()}
        candidate = outputs['prefix'][c]['token_ids']
        for reference in LABELS[:-1]:
            tokens = outputs[reference][c]['token_ids']
            same = tokens == candidate
            n = next((i for i, (a, b) in enumerate(zip(tokens, candidate)) if a != b), min(len(tokens), len(candidate)))
            row['prefix_vs'][reference] = {'full_tokens_equal': same, 'common_prefix_tokens': n,
                                          'first_difference_output_token_one_based': None if same else n + 1,
                                          'reference_token_id': tokens[n] if n < len(tokens) else None,
                                          'prefix_token_id': candidate[n] if n < len(candidate) else None}
        member_comparisons.append(row)

    summary = {
        'schema': 'muxpd-new-loss-context-v1', 'passed': True, 'target': TARGET, 'batch': batch,
        'derived_runs': results,
        'observed_conclusion': 'No scheduler-level member removal, natural finish, or saved dict-order change near target207; all8 scheduled and D-retired in original order through target190..220 in all4 runs.',
        'seed_runahead': 'Native seed leads target by1 output token; raw/prefix lead by10 (9 extra). Prefix/raw seed ends at target435, native seed at target288, not near207.',
        'earliest_other_finish': 'Native seed at target288; raw peer07213 at target237; prefix peer07213 at target302.',
        'limits': [
            'Saved scheduled_tokens/phases_before preserve scheduler dictionary insertion order, not physical model-runner tensor row order.',
            'decode_retired_ids records requests whose num_output_tokens increased; it does not directly record increments or finished_req_ids. One-token scheduling plus final token count=len(retirements)+1 validates derived token alignment.',
            'Final finish step is inferred from final D-retirement, saved stop/length output, and absence in every later scheduler receipt; not an explicit per-step finished event.',
            'No saved actual decode input_batch order, row index, graph bucket, KV/slot map, logits, layer tensors or device timing near207; this audit cannot exclude internal row changes or establish the numerical cause.',
            'Source-scoped observation only; no case-specific fallback, budget extension or production special case is proposed.',
        ],
        'next_probe_guidance': 'Preserve original batch and admission gate; instrument equivalent target190..220 D transactions for actual runner req_ids/row indices, graph bucket, positions/KV slots and target logits. Capture post-P seed/peer relative counts. Bridge the observer against unobserved outputs before numerical interpretation.',
        'producer_sha256': sha(Path(__file__)),
    }
    for name, value in [('summary.json', summary), ('members.json', member_comparisons),
                        ('target-window190-220.json', windows), ('source-identities.json', sources)]:
        with (OUT / name).open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
    print(json.dumps({'passed': True, 'member_rows': member_comparisons, 'runs': results}, indent=2))


if __name__ == '__main__':
    main()
