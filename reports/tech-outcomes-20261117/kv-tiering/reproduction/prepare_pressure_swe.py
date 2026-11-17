"""Prepare published SWE query histories without executing tools or rewriting text."""
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path('/root/CGCL/kv-tiering-performance-recheck-20261116')
if len(sys.argv) > 1:
    ROOT = Path(sys.argv[1])


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def convert(query):
    if not isinstance(query, list) or len(query) < 2:
        raise ValueError('incomplete_published_history')
    if query[0].get('role') != 'system' or query[1].get('role') != 'user':
        raise ValueError('missing_original_initial_history')
    messages = []
    awaiting = set()
    for source in query:
        role = source.get('role')
        content = source.get('content')
        if role not in ('system', 'user', 'assistant', 'tool') or not isinstance(content, str):
            raise ValueError('invalid_role_or_missing_content')
        message = {'role': role, 'content': content}
        calls = source.get('tool_calls', [])
        if calls:
            if role != 'assistant':
                raise ValueError('tool_call_outside_assistant')
            message['tool_calls'] = []
            for call in calls:
                function = call['function']
                arguments = json.loads(function['arguments'])
                if not isinstance(arguments, dict):
                    raise ValueError('tool_arguments_not_object')
                if not call.get('id') or call['id'] in awaiting:
                    raise ValueError('invalid_tool_call_identity')
                awaiting.add(call['id'])
                message['tool_calls'].append({'id': call['id'], 'type': 'function',
                    'function': {'name': function['name'], 'arguments': arguments}})
        if role == 'tool':
            ids = source.get('tool_call_ids', [])
            if len(ids) != 1 or ids[0] not in awaiting:
                raise ValueError('ambiguous_or_unmatched_tool_response')
            message['tool_call_id'] = ids[0]
            awaiting.remove(ids[0])
        if role == 'assistant' and not content and not calls:
            raise ValueError('empty_published_assistant')
        messages.append(message)
    if awaiting:
        raise ValueError('unanswered_published_tool_call')
    return messages


def main():
    out = ROOT / 'swe-prepared'
    out.mkdir(exist_ok=False)
    selection = json.loads((ROOT / 'swe-selection.json').read_text())
    bundle = json.loads((ROOT / 'swe-agent-v110-tools.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained('/models/Qwen3.5-35B-A3B', local_files_only=True)
    frozen = json.loads((ROOT / 'public-agent-trace-fixed32-v1.json').read_text())
    reference = {(r['task_id'], r['source_index']): r for r in frozen['datasets']['swe']['requests']}
    tasks, exclusions, checkpoints = [], [], []
    excluded_tasks = []
    total_lcp = total_prompt = 0
    for task_index, source in enumerate(selection['sources']):
        data = (ROOT / 'swe-sources' / Path(source['cache_key']).name).read_bytes()
        assert len(data) == source['size']
        assert hashlib.sha256(data).hexdigest() == source['sha256']
        trace = json.loads(data)
        requests = []
        previous = []
        task_checkpoints = []
        task_total_lcp = task_total_prompt = 0
        task_over_context = False
        for index, step in enumerate(trace['trajectory']):
            try:
                messages = convert(step.get('query'))
            except (ValueError, KeyError, TypeError) as exc:
                exclusions.append({'task_id': source['task_id'], 'source_index': index,
                    'reason': str(exc)})
                continue
            tokens = tokenizer.apply_chat_template(messages, tools=bundle['tools'],
                add_generation_prompt=True, tokenize=True, enable_thinking=False,
                return_dict=False)
            if not isinstance(tokens, list) or not tokens or not isinstance(tokens[0], int):
                raise ValueError('unexpected_tokenizer_result')
            if len(tokens) + 32 > 261120:
                task_over_context = True
            lcp = 0
            for a, b in zip(previous, tokens):
                if a != b:
                    break
                lcp += 1
            previous = tokens
            task_total_lcp += lcp
            task_total_prompt += len(tokens)
            name = f'task-{task_index:02d}-step-{index:03d}'
            (out / (name + '.json')).write_text(json.dumps(tokens) + '\n')
            (out / (name + '-messages.json')).write_text(json.dumps(messages, ensure_ascii=False) + '\n')
            record = {'prefix_id': task_index * 1000 + index,
                'task_id': source['task_id'], 'source_index': index,
                'prompt_file': str(out / (name + '.json')), 'prompt_tokens': len(tokens),
                'prompt_sha256': digest(tokens), 'messages_sha256': digest(messages),
                'source_query_sha256': digest(step['query']), 'lcp_previous_tokens': lcp,
                'recorded_tool_seconds': float(step.get('execution_time', 0) or 0)}
            requests.append(record)
            if (source['task_id'], index) in reference:
                expected = reference[source['task_id'], index]
                task_checkpoints.append({'task_id': source['task_id'], 'source_index': index,
                    'expected_prompt_tokens': expected['prompt_tokens'], 'actual_prompt_tokens': len(tokens),
                    'length_matches': expected['prompt_tokens'] == len(tokens)})
        if not requests or task_over_context:
            excluded_tasks.append({'task_id': source['task_id'], 'reason': 'whole_task_exceeds_261120' if task_over_context else 'no_complete_published_history', 'complete_requests': len(requests), 'max_prompt_tokens': max((r['prompt_tokens'] for r in requests), default=0)})
            continue
        total_lcp += task_total_lcp
        total_prompt += task_total_prompt
        checkpoints.extend(task_checkpoints)
        tasks.append({'task_id': source['task_id'], 'published_steps': len(trace['trajectory']),
            'requests': requests})
    phases = []
    for turn in range(max(len(t['requests']) for t in tasks)):
        requests = [t['requests'][turn] for t in tasks if len(t['requests']) > turn]
        phases.append({'name': f'swe-turn-{turn:03d}', 'concurrency': len(tasks), 'requests': requests})
    plan = {'seed': 20261116, 'prefix_len': 0, 'suffix_len': 0, 'output_len': 32,
        'purpose': 'SWE published full-text histories, fixed-length prefill diagnostic; NOT official zero-reuse profile or live-agent quality',
        'phases': phases}
    (ROOT / 'swe-plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    audit = {'scope': plan['purpose'], 'selected_tasks': len(tasks), 'source_tasks': len(selection['sources']), 'whole_task_exclusions': excluded_tasks,
        'complete_published_requests': sum(len(t['requests']) for t in tasks),
        'published_steps': sum(t['published_steps'] for t in tasks),
        'logical_prompt_tokens': total_prompt, 'lcp_previous_tokens': total_lcp,
        'potential_lcp_fraction': total_lcp / total_prompt,
        'exclusions': exclusions, 'frozen_protocol_length_checks': checkpoints,
        'tasks': tasks, 'plan_sha256': hashlib.sha256((ROOT / 'swe-plan.json').read_bytes()).hexdigest(),
        'deviations_from_official': ['all complete published histories, no 16K first/last selection',
            'reusable prefixes; no fresh request salt', '32 forced output tokens, temperature0',
            'round barriers; historical tool delays not replayed', 'all source tasks frozen before performance; whole-task context eligibility261120', 'no task execution or task success scoring'],
        'no_prompt_truncation_padding_or_alignment': True}
    audit['tool_argument_representation'] = 'JSON strings parsed to objects for Qwen template; names and argument values preserved'
    (ROOT / 'swe-data-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k not in ('tasks', 'exclusions', 'frozen_protocol_length_checks')}, ensure_ascii=False))
    print('length_checks', json.dumps(checkpoints))
    print('exclusions', len(exclusions))


if __name__ == '__main__':
    main()
