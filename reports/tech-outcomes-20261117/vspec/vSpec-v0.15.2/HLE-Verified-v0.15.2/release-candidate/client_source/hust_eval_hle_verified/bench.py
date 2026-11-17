"""Pure vLLM bench serve client, with frozen inputs and explicit request semantics."""
from __future__ import annotations
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List

from .hashing import sha256_file
from .bench_inputs import validate_input


def build_command(vllm_bin: str, base_url: str, served_model: str, tokenizer: str, dataset: str,
                  num_prompts: int, output_len: int, result_dir: str, result_name: str,
                  modality: str = 'text') -> List[str]:
    if modality not in ('text', 'image'):
        raise ValueError('unknown modality')
    extra = {'chat_template_kwargs': {'enable_thinking': False}, 'seed': 0, 'ignore_eos': False}
    return [vllm_bin, 'bench', 'serve', '--backend', 'openai-chat',
            '--base-url', base_url.rstrip('/'), '--endpoint', '/v1/chat/completions',
            '--model', served_model, '--tokenizer', tokenizer,
            '--dataset-name', 'custom_image' if modality == 'image' else 'custom',
            '--dataset-path', dataset, '--disable-shuffle', '--no-oversample',
            '--skip-chat-template', '--custom-output-len', str(output_len),
            '--num-prompts', str(num_prompts), '--request-rate', 'inf',
            '--temperature', '0', '--seed', '0', '--extra-body', json.dumps(extra, separators=(',', ':')),
            '--num-warmups', '0', '--save-result', '--save-detailed',
            '--result-dir', result_dir, '--result-filename', result_name + '.json']


def bench(run_dir, name: str, dataset: str, base_url: str, served_model: str, tokenizer: str,
          num_prompts: int = 200, output_len: int = 256, vllm_bin: str = 'vllm', dry_run: bool = False,
          modality: str = 'text', input_manifest=None) -> dict:
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('name must be a filename stem')
    if num_prompts <= 0 or output_len <= 0:
        raise ValueError('positive prompt count and output length required')
    available = validate_input(dataset, modality, input_manifest)
    if num_prompts > available:
        raise ValueError('num-prompts exceeds dataset rows; do not silently repeat cases')
    run = Path(run_dir)
    outputs, inputs = run/'outputs', run/'inputs'
    outputs.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    lock = run/'.active.lock'
    with lock.open('x') as f:
        f.write(name + '\n')
    try:
        result = outputs/(name+'.json')
        frozen = inputs/(name+'.materialized.jsonl')
        if result.exists() or frozen.exists():
            raise FileExistsError('bench input/result already exists; use a new run directory')
        shutil.copyfile(dataset, frozen)
        if input_manifest:
            shutil.copyfile(input_manifest, inputs/(name+'.manifest.json'))
        cmd = build_command(vllm_bin, base_url, served_model, tokenizer, str(frozen), num_prompts,
                            output_len, str(outputs), name, modality)
        (outputs/(name+'.cmd.txt')).write_text(shlex.join(cmd)+'\n')
        (outputs/(name+'.argv.json')).write_text(json.dumps(cmd, indent=2)+'\n')
        (inputs/(name+'.materialized.sha256')).write_text(sha256_file(frozen)+'  '+frozen.name+'\n')
        (inputs/(name+'.rows.txt')).write_text(str(available)+'\n')
        semantics = {'modality': modality, 'selected_rows': num_prompts, 'available_rows': available,
                     'output_budget': output_len, 'temperature': 0, 'request_seed': 0,
                     'dataset_seed': 0, 'request_rate': 'inf', 'client_max_concurrency': None,
                     'enable_thinking': False, 'ignore_eos': False,
                     'chat_template': 'server model template, once; no client rendering',
                     'messages': 'one user message; original text plus image for image input; no system prompt',
                     'num_warmups': 0, 'vllm_initial_test_request': 'first sample once, outside measured N',
                     'input_tokens': 'vLLM uses streaming usage.prompt_tokens when present; otherwise a text-only estimate. Verify per-request input_lens against server /tokenize before reporting total-token throughput.',
                     'not_accuracy_replay': True}
        (inputs/(name+'.semantics.json')).write_text(json.dumps(semantics, indent=2)+'\n')
        if dry_run:
            return {'command': cmd, 'dataset_rows': available}
        with (outputs/(name+'.bench.log')).open('wb') as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not result.exists():
            raise RuntimeError('vllm bench serve failed (exit %s); see %s' % (proc.returncode, outputs/(name+'.bench.log')))
        data = json.loads(result.read_text())
        keys = ['date', 'num_prompts', 'completed', 'failed', 'duration', 'request_throughput',
                'output_throughput', 'total_token_throughput', 'mean_ttft_ms', 'mean_tpot_ms',
                'total_input_tokens', 'total_output_tokens']
        summary = {k: data.get(k) for k in keys}
        summary['input_token_accounting'] = 'requires external usage/tokenize validation; see semantics'
        (outputs/(name+'.summary.json')).write_text(json.dumps(summary, indent=2)+'\n')
        (outputs/(name+'.json.sha256')).write_text(sha256_file(result)+'  '+result.name+'\n')
        if data.get('completed') != num_prompts or data.get('failed', 0):
            raise RuntimeError('benchmark contains failed or missing requests; raw result retained')
        return summary
    finally:
        lock.unlink()
