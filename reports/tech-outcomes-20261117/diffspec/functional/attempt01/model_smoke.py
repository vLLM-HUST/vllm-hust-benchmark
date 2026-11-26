"""Two actual TP4 model requests; no performance claim is made from this run."""
import dataclasses
import hashlib
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parent
BASE = Path('/root/diffspec-experiments')
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    from vllm import LLM, SamplingParams

    old = BASE / 'runs/diffspec-results-20261030/eval'
    options = json.loads((old / 'configs/diffspec.json').read_text())
    views = BASE / 'models-and-views/qwen35-moe-diagnosis-20261030'
    options.update(model=str(views / 'target-model'), max_model_len=8192,
                   max_num_seqs=2, worker_extension_cls='trace_worker.TraceWorker')
    options['speculative_config']['model'] = str(views / 'draft-matched')
    options['compilation_config']['cudagraph_capture_sizes'] = [4, 8, 12]
    save('config.json', options)
    original = json.loads((old / 'inputs.json').read_text())['scenarios'][0]['cases']
    cases = []
    for index, length in enumerate([4096, 4352]):
        ids = original[index]['prompt_token_ids']
        ids = ids[:length - 256] + ids[-256:]
        cases.append({'case_id': f'real-request-{index}', 'prompt_token_ids': ids,
                      'prompt_tokens': len(ids), 'max_tokens': 256,
                      'prompt_sha256': hashlib.sha256(
                          json.dumps(ids, separators=(',', ':')).encode()).hexdigest()})
    save('inputs.json', cases)
    llm = None
    try:
        llm = LLM(**options)
        save('loaded-workers.json', llm.collective_rpc('install_trace'))
        responses = llm.generate(
            [{'prompt_token_ids': case['prompt_token_ids']} for case in cases],
            SamplingParams(temperature=0, max_tokens=256, ignore_eos=True, seed=17),
            use_tqdm=False)
        outputs = []
        for case, response in zip(cases, responses):
            answer = response.outputs[0]
            outputs.append({'case_id': case['case_id'], 'request_id': response.request_id,
                'prompt_tokens': len(response.prompt_token_ids), 'text': answer.text,
                'token_ids': answer.token_ids, 'finish_reason': answer.finish_reason,
                'spec_decode_metrics': answer.spec_decode_metrics.to_dict()
                    if answer.spec_decode_metrics else None,
                'metrics': dataclasses.asdict(response.metrics) if response.metrics else None})
        save('outputs.json', outputs)
        save('workers-after.json', llm.collective_rpc('trace_status'))
        assert len(outputs) == 2 and all(len(row['token_ids']) == 256 for row in outputs)
        save('model-completion.json', {'passed': True, 'actual_model_requests': 2,
             'output_tokens': 512, 'performance_measurement': False})
    except BaseException as error:
        save('failure.json', {'type': type(error).__name__, 'message': str(error),
                              'traceback': traceback.format_exc()})
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
