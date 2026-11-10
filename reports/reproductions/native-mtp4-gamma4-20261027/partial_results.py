"""Persist completed requests before a later timeout can lose the whole batch."""
import json

def install(engine, path, rows, prompts, serializer, reject_degenerate_token0=False):
    by_prompt = {tuple(p['prompt_token_ids']):r for r,p in zip(rows,prompts,strict=True)}
    original = engine.step
    seen = set()
    path.touch(exist_ok=False)
    def step(*args, **kwargs):
        outputs = original(*args, **kwargs)
        for output in outputs:
            if not output.finished or output.request_id in seen:
                continue
            row = by_prompt[tuple(output.prompt_token_ids)]
            completion = output.outputs[0]
            record = {'question_id':row['question_id'], 'request_id':output.request_id,
                      'text':completion.text,'output_tokens':list(completion.token_ids),
                      'finish_reason':completion.finish_reason,'metrics':output.metrics}
            with path.open('a') as stream:
                stream.write(json.dumps(record,default=serializer)+'\n')
            seen.add(output.request_id)
            if reject_degenerate_token0 and len(completion.token_ids) >= 16 and sum(t == 0 for t in completion.token_ids) / len(completion.token_ids) > .9:
                raise RuntimeError(f'Degenerate token-0 output for question {row["question_id"]}; completed output preserved, stop correctness gate')
        return outputs
    engine.step = step
