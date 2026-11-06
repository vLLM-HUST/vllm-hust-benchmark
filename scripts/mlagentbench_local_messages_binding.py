"""Explicit local Messages binding; retain original Agent and scientific tools."""

import os

from mlagentbench_local_messages import complete_original_messages, queue_opener

FAST_MODEL = None


def complete_text(prompt, log_file, model, **kwargs):
    return complete_original_messages(
        prompt,
        model,
        api_base=os.environ["MLAGENTBENCH_LOCAL_API_BASE"],
        allowed_model=os.environ["MLAGENTBENCH_LOCAL_MODEL"],
        log_file=log_file,
        opener=queue_opener(os.environ["MLAGENTBENCH_MESSAGES_QUEUE"]),
        **kwargs,
    )


def complete_text_fast(prompt, **kwargs):
    return complete_text(prompt=prompt, model=FAST_MODEL, temperature=0.01, **kwargs)


def complete_text_claude(*_args, **_kwargs):
    raise RuntimeError(
        "Interactive Claude task naming is not this actual-model protocol"
    )
