"""Disclosed replacement transport for MLAgentBench.LLM in a fresh workspace.

Original Agent/task/tool code remains upstream code. Select the original supported
OpenAI-chat protocol explicitly, with an actual local served model, never an alias
to the original default Claude. This file must not overwrite an upstream checkout.
"""

import os

from mlagentbench_local_chat import complete_original_chat

FAST_MODEL = None  # Set by the unchanged original runner from --fast-llm-name.


def complete_text(prompt, log_file, model, **kwargs):
    return complete_original_chat(
        prompt,
        model,
        api_base=os.environ["MLAGENTBENCH_LOCAL_API_BASE"],
        allowed_model=os.environ["MLAGENTBENCH_LOCAL_MODEL"],
        log_file=log_file,
        **kwargs,
    )


def complete_text_fast(prompt, **kwargs):
    return complete_text(prompt=prompt, model=FAST_MODEL, temperature=0.01, **kwargs)


def complete_text_claude(*_args, **_kwargs):
    # The original noninteractive selected task never uses this helper. Refuse
    # interactive Claude task naming rather than silently substituting a model.
    raise RuntimeError("Claude interactive task naming is not this local-chat protocol")
