"""Adapt the pinned Qwen3VL reset signature to its author's single-task runner.

The runner supplies vm_ip, but this model-only Agent's reset accepts only a
logger. Preserve the original reset implementation and every other Agent method.
This disclosed compatibility adapter does not change tasks, prompts or scoring.
"""

import inspect


def reset_compatible_agent(original):
    parameters = list(inspect.signature(original.reset).parameters.values())
    if (
        [parameter.name for parameter in parameters] != ["self", "_logger"]
        or parameters[1].default is not None
        or parameters[1].kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
    ):
        raise ValueError("expected pinned Qwen3VL reset(self, _logger=None) signature")

    class ResetCompatibleAgent(original):
        def reset(self, _logger=None, *, vm_ip=None):
            # Qwen3VL reset only resets its logger/history. Unlike remote
            # computer-use agents, it has no VM-IP state to update here.
            return super().reset(_logger)

    return ResetCompatibleAgent
