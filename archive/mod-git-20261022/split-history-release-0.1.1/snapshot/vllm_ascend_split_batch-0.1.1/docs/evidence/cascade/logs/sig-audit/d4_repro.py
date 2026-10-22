"""Decisive repro for D4: plugin's _model_forward override vs the host's
keyword-only call site.

Host (gpu_model_runner.py:4550) calls:
    self._model_forward(input_ids=..., positions=...,
                        intermediate_tensors=..., inputs_embeds=..., **model_kwargs)
Plugin override (cascade_runner_patch.py:274) declares a REQUIRED leading
positional `num_tokens_padded`, which that call never supplies.
"""
import inspect


class FakeRunner:
    def _model_forward(self, input_ids=None, positions=None,
                       intermediate_tensors=None, inputs_embeds=None,
                       **model_kwargs):
        return "host-original"


# verbatim copy of the plugin override's parameter list (defaults included)
def plugin_model_forward(self, num_tokens_padded, input_ids=None, positions=None,
                         intermediate_tensors=None, inputs_embeds=None,
                         **model_kwargs):
    return "plugin"


r = FakeRunner()
r._model_forward = plugin_model_forward.__get__(r, FakeRunner)

print("plugin signature:", inspect.signature(plugin_model_forward))
try:
    r._model_forward(input_ids=None, positions=None,
                     intermediate_tensors=None, inputs_embeds=None)
    print("D4: NOT REPRODUCED (call succeeded)")
except TypeError as exc:
    print(f"D4: REPRODUCED -> TypeError: {exc}")
