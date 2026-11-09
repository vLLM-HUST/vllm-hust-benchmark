"""Opt-in FP32 GRU decomposition; no checkpoint, precision, or task changes.

Equations: https://docs.pytorch.org/docs/2.10/generated/torch.nn.GRU.html
This inference-only compatibility candidate is not a native synthesis receipt.
The CPU reference switch exists solely for numerical unit fixtures, not fallback.
"""

from types import MethodType


def forward_equations(module, inputs, hidden=None, *, require_npu=True):
    import torch
    from torch.nn import functional as functional

    if module.training:
        raise ValueError("inference-only GRU decomposition; training is unsupported")
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 3:
        raise ValueError("only original dense batched GRU input is supported")
    if require_npu and inputs.device.type != "npu":
        raise ValueError("NPU input required; CPU/CUDA fallback is forbidden")
    if inputs.dtype != torch.float32:
        raise ValueError("retain original FP32 GRU precision")
    if inputs.shape[-1] != module.input_size:
        raise ValueError("original input feature dimension mismatch")
    sequence = inputs.transpose(0, 1) if module.batch_first else inputs
    steps, batch, _ = sequence.shape
    if steps == 0:
        raise ValueError("empty GRU sequence is unsupported")
    directions = 2 if module.bidirectional else 1
    expected = (module.num_layers * directions, batch, module.hidden_size)
    if hidden is None:
        hidden = inputs.new_zeros(expected)
    if (
        tuple(hidden.shape) != expected
        or hidden.device != inputs.device
        or hidden.dtype != inputs.dtype
    ):
        raise ValueError("original hidden-state shape, device or dtype mismatch")
    for parameter in module.parameters():
        if parameter.device != inputs.device or parameter.dtype != inputs.dtype:
            raise ValueError(
                "all original GRU parameters must retain input device/dtype"
            )
    final_states = []
    for layer in range(module.num_layers):
        outputs = []
        for direction in range(directions):
            suffix = f"_l{layer}" + ("_reverse" if direction else "")
            weight_input = getattr(module, "weight_ih" + suffix)
            weight_hidden = getattr(module, "weight_hh" + suffix)
            bias_input = getattr(module, "bias_ih" + suffix, None)
            bias_hidden = getattr(module, "bias_hh" + suffix, None)
            state = hidden[layer * directions + direction]
            values = [None] * steps
            indices = range(steps - 1, -1, -1) if direction else range(steps)
            for index in indices:
                input_gates = functional.linear(
                    sequence[index], weight_input, bias_input
                )
                hidden_gates = functional.linear(state, weight_hidden, bias_hidden)
                input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
                hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
                reset = torch.sigmoid(input_reset + hidden_reset)
                update = torch.sigmoid(input_update + hidden_update)
                # PyTorch applies reset AFTER hidden affine, including its bias.
                candidate = torch.tanh(input_new + reset * hidden_new)
                state = candidate + update * (state - candidate)
                values[index] = state
            outputs.append(torch.stack(values, dim=0))
            final_states.append(state)
        sequence = torch.cat(outputs, dim=-1) if directions == 2 else outputs[0]
    output = sequence.transpose(0, 1) if module.batch_first else sequence
    return output, torch.stack(final_states, dim=0)


def install_npu_gru_forward(module):
    """Change only this GRU instance's forward; retain parameter objects and keys."""
    import torch

    if not isinstance(module, torch.nn.GRU):
        raise TypeError("explicit original torch.nn.GRU instance required")
    if module.training:
        raise ValueError("call original model.eval() before installing compatibility")
    if "forward" in module.__dict__:
        raise ValueError("refuse to replace an already customized GRU forward")

    def npu_forward(self, inputs, hx=None):
        return forward_equations(self, inputs, hx, require_npu=True)

    module.forward = MethodType(npu_forward, module)
    return module
