"""Synthetic CPU equation controls only; no task/checkpoint/NPU result implied."""

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "dailytalk_gru_npu",
    Path(__file__).resolve().parents[1] / "scripts/dailytalk_gru_npu.py",
)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_equations_match_pytorch_fp32_dense_eval(
    layers, bidirectional, batch_first, bias
):
    torch.manual_seed(113)
    model = torch.nn.GRU(
        5,
        7,
        layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
        bias=bias,
        dropout=0.2 if layers > 1 else 0,
    ).eval()
    inputs = torch.randn((3, 10, 5) if batch_first else (10, 3, 5))
    hidden = torch.randn(layers * (2 if bidirectional else 1), 3, 7)
    before = {k: value.clone() for k, value in model.state_dict().items()}
    with torch.inference_mode():
        expected = model(inputs, hidden)
        actual = entry.forward_equations(model, inputs, hidden, require_npu=False)
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, atol=1e-6, rtol=1e-5)
    assert all(
        torch.equal(value, before[key]) for key, value in model.state_dict().items()
    )


def test_default_rejects_cpu_and_install_preserves_original_parameters():
    model = torch.nn.GRU(5, 7, batch_first=True).eval()
    original = {key: id(value) for key, value in model.named_parameters()}
    entry.install_npu_gru_forward(model)
    assert {key: id(value) for key, value in model.named_parameters()} == original
    with pytest.raises(ValueError, match="NPU input required"):
        model(torch.zeros(2, 10, 5))
    with pytest.raises(ValueError, match="already customized"):
        entry.install_npu_gru_forward(model)


def test_reject_training_and_precision_change():
    model = torch.nn.GRU(5, 7)
    with pytest.raises(ValueError, match="training"):
        entry.forward_equations(model, torch.zeros(10, 2, 5), require_npu=False)
    model.eval()
    with pytest.raises(ValueError, match="FP32"):
        entry.forward_equations(model, torch.zeros(10, 2, 5).half(), require_npu=False)


@pytest.mark.parametrize("steps", [1, 10])
def test_publisher_guo_dimensions_and_default_zero_hidden(steps):
    torch.manual_seed(113)
    model = torch.nn.GRU(
        128, 128, 2, batch_first=True, bidirectional=True, dropout=0.2
    ).eval()
    inputs = torch.randn(3, steps, 128)
    with torch.inference_mode():
        expected = model(inputs)
        actual = entry.forward_equations(model, inputs, require_npu=False)
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, atol=1e-6, rtol=1e-5)
