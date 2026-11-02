import gzip
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("audit_fixed018_continuous")


def wire(done=True, reverse=False):
    event = dict(
        choices=[dict(delta=dict(reasoning_content="x"), finish_reason="length")],
        usage=dict(prompt_tokens=16407, completion_tokens=256),
    )
    lines = [dict(clock=dict(monotonic=2), raw="data: " + json.dumps(event))]
    if done:
        lines.append(
            dict(clock=dict(monotonic=1 if reverse else 3), raw="data: [DONE]")
        )
    return gzip.compress("\n".join(json.dumps(x) for x in lines).encode())


def test_nearest_rank_keeps_small_sample_max(audit):
    assert audit.quantile([1, 3, 2], 0.99) == 3
    assert audit.quantile([], 0.95) is None


def test_rebuild_from_raw_wire(audit):
    first, last, usage, finish = audit.reconstruct(wire())
    assert first == last == 2 and finish == "length"
    assert usage["completion_tokens"] == 256


def test_no_done_is_not_success(audit):
    with pytest.raises(ValueError, match="Incomplete"):
        audit.reconstruct(wire(done=False))


def test_reversed_time_is_not_success(audit):
    with pytest.raises(ValueError, match="Non-monotonic"):
        audit.reconstruct(wire(reverse=True))


def test_utilization_time_weighted_not_sample_mean(audit):
    samples = [
        dict(time=t, values={"5": v, "6": v}) for t, v in [(0, 10), (1, 90), (10, 0)]
    ]
    value = audit.utilization(samples, 0, 10)
    assert value["mean_percent"] == 82
    assert value["missing_seconds"] == 0


def test_missing_samples_are_not_zero_filled(audit):
    samples = [
        dict(time=0, values={"5": 10, "6": 10}),
        dict(time=1, values=None),
        dict(time=5, values={"5": 90, "6": 90}),
    ]
    assert audit.utilization(samples, 0, 5)["mean_percent"] is None
    assert audit.utilization(samples, 0, 5)["missing_seconds"] == 4
