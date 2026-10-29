import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "retrieval_audit", ROOT / "scripts/audit_fixed018_retrieval.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def stream(delta, done=True):
    event = dict(
        choices=[dict(delta=delta, finish_reason="stop")],
        usage=dict(prompt_tokens=16384, completion_tokens=2),
    )
    value = b"data: " + json.dumps(event).encode() + b"\n\n"
    if done:
        value += b"data: [DONE]\n\n"
    return gzip.compress(value)


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_both_wire_fields_preserve_answer_text(field):
    answer, thought, finish, usage = AUDIT.reconstruct(
        stream({"content": " 123456 ", field: "reason"})
    )
    assert answer == " 123456 " and thought == "reason" and finish == "stop"
    assert usage["prompt_tokens"] == 16384


def test_conflicting_reasoning_is_not_silently_selected():
    with pytest.raises(ValueError, match="Conflicting"):
        AUDIT.reconstruct(
            stream(dict(content="123456", reasoning="a", reasoning_content="b"))
        )


def test_incomplete_stream_is_not_a_pass():
    with pytest.raises(ValueError, match="Incomplete"):
        AUDIT.reconstruct(stream(dict(content="123456"), done=False))


def test_development_stack_cannot_be_b0():
    raw = {
        "receipt.json": json.dumps(dict(status="failed")).encode(),
        "runtime-identity.json": json.dumps(
            dict(role="B0", vllm_version="0.28.1", ascend_version="0.25.1")
        ).encode(),
    }
    raw["SHA256SUMS"] = "\n".join(
        hashlib.sha256(v).hexdigest() + "  " + k for k, v in raw.items()
    ).encode()
    with pytest.raises(ValueError, match="fixed018"):
        AUDIT.arm({"run/" + k: v for k, v in raw.items()}, "run", "B0")


def test_published_capsule_recomputes():
    directory = ROOT / "reports/long-context-fixed018-paired-20261029"
    result = AUDIT.audit(directory)
    assert result == json.loads((directory / "summary.json").read_text())
    assert result["arms"][0]["runtime"]["vllm_version"] == "0.18.0"
    assert result["same_request_and_resource_pair"] is True
    assert [arm["correct"] for arm in result["arms"]] == [6, 6]
    assert all(arm["all_reserved_devices_released"] for arm in result["arms"])
    assert result["corrected_b1_activation"]["request_scoped_recompute_admissions"] == 7
    assert result["prior_failed_startup"]["status"] == "failed"
    assert result["continuous_stability_pass"] is None
