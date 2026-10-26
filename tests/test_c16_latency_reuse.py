import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "reports/c16-request-latency-reuse-20261026/derive.py"
SPEC = importlib.util.spec_from_file_location("c16_latency", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample():
    return dict(
        success=True,
        start=1,
        first_token=3,
        last_token=9,
        end=10,
        ttft_seconds=2,
        token_ids=[1, 2, 3, 4],
        chunks=[[3, 1], [9, 3]],
        prompt_tokens=17000,
    )


def test_equal_request_not_token_weighting():
    a = sample()
    b = sample()
    b.update(token_ids=[1, 2], chunks=[[3, 1], [9, 1]])
    result = MODULE.summarize({"duration": 900}, [a, b])
    assert result["tpot"]["mean_seconds"] == 4
    assert result["actual_ge16k_requests"] == 2


def test_drain_failure_and_undefined_separate():
    a = sample()
    b = sample()
    b["end"] = 901
    c = sample()
    c.update(last_token=3, token_ids=[1], chunks=[[3, 1]])
    failed = dict(success=False, start=0, end=1)
    result = MODULE.summarize({"duration": 900}, [a, b, c, failed])
    assert result["requests"] == 2
    assert result["undefined_tpot"] == 1
    assert result["tpot"]["n"] == 1
    assert MODULE.summarize({"duration": 900}, [a, b, c, failed], True)["requests"] == 3


@pytest.mark.parametrize(
    "field,value", [("ttft_seconds", 3), ("last_token", 11), ("token_ids", [1])]
)
def test_inconsistent_raw_values_rejected(field, value):
    row = copy.deepcopy(sample())
    row[field] = value
    with pytest.raises(AssertionError):
        MODULE.summarize({"duration": 900}, [row])


def test_empty_is_unknown_not_zero():
    result = MODULE.summarize({"duration": 900}, [])
    assert result["ttft"]["mean_seconds"] is None
    assert result["tpot"]["p99_seconds"] is None


def test_archived_result_is_reproducible():
    result = MODULE.build()
    assert result == json.loads((PATH.parent / "summary.json").read_text())
    assert [x["window"]["requests"] for x in result["arms"]] == [392, 1280]
    assert [x["including_drain"]["requests"] for x in result["arms"]] == [408, 1296]
    assert result["new_inference_requests"] == 0
