from copy import deepcopy

import pytest

from vllm_hust_benchmark.bio_pair import order_cases, score_difference, validate_pairs


def fixture():
    spec = {"option": "cache", "pairs": {"fixed": ["a", "b"]}}
    cases = [
        {
            "id": key,
            "profile_id": "p",
            "samples": [{"start": 1}],
            "options": {"cache": on, "eos": True},
        }
        for key, on in [("a", False), ("b", True)]
    ]
    return spec, cases


def test_counterbalance():
    spec, cases = fixture()
    assert [c["id"] for c in order_cases(spec, cases, 0)[0]] == ["a", "b"]
    assert [c["id"] for c in order_cases(spec, cases, 1)[0]] == ["b", "a"]


def test_reject_changed_sample_or_option():
    spec, cases = fixture()
    for change in [
        {"samples": [{"start": 0}]},
        {"options": {"cache": True, "eos": False}},
    ]:
        second = deepcopy(cases)
        second[1].update(change)
        with pytest.raises(ValueError, match="differ"):
            validate_pairs(spec, second)


def test_reject_mismatched_output_axes():
    a = [{"score_mask": [True], "target_token_ids": [7], "token_logprobs": [-2.0]}]
    b = deepcopy(a)
    b[0]["target_token_ids"] = [4]
    with pytest.raises(ValueError, match="targets"):
        score_difference(a, b)
