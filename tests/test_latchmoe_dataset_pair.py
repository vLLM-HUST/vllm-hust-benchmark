"""Contract/scorer fixtures only: these are not model benchmark results."""

import copy
import json

import pytest

from vllm_hust_benchmark import latchmoe_dataset_pair as m


def pair(tmp_path):
    common = {
        "model": {
            "id": "Qwen/Qwen3.5-35B-A3B",
            "revision": "a" * 40,
            "config_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "chat_template_sha256": "d" * 64,
            "dtype": "bfloat16",
        },
        "runtime": {
            "core_revision": "e" * 40,
            "seam_revision": "f" * 40,
            "mod_wheel_sha256": "a" * 64,
            "packages": {"torch": "test"},
        },
        "hardware": {"chip_model": "910B2", "device_ids": [0, 1], "tp": 2, "nodes": 1},
        "server": {
            "graph_mode": "PIECEWISE",
            "prefix_caching": False,
            "cpu_offload_gb": 0,
            "quantization": None,
            "argv": ["vllm", "serve"],
        },
        "evaluator": {"revision": "b" * 40, "sha256": "e" * 64, "metric": "accuracy"},
        "scaffold": {"id": "no-tools-v1", "sha256": "f" * 64},
        "budgets": {"max_tokens": 32, "concurrency": 1, "tools": 0},
        "dataset": {
            "source": "TIGER-Lab/MMLU-Pro",
            "provider": "huggingface",
            "track": None,
            "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",  # pragma: allowlist secret
            "manifest_sha256": "b" * 64,
            "scope": "fixture-only",
        },
    }
    request = {
        "task_id": "fixture-task",
        "prompt_sha256": "c" * 64,
        "parameters_sha256": "d" * 64,
    }
    contract = {
        "schema_version": m.SCHEMA,
        "dataset": "MMLU-Pro",
        "track": None,
        "common": common,
        "requests": [request],
        "mod_config": {"num_slots": 16, "indexed_mode": "graph", "offload_gb": 14},
    }
    roots, arms = {}, {}
    for label in ("B0", "B1"):
        root = tmp_path / label
        root.mkdir(parents=True)
        roots[label] = root
        raw = [{**request, "success": False, "error": "fixture failure"}]
        telemetry = {
            "phase": "measured_requests_only",
            "worker_identity": "fixture-worker",
            "request_manifest_sha256": m.digest([request]),
            "before": {
                "calls": 10 if label == "B1" else 0,
                "waves": 20 if label == "B1" else 0,
            },
            "after": {
                "calls": 11 if label == "B1" else 0,
                "waves": 21 if label == "B1" else 0,
            },
        }
        (root / "outcomes.json").write_text(json.dumps(raw))
        (root / "telemetry.json").write_text(json.dumps(telemetry))
        artifacts = {
            name: m.file_digest(root / name)
            for name in ("outcomes.json", "telemetry.json")
        }
        arms[label] = {
            "arm": label,
            "contract_sha256": m.digest(contract),
            "common": copy.deepcopy(common),
            "requests": [request],
            "effective_mod": {
                "enabled": label == "B1",
                "num_slots": 16,
                "indexed_mode": "graph",
                "offload_gb": 14 if label == "B1" else 0,
            },
            "artifacts": artifacts,
            "outcomes_path": "outcomes.json",
            "telemetry_path": "telemetry.json",
        }
    return contract, arms, roots


def rewrite(arm, root, name, value):
    (root / name).write_text(json.dumps(value))
    arm["artifacts"][name] = m.file_digest(root / name)


def test_matched_fixture_is_not_formal_acceptance_or_model_score(tmp_path):
    contract, arms, roots = pair(tmp_path)
    verdict = m.verify_pair(contract, arms, roots)
    assert verdict["status"] == "matched-evidence"
    assert verdict["mod_exercised"] is True
    assert verdict["formal_v5_4_accepted"] is False
    assert verdict["quality_scores"] is None
    assert verdict["failures"] == {"B0": 1, "B1": 1}


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "runtime",
        "hardware",
        "server",
        "evaluator",
        "scaffold",
        "budgets",
        "dataset",
    ],
)
def test_reject_changed_common_contract(tmp_path, field):
    contract, arms, roots = pair(tmp_path)
    arms["B1"]["common"][field]["extra"] = "changed"
    with pytest.raises(ValueError, match="common"):
        m.verify_pair(contract, arms, roots)


def test_arm_label_does_not_enable_mod(tmp_path):
    contract, arms, roots = pair(tmp_path)
    arms["B1"]["effective_mod"]["enabled"] = False
    with pytest.raises(ValueError, match="activation"):
        m.verify_pair(contract, arms, roots)


def test_no_measured_path_is_not_exercised_despite_warmup_counts(tmp_path):
    contract, arms, roots = pair(tmp_path)
    telemetry = json.loads((roots["B1"] / "telemetry.json").read_text())
    telemetry["after"] = dict(telemetry["before"])
    rewrite(arms["B1"], roots["B1"], "telemetry.json", telemetry)
    verdict = m.verify_pair(contract, arms, roots)
    assert verdict["status"] == "not-exercised"
    assert verdict["mod_exercised"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "baseline_calls",
        "negative",
        "warmup_phase",
        "wrong_requests",
        "waves_without_calls",
    ],
)
def test_reject_invalid_measurement_telemetry(tmp_path, mutation):
    contract, arms, roots = pair(tmp_path)
    label = "B0" if mutation == "baseline_calls" else "B1"
    telemetry = json.loads((roots[label] / "telemetry.json").read_text())
    if mutation == "baseline_calls":
        telemetry["after"]["calls"] += 1
    elif mutation == "negative":
        telemetry["after"]["calls"] = 0
    elif mutation == "warmup_phase":
        telemetry["phase"] = "whole_process_including_warmup"
    elif mutation == "wrong_requests":
        telemetry["request_manifest_sha256"] = "0" * 64
    else:
        telemetry["after"]["calls"] = telemetry["before"]["calls"]
    rewrite(arms[label], roots[label], "telemetry.json", telemetry)
    with pytest.raises(ValueError, match="telemetry|baseline"):
        m.verify_pair(contract, arms, roots)


def test_tampered_raw_artifact_rejected(tmp_path):
    contract, arms, roots = pair(tmp_path)
    (roots["B1"] / "outcomes.json").write_text("[]")
    with pytest.raises(ValueError, match="SHA"):
        m.verify_pair(contract, arms, roots)


def test_path_escape_and_missing_arm_rejected(tmp_path):
    contract, arms, roots = pair(tmp_path)
    with pytest.raises(ValueError, match="B0.*B1"):
        m.verify_pair(contract, {"B1": arms["B1"]}, roots)
    arms["B1"]["artifacts"]["../outside"] = "a" * 64
    with pytest.raises(ValueError, match="path"):
        m.verify_pair(contract, arms, roots)


def test_request_budget_missing_or_duplicate_outcome_rejected(tmp_path):
    contract, arms, roots = pair(tmp_path)
    arms["B1"]["requests"][0] = {
        "task_id": "fixture-task",
        "prompt_sha256": "c" * 64,
        "parameters_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="requests"):
        m.verify_pair(contract, arms, roots)
    contract, arms, roots = pair(tmp_path / "next")
    raw = json.loads((roots["B1"] / "outcomes.json").read_text())
    rewrite(arms["B1"], roots["B1"], "outcomes.json", raw + raw)
    with pytest.raises(ValueError, match="outcomes"):
        m.verify_pair(contract, arms, roots)


@pytest.mark.parametrize(
    "dataset,track",
    [
        ("MMLU", None),
        ("SWE-bench-Verified", None),
        ("Terminal-Bench 2.0", None),
        ("FrontierScience", None),
    ],
)
def test_no_similar_dataset_or_combined_frontier_track(tmp_path, dataset, track):
    contract, arms, roots = pair(tmp_path)
    contract.update(dataset=dataset, track=track)
    with pytest.raises(ValueError, match="dataset|track"):
        m.verify_pair(contract, arms, roots)


def test_mmlu_failures_remain_in_denominator_without_compression_eligibility():
    cases = [
        {
            "case_id": name,
            "prompt_sha256": "hash",
            "prompt_tokens": n,
            "max_tokens": 32,
            "ignore_eos": False,
            "category": "math",
            "answer": "B",
            "option_count": 2,
        }
        for name, n in [("short", 20), ("long", 5000), ("failed", 30)]
    ]
    results = [
        {**case, "output": "answer is B", "success": case["case_id"] != "failed"}
        for case in cases
    ]
    score = m.score_mmlu(results, cases)
    assert score["count"] == 3
    assert score["correct"] == 2
    assert score["accuracy"] == 2 / 3
    assert score["transport_failures"] == 1
    assert "eligible" not in score and "ineligible" not in score
    assert all("eligible" not in row for row in score["outcomes"])


@pytest.mark.parametrize(
    "dataset,track", [("Terminal-Bench 2.1", None), ("FrontierScience", "research")]
)
def test_dataset_label_cannot_relabel_another_source(tmp_path, dataset, track):
    contract, arms, roots = pair(tmp_path)
    contract.update(dataset=dataset, track=track)
    for arm in arms.values():
        arm["contract_sha256"] = m.digest(contract)
    with pytest.raises(ValueError, match="source|track"):
        m.verify_pair(contract, arms, roots)


def test_raw_outcomes_cannot_contradict_request_manifest(tmp_path):
    contract, arms, roots = pair(tmp_path)
    raw = json.loads((roots["B1"] / "outcomes.json").read_text())
    raw[0].update(prompt_sha256="0" * 64, parameters_sha256="0" * 64)
    rewrite(arms["B1"], roots["B1"], "outcomes.json", raw)
    with pytest.raises(ValueError, match="request"):
        m.verify_pair(contract, arms, roots)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("runtime", "packages", None),
        ("budgets", "max_tokens", None),
        ("budgets", "tools", True),
        ("server", "prefix_caching", 0),
        ("hardware", "tp", True),
    ],
)
def test_invalid_common_values_do_not_freeze_unspecified_contract(
    tmp_path, section, key, value
):
    contract, arms, roots = pair(tmp_path)
    contract["common"][section][key] = value
    for arm in arms.values():
        arm["common"] = copy.deepcopy(contract["common"])
        arm["contract_sha256"] = m.digest(contract)
    with pytest.raises(ValueError):
        m.verify_pair(contract, arms, roots)


def test_mmlu_boolean_success_is_required():
    case = {
        "case_id": "fixture",
        "prompt_sha256": "hash",
        "prompt_tokens": 20,
        "max_tokens": 32,
        "ignore_eos": False,
        "category": "math",
        "answer": "B",
        "option_count": 2,
    }
    result = {**case, "success": "false", "error": "failure", "output": "answer is B"}
    with pytest.raises(ValueError, match="boolean"):
        m.score_mmlu([result], [case])


def test_hash_and_parse_cannot_consume_different_evidence_bytes(tmp_path, monkeypatch):
    contract, arms, roots = pair(tmp_path)
    telemetry = json.loads((roots["B1"] / "telemetry.json").read_text())
    telemetry["after"] = dict(telemetry["before"])
    rewrite(arms["B1"], roots["B1"], "telemetry.json", telemetry)
    original_digest = m.file_digest

    def mutate_after_hash(path):
        result = original_digest(path)
        if path == roots["B1"] / "telemetry.json":
            changed = copy.deepcopy(telemetry)
            for key in ("calls", "waves"):
                changed["after"][key] += 1
            path.write_text(json.dumps(changed))
        return result

    monkeypatch.setattr(m, "file_digest", mutate_after_hash)
    assert m.verify_pair(contract, arms, roots)["status"] == "not-exercised"


@pytest.mark.parametrize(
    "dataset,track",
    [
        ("HLE-Verified", None),
        ("SWE-bench-Pro", None),
        ("Terminal-Bench 2.1", None),
        ("FrontierScience", "olympiad"),
        ("FrontierScience", "research"),
    ],
)
def test_each_pinned_source_has_a_distinct_contract(tmp_path, dataset, track):
    contract, arms, roots = pair(tmp_path)
    contract.update(dataset=dataset, track=track)
    provider, source, revision = m.SOURCE_IDENTITIES[dataset]
    contract["common"]["dataset"].update(
        provider=provider, source=source, revision=revision, track=track
    )
    if dataset == "FrontierScience":
        contract["common"]["dataset"]["source_sha256"] = m.FRONTIER_SOURCES[track][1]
    for arm in arms.values():
        arm["common"] = copy.deepcopy(contract["common"])
        arm["contract_sha256"] = m.digest(contract)
    assert m.verify_pair(contract, arms, roots)["quality_scores"] is None
    if dataset == "FrontierScience":
        contract["common"]["dataset"]["source_sha256"] = m.FRONTIER_SOURCES[
            "research" if track == "olympiad" else "olympiad"
        ][1]
        with pytest.raises(ValueError, match="track"):
            m.verify_pair(contract, arms, roots)


def test_boolean_common_value_is_not_equivalent_to_integer_receipt(tmp_path):
    contract, arms, roots = pair(tmp_path)
    arms["B1"]["common"]["server"]["prefix_caching"] = 0
    with pytest.raises(ValueError, match="common"):
        m.verify_pair(contract, arms, roots)


def test_effective_mod_boolean_is_not_equivalent_to_numeric_control(tmp_path):
    contract, arms, roots = pair(tmp_path)
    contract["mod_config"].update(num_slots=1, offload_gb=1)
    for label, arm in arms.items():
        arm["contract_sha256"] = m.digest(contract)
        arm["effective_mod"].update(num_slots=True, offload_gb=label == "B1")
    with pytest.raises(ValueError, match="activation"):
        m.verify_pair(contract, arms, roots)
