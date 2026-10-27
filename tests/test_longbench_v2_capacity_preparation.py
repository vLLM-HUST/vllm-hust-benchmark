"""Software-only fixtures; no original data, native model or scoring evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runner_module():
    spec = importlib.util.spec_from_file_location(
        "longbench_v2_capacity",
        ROOT / "scripts/prepare_longbench_v2_original_capacity.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_official_module(temperature=0.1):
    """Exercise the boundary protocol, not the upstream benchmark implementation."""

    def forbidden(*args, **kwargs):
        raise AssertionError("software fixture must not reach original inference")

    module = SimpleNamespace(
        query_llm=forbidden,
        OpenAI=forbidden,
        AutoTokenizer=forbidden,
        URL="SOFTWARE_FIXTURE_NO_NETWORK",
        calls=0,
    )

    def get_pred(items, args, fout):
        assert fout is None
        (item,) = items
        tokenizer = module.AutoTokenizer.from_pretrained(
            "SOFTWARE_TOKENIZER_PATH", trust_remote_code=True
        )
        client = module.OpenAI(
            base_url=module.URL,
            api_key="SOFTWARE_FIXTURE",  # pragma: allowlist secret - software-only dummy key
        )
        module.calls += 1
        try:
            module.query_llm(
                item["software_prompt"],
                args.model,
                tokenizer,
                client,
                temperature=temperature,
                max_new_tokens=128,
            )
        except Exception:
            module.calls += 1
            raise
        raise AssertionError("no fabricated model response or answer may be returned")

    module.get_pred = get_pred
    return module


def test_prompt_capture_stops_before_generation_without_retry_and_restores_module():
    runner = runner_module()
    official = fake_official_module()
    before = official.query_llm, official.OpenAI, official.AutoTokenizer
    item = {"software_prompt": "SOFTWARE FIXTURE NOT A DATASET REQUEST"}
    original = dict(item)
    tokenizer = object()
    prompt = runner.capture_original_prompt(
        official, item, "fixture-local-model", "SOFTWARE_TOKENIZER_PATH", tokenizer
    )
    assert prompt == item["software_prompt"]
    assert official.calls == 1
    assert item == original
    assert before == (official.query_llm, official.OpenAI, official.AutoTokenizer)


def test_changed_query_contract_fails_closed_and_restores_module():
    runner = runner_module()
    official = fake_official_module(temperature=0.2)
    before = official.query_llm, official.OpenAI, official.AutoTokenizer
    with pytest.raises(ValueError, match="contract changed"):
        runner.capture_original_prompt(
            official,
            {"software_prompt": "SOFTWARE FIXTURE"},
            "fixture-local-model",
            "SOFTWARE_TOKENIZER_PATH",
            object(),
        )
    assert before == (official.query_llm, official.OpenAI, official.AutoTokenizer)


def test_no_captured_prompt_is_not_success():
    runner = runner_module()
    official = fake_official_module()
    official.get_pred = lambda *args: None
    with pytest.raises(ValueError, match="exactly once"):
        runner.capture_original_prompt(
            official, {}, "fixture-local-model", "SOFTWARE_TOKENIZER_PATH", object()
        )


@pytest.mark.parametrize("indices", [[], [1, 1], [4, 1], [-1], [503], [True]])
def test_original_selection_never_filters_or_reorders(indices):
    with pytest.raises(ValueError):
        runner_module().original_indices(indices)


def test_explicit_and_full_indices_are_fixed_before_capacity_checks():
    runner = runner_module()
    indices = [0, 1, 2]
    assert runner.original_indices(indices) is indices
    assert runner.original_indices(full=True) == list(range(503))
    with pytest.raises(ValueError, match="combined"):
        runner.original_indices(indices, full=True)


@pytest.mark.parametrize(
    "count,status",
    [(130944, "FULL_INPUT_FITS"), (130945, "RESOURCE_LIMIT_FULL_CONTEXT")],
)
def test_full_chat_budget_marks_overflow_without_truncation(count, status):
    runner = runner_module()
    calls = []

    def apply(messages, **kwargs):
        calls.append((messages, kwargs))
        return [1] * count

    prompt = "SOFTWARE FIXTURE NOT ORIGINAL DATA"
    row = runner.capacity_row(
        {"_id": "software-fixture", "length": "software-fixture"},
        0,
        prompt,
        SimpleNamespace(apply_chat_template=apply),
    )
    assert row["row_index"] == 0 and row["status"] == status
    assert row["full_prompt_tokens"] == count
    assert row["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert calls == [
        (
            [{"role": "user", "content": prompt}],
            {
                "tokenize": True,
                "return_dict": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )
    ]


@pytest.mark.parametrize("tokens", [[], [True], [1.0], [[1]]])
def test_invalid_tokenizer_output_is_not_capacity_evidence(tokens):
    with pytest.raises(ValueError, match="complete tokenized"):
        runner_module().capacity_row(
            {"_id": "software-fixture", "length": "software-fixture"},
            0,
            "SOFTWARE FIXTURE",
            SimpleNamespace(apply_chat_template=lambda *a, **k: tokens),
        )


@pytest.mark.parametrize(
    "count,status",
    [(262016, "FULL_INPUT_FITS"), (262017, "RESOURCE_LIMIT_FULL_CONTEXT")],
)
def test_explicit_larger_deployment_keeps_original_output_budget(count, status):
    row = runner_module().capacity_row(
        {"_id": "software-fixture", "length": "medium"},
        3,
        "SOFTWARE FIXTURE",
        SimpleNamespace(apply_chat_template=lambda *a, **k: [1] * count),
        model_max_length=262144,
    )
    assert row["full_prompt_tokens"] == count
    assert row["status"] == status


@pytest.mark.parametrize("window", [True, 128, 0, 2097153, 262144.0])
def test_invalid_deployment_window_rejected_before_tokenization(window):
    with pytest.raises(ValueError, match="deployment window"):
        runner_module().capacity_row(
            {}, 0, "SOFTWARE FIXTURE", None, model_max_length=window
        )


def test_help_needs_no_optional_model_environment():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_longbench_v2_original_capacity.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--row-index" in result.stdout and "--full" in result.stdout


@pytest.mark.parametrize("overflow", [False, True])
def test_main_software_fixture_preserves_all_selected_tasks(
    tmp_path, monkeypatch, overflow
):
    """Checks CLI/file plumbing only: fake source/hash/tokenizer, no real assets."""
    runner = runner_module()
    rows = [
        {
            "_id": f"software-fixture-{index}",
            "length": "software-fixture",
            "software_prompt": f"SOFTWARE_FIXTURE_{index}",
        }
        for index in range(503)
    ]
    data = tmp_path / "software-fixture.json"
    data.write_text(json.dumps(rows), encoding="utf-8")
    archive = tmp_path / "software-fixture.zip"
    code = """model_map = {}
URL = "SOFTWARE_FIXTURE_NO_NETWORK"
AutoTokenizer = None
OpenAI = None
def query_llm(*args, **kwargs):
    raise AssertionError("no inference allowed in software test")
def get_pred(data, args, fout):
    tokenizer = AutoTokenizer.from_pretrained(model_map[args.model], trust_remote_code=True)
    client = OpenAI(base_url=URL, api_key="SOFTWARE_FIXTURE")  # pragma: allowlist secret - software-only dummy key
    for item in data:
        query_llm(item["software_prompt"], args.model, tokenizer, client,
                  temperature=0.1, max_new_tokens=128)
        raise AssertionError("no fabricated answer or scoring allowed")
"""
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("THUDM-LongBench-2e00731/pred.py", code)
    tokenizer_path = tmp_path / "software-tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer.json").write_text(
        "SOFTWARE FIXTURE NOT TOKENIZER DATA"
    )
    monkeypatch.setattr(
        runner, "DATA_SHA", hashlib.sha256(data.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        runner, "ARCHIVE_SHA", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(runner.metadata, "version", lambda name: "software-fixture")

    def apply(messages, **kwargs):
        return [1] * (
            131000
            if overflow and messages[0]["content"] == "SOFTWARE_FIXTURE_1"
            else 10
        )

    def load(path, *, trust_remote_code, local_files_only):
        assert path == str(tokenizer_path.resolve())
        assert trust_remote_code is True and local_files_only is True
        return SimpleNamespace(apply_chat_template=apply)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "software-fixture",
            "--data-file",
            str(data),
            "--official-archive",
            str(archive),
            "--tokenizer",
            str(tokenizer_path),
            "--output",
            "prepared-relative-output",
            "--row-index",
            "0",
            "--row-index",
            "1",
            "--row-index",
            "2",
        ],
    )
    runner.main()
    assert Path.cwd() == tmp_path
    receipt = json.loads(
        (
            tmp_path / "prepared-relative-output/capacity-selection.private.json"
        ).read_text()
    )
    assert [row["row_index"] for row in receipt["selected"]] == [0, 1, 2]
    assert receipt["ready_for_execution"] is (not overflow)
    assert receipt["selected"][1]["status"] == (
        "RESOURCE_LIMIT_FULL_CONTEXT" if overflow else "FULL_INPUT_FITS"
    )
    assert receipt["model_generation_calls"] == 0
    assert not receipt["official_scoring_executed"]
    assert not receipt["native_benchmark_executed"]
    assert not receipt["source_completion_checked"]


def test_server_original_capacity_audit_and_dependency_snapshot_are_not_native_run():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/longbench-v2-original-capacity-cpu-20261024.json"
        ).read_text("utf-8")
    )
    # Preserve the actual server-run digest; later source-only formatting must
    # not rewrite historical execution evidence as if the new bytes ran there.
    assert receipt["helper"] == "scripts/prepare_longbench_v2_original_capacity.py"
    assert (
        receipt["helper_sha256"]
        == (
            "d4ca6bd413090c6a936b4aac3681c7d4dd3735aa1972db176cf6d3bbdbcb05f2"  # pragma: allowlist secret
        )
    )
    assert receipt["fixed_original_indices"] == [1, 4, 6]
    assert receipt["prior_original_prompt_identity_and_token_counts_equal"]
    assert receipt["full_prompt_tokens"] == [38444, 24590, 46971]
    assert receipt["counterexample_original_indices"] == [0, 1, 2]
    assert receipt["counterexample_statuses"] == [
        "RESOURCE_LIMIT_FULL_CONTEXT",
        "FULL_INPUT_FITS",
        "RESOURCE_LIMIT_FULL_CONTEXT",
    ]
    assert receipt["counterexample_not_filtered_replaced_or_truncated"]
    assert receipt["native_entry_rejects_blocked_selection_before_model"]
    assert receipt["isolated_client_without_system_site_packages"]
    assert receipt["pip_check_passed"]
    snapshot = ROOT / "configs/workload-native/longbench-v2-qwen35-client-py312.txt"
    assert (
        hashlib.sha256(snapshot.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == receipt["dependency_snapshot_sha256"]
    )
    assert not receipt["original_upstream_software_stack_reproduction_claim"]
    assert receipt["model_generation_calls"] == 0
    assert not receipt["official_scoring_executed"]
    assert not receipt["native_benchmark_executed"]
    assert not receipt["source_completion_checked"]
