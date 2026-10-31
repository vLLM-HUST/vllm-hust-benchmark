"""Inert scorer-controller contracts; no Docker, SQL, original data or model."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "spider_scoring", ROOT / "scripts/run_spider_official_scoring.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assets = tmp_path / "assets"
    for name in ("author", "inputs", "database/fixture_db"):
        (assets / name).mkdir(parents=True, exist_ok=True)
    author = assets / "author/evaluation.py"
    author.write_bytes(b"inert author fixture, never executed")
    monkeypatch.setattr(
        module, "AUTHOR_FILES", {"evaluation.py": module.digest(author)}
    )
    suite = assets / "database/fixture_db/fixture_db.sqlite"
    suite.write_bytes(b"inert non-database fixture")
    inputs = [
        {
            "original_dev_index": i,
            "db_id": "fixture_db",
            "question": "original question",
            "schema": {"db_id": "fixture_db"},
        }
        for i in range(3)
    ]
    refs = [
        {
            "original_dev_index": i,
            "original_record": {
                "db_id": "fixture_db",
                "question": "original question",
                "query": "SELECT reference",
            },
        }
        for i in range(3)
    ]
    for name, values in (
        ("inputs.private.jsonl", inputs),
        ("references.private.jsonl", refs),
    ):
        (assets / "inputs" / name).write_text(
            "".join(json.dumps(v) + "\n" for v in values)
        )
    (assets / "inputs/gold.private.txt").write_bytes(
        b"SELECT reference\tfixture_db\n" * 3
    )
    receipt = {
        "source_id": "SPIDER",
        "selected_original_dev_indices": [0, 1, 2],
        "reference_sql_in_model_inputs": False,
        "suite_files": [
            {
                "archive_member": "database/fixture_db/fixture_db.sqlite",
                "sha256": module.digest(suite),
            }
        ],
    }
    (assets / "inputs/preparation-public.json").write_text(json.dumps(receipt))
    pred = tmp_path / "predictions.private.txt"
    pred.write_text("SELECT prediction\n" * 3)
    return module, assets, pred


def test_default_plan_makes_no_docker_or_sql_calls(prepared, monkeypatch):
    module, assets, pred = prepared

    def forbidden(*a, **k):
        pytest.fail("plan must not execute Docker, SQL or models")

    monkeypatch.setattr(module.subprocess, "run", forbidden)
    monkeypatch.setattr(module.subprocess, "check_output", forbidden)
    result = module.run(assets=assets, predictions=pred, image="sha256:" + "0" * 64)
    assert result["status"] == "PLAN_ONLY_NO_SQL_EXECUTION"
    assert result["original_dev_indices"] == [0, 1, 2]
    assert result["model_calls"] == 0
    assert not result["sql_executed_on_host"]
    assert not result["plug_value"] and result["keep_distinct"]
    assert not result["source_checkbox_eligible"]
    assert not any(
        "references.private" in p or "inputs.private" in p
        for p in result["staged_files"]
    )


@pytest.mark.parametrize(
    "kind",
    [
        "author",
        "gold",
        "suite",
        "extra_suite",
        "prediction_count",
        "input_question",
        "input_reference",
    ],
)
def test_changed_original_or_prediction_contract_is_rejected(prepared, kind):
    module, assets, pred = prepared
    if kind == "author":
        (assets / "author/evaluation.py").write_text("changed")
    elif kind == "gold":
        (assets / "inputs/gold.private.txt").write_text("changed")
    elif kind == "suite":
        (assets / "database/fixture_db/fixture_db.sqlite").write_text("changed")
    elif kind == "extra_suite":
        (assets / "database/fixture_db/extra.sqlite").write_text("invented")
    elif kind == "prediction_count":
        pred.write_text("one line\n")
    else:
        path = assets / "inputs/inputs.private.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["question" if kind == "input_question" else "query"] = (
            "changed or leaked"
        )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        module.plan(assets, pred, "sha256:" + "0" * 64)


def test_worker_has_no_host_sql_or_network_escape(prepared):
    module, _, _ = prepared
    args = module.worker_command("sha256:" + "0" * 64, "owned", "owned-assets")
    for value in (
        "--network=none",
        "--read-only",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=2g",
        "--pids-limit=64",
        "--keep_distinct",
    ):
        assert value in args
    assert "--plug_value" not in args
    assert "--privileged" not in args
    assert "readonly" in args[args.index("--mount") + 1]
    assert "--pull=never" in args


def test_zero_original_score_is_completed_not_replaced(prepared):
    module, _, _ = prepared
    assert module.parse_metric("count 2 1 0 0 3\nexecution 0 0 0 0 0\n", 3) == 0
    with pytest.raises(ValueError, match="count"):
        module.parse_metric("count 2 1 0 0 2\nexecution 1 1 0 0 1\n", 3)
