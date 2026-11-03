"""Invented saved-ranking and subprocess controls; not original QReCC runs."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "qrecc_retrieval_score", ROOT / "scripts/score_qrecc_original_retrieval.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def ranking(index=0):
    return {
        "original_row_index": index,
        "Conversation_no": 7,
        "Turn_no": index + 1,
        "retrieved_passages": [
            {
                "docid": "fixture-document",
                "content": "actual saved fixture passage",
                "rank": 1,
                "score": 0.5,
            }
        ],
    }


@pytest.fixture
def original(tmp_path, monkeypatch):
    data = tmp_path / "fixture-test.json"
    rows = [
        {"Conversation_no": 7, "Turn_no": i + 1, "Answer": "fixture answer"}
        for i in range(3)
    ]
    data.write_text(json.dumps(rows))
    saved = tmp_path / "rankings.jsonl"
    saved.write_text("\n".join(json.dumps(ranking(i)) for i in range(3)) + "\n")
    pins = tmp_path / "pins.json"
    pins.write_text("{}")
    monkeypatch.setattr(
        module, "TEST_SHA", hashlib.sha256(data.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(module, "PINS", pins)
    monkeypatch.setattr(module, "verify_source", lambda root, pins: Path(root))
    return dict(
        data_file=data,
        author_root=tmp_path / "author",
        rankings_file=saved,
        output=tmp_path / "scores",
    )


def stdout(count=3):
    return f"Total number of unique queries: {count}\nEmpty retrieval queries retained: 0\nMean Reciprocal Rank (MRR): 0.0000\nRecall@10: 0.00%\nRecall@100: 0.00%\nExtractive Upper Bound for EM (100 point scale): 0.00\nExtractive Upper Bound for F1 (100 point scale): 0.00\n"


def test_scoring_joins_references_only_after_saved_retrieval(original, monkeypatch):
    before = original["rankings_file"].read_bytes()
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=stdout(), stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    report = module.score(**original)
    assert len(calls) == 1 and calls[0][1]["cwd"] == original["output"]
    assert "-IB" in calls[0][0] and "nltk" in calls[0][0][3]
    assert original["rankings_file"].read_bytes() == before
    rows = [
        json.loads(x)
        for x in (original["output"] / "evaluation-input.private.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 3 and all(x["answer"] == "fixture answer" for x in rows)
    assert rows[0]["content"] == "actual saved fixture passage"
    assert (
        report["queries"] == 3
        and report["author_reported_metrics"]["Mean Reciprocal Rank (MRR)"] == "0.0000"
    )
    assert (
        not report["source_checkbox_eligible"]
        and not report["retrieval_provenance_verified"]
    )
    assert report["extractive_upper_bounds_are_not_model_answers"]


@pytest.mark.parametrize(
    "change",
    [
        "reference",
        "wrong-turn",
        "duplicate-row",
        "too-many-hits",
        "duplicate-doc",
        "wrong-rank",
        "nan-score",
        "reference-in-hit",
    ],
)
def test_bad_or_missing_results_not_silently_filtered(original, monkeypatch, change):
    items = [ranking(i) for i in range(3)]
    if change == "reference":
        items[0]["Answer"] = "leak"
    elif change == "wrong-turn":
        items[0]["Turn_no"] = 99
    elif change == "duplicate-row":
        items[1] = items[0]
    elif change == "too-many-hits":
        items[0]["retrieved_passages"] *= 101
    elif change == "duplicate-doc":
        items[0]["retrieved_passages"] *= 2
    elif change == "wrong-rank":
        items[0]["retrieved_passages"][0]["rank"] = 2
    elif change == "nan-score":
        items[0]["retrieved_passages"][0]["score"] = float("nan")
    else:
        items[0]["retrieved_passages"][0]["answer"] = "leak"
    original["rankings_file"].write_text("\n".join(json.dumps(x) for x in items))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: pytest.fail("must reject before author execution"),
    )
    with pytest.raises(ValueError):
        module.score(**original)
    report = json.loads((original["output"] / "receipt-private.json").read_text())
    assert report["status"] == "FAILED_SCORING_RETAINED_NOT_NATIVE_DELIVERY"
    assert not report["source_checkbox_eligible"]


@pytest.mark.parametrize("kind", ["failure", "count", "metric"])
def test_author_failure_or_omitted_query_never_becomes_success(
    original, monkeypatch, kind
):
    out = stdout(2 if kind == "count" else 3)
    if kind == "metric":
        out = out.replace("Recall@100: 0.00%\n", "")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1 if kind == "failure" else 0, stdout=out, stderr="fixture error"
        ),
    )
    with pytest.raises((ValueError, RuntimeError)):
        module.score(**original)
    report = json.loads((original["output"] / "receipt-private.json").read_text())
    assert report["status"].startswith("FAILED")
    assert (original["output"] / "author-stderr.txt").read_text() == "fixture error"


def test_default_plan_does_not_read_missing_inputs_or_create_output(tmp_path):
    missing = tmp_path / "absent"
    output = tmp_path / "plan"
    result = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(ROOT / "scripts/score_qrecc_original_retrieval.py"),
                "--data-file",
                str(missing),
                "--author-root",
                str(missing),
                "--rankings-file",
                str(missing),
                "--output",
                str(output),
            ],
            text=True,
        )
    )
    assert result["status"] == "PLAN_ONLY" and not output.exists()


def test_zero_hit_query_retained_without_fabricated_passage(original, monkeypatch):
    items = [ranking(i) for i in range(3)]
    items[-1]["retrieved_passages"] = []
    original["rankings_file"].write_text("\n".join(json.dumps(x) for x in items))
    result = stdout().replace(
        "Empty retrieval queries retained: 0", "Empty retrieval queries retained: 1"
    )
    result = result.replace("(100 point scale): 0.00", "(100 point scale): undefined")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=result, stderr=""),
    )
    report = module.score(**original)
    assert report["queries"] == 3 and report["empty_retrieval_queries"] == 1
    assert (
        len(
            json.loads(
                (original["output"] / "query-identities.private.json").read_text()
            )
        )
        == 3
    )
    assert (
        len(
            (original["output"] / "evaluation-input.private.jsonl")
            .read_text()
            .splitlines()
        )
        == 2
    )
    assert (
        report["author_reported_metrics"][
            "Extractive Upper Bound for F1 (100 point scale)"
        ]
        is None
    )
