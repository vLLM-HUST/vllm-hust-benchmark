"""No model or external data used: preserve candidate completeness and native scores."""

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "run_msmarco_original", ROOT / "scripts/run_msmarco_original.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.remove(str(ROOT / "scripts"))


def record():
    from vllm_hust_workload_data.records import WorkloadRecord

    return WorkloadRecord(
        source={"source_id": "MSMARCO_PASSAGE"},
        task={
            "task_id": "10",
            "input": {
                "candidates": [
                    {"passage_id": "1"},
                    {"passage_id": "2"},
                    {"passage_id": "3"},
                ]
            },
        },
        runtime={},
        replay={},
        outcome={"references": {"original_qrel_rows": [["10", "0", "99", "1"]]}},
    )


def test_native_score_sort_is_stable_and_keeps_every_candidate():
    candidates = record().task["input"]["candidates"]
    assert MODULE.rank_candidates(candidates, [0.5, 1.0, 1.0]) == ["2", "3", "1"]
    for scores in [[0.5], [0.5, float("nan"), 1.0], [0.5, True, 1.0]]:
        with pytest.raises(ValueError):
            MODULE.rank_candidates(candidates, scores)


def test_saved_rankings_cannot_drop_candidates_or_change_ids(tmp_path):
    path = tmp_path / "ranks.tsv"
    path.write_text("10\t2\t1\n10\t3\t2\n10\t1\t3\n", encoding="utf-8")
    assert MODULE.read_rankings(path, [record()]) == {"10": ["2", "3", "1"]}
    for wrong in [
        "10\t2\t1\n",
        "10\t2\t1\n10\t2\t2\n10\t1\t3\n",
        "10\t2\t2\n10\t3\t3\n10\t1\t4\n",
    ]:
        path.write_text(wrong, encoding="utf-8")
        with pytest.raises(ValueError):
            MODULE.read_rankings(path, [record()])


def test_cpu_model_fallback_is_rejected_before_data_access(tmp_path):
    with pytest.raises(SystemExit) as error:
        MODULE.main(
            [
                "run",
                "--data-root",
                str(tmp_path),
                "--author-root",
                str(tmp_path),
                "--model-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
                "--device",
                "cpu",
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


def test_scorer_receives_original_selected_qrels_not_candidate_derived_labels(
    tmp_path, monkeypatch
):
    class Result:
        stdout = "MRR @10: 0.0\nQueriesRanked: 1\n"
        stderr = ""

    monkeypatch.setattr(MODULE, "verify", lambda *args: None)
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: Result())
    result = MODULE.score_original(
        tmp_path, tmp_path, [record()], {"10": ["2", "3", "1"]}
    )
    assert result["official_mrr_at10"] == 0.0
    assert (tmp_path / "selected-original-qrels.private.tsv").read_text(
        "utf-8"
    ) == "10\t0\t99\t1\n"
