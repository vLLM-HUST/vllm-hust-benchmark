import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/terminal-openssl-model-recovery-20261026"


def module():
    spec = importlib.util.spec_from_file_location(
        "recovery_verify", FOLDER / "verify.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_recovery_verifier():
    assert module().verify()["tests_passed"] == 6


@pytest.mark.parametrize(
    "case",
    [
        "coverage",
        "original_score",
        "script_snapshot",
        "extra_model",
        "release",
        "transport",
        "budget",
    ],
)
def test_recovery_rejects_semantic_changes_even_with_new_hash(tmp_path, case):
    shutil.copytree(FOLDER, tmp_path / "evidence")
    folder = tmp_path / "evidence"
    data = json.loads((folder / "summary.json").read_text())
    if case == "coverage":
        data["full_dataset_acceptance"] = True
    elif case == "original_score":
        data["original_verifier"]["reward"] = 1
    elif case == "script_snapshot":
        data["script_provenance"]["original_filesystem_sha_captured"] = True
    elif case == "extra_model":
        data["recovery"]["additional_model_calls"] = 1
    elif case == "release":
        data["recovery"]["all_owned_containers_released"] = False
    elif case == "transport":
        data["transport"]["verified_after_execution"] = True
    else:
        data["recovery"]["timeout_seconds"] = 1800
    raw = json.dumps(data).encode()
    (folder / "summary.json").write_bytes(raw)
    review = json.loads((folder / "profiling-review.json").read_text())
    review["summary_sha256"] = hashlib.sha256(raw).hexdigest()
    (folder / "profiling-review.json").write_text(json.dumps(review))
    m = module()
    m.HERE = folder
    with pytest.raises(AssertionError):
        m.verify()


def test_v091_projection_preserves_prior_results(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v091 as b

    data = b.build("fixture", lambda p: (ROOT / p).read_bytes())
    old = json.loads((ROOT / "website-publication/data/midterm-v090.json").read_text())
    for key in (
        "wholechain_reference",
        "m01_method",
        "paired_own_compute",
        "frontierscience_designated_pair",
        "hle_designated_baseline",
        "c16_equal_request_latency",
        "swe_incremental_tasks",
        "terminal_arm64_controls",
    ):
        assert data[key] == old[key]
    assert (
        hashlib.sha256((ROOT / b.OLD_REPORT).read_bytes()).hexdigest()
        == old["report_sha256"]
    )
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    for key in ("tables", "tables_en"):
        row = next(
            r for t in remaining[key] for r in t["rows"] if r[0] == "TerminalBench 2.1"
        )
        assert "88" in row[1] and "88" in row[2]
    section = next(s for s in data["sections"] if s["id"] == "terminal-model-recovery")
    assert section["tables"][0]["rows"][1][1:4] == ["0", "6 / 6", "1"]
    assert "not an original file snapshot" in section["scope_en"]
    assert "原reward=null" in (ROOT / b.REPORT).read_text()
    assert data["terminal_model_recovery"]["model_task_attempts"] == 1
