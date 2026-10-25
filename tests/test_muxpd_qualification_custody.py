"""Bounded synthetic integrity tests; these do not qualify a runtime or device."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "reports/muxpd-ecpa-20261025"


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_valid_response_difference_is_not_text_equivalence():
    facts = module("extract_qualification_facts")

    def row(text, tokens):
        return {
            "case": 2,
            "payload": {"temperature": 0},
            "http_status": 200,
            "passed": True,
            "response": {
                "choices": [
                    {
                        "message": {"content": text},
                        "finish_reason": "stop",
                        "token_ids": None,
                    }
                ],
                "usage": {"completion_tokens": tokens},
            },
        }

    left, right = row("说明甲", 35), row("说明乙", 34)
    result = facts.natural_comparison([left], [right])
    assert result["exact_text_count"] == 0
    assert result["rows"][0]["candidate_check_passed"]
    assert not result["rows"][0]["token_ids_observed"]
    right["payload"] = {"temperature": 1}
    with pytest.raises(ValueError, match="payload"):
        facts.natural_comparison([left], [right])


def test_archive_is_deterministic_and_source_independent(tmp_path):
    archive = module("archive_qualification")
    source = tmp_path / "input.json"
    source.write_bytes(b'{"failed":true}\n')
    index = archive.make_index({"qualification/failed/input.json": source})
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    archive.pack(first, index)
    archive.pack(second, index)
    assert first.read_bytes() == second.read_bytes()
    source.unlink()
    assert archive.verify(first, index)["verified"]
    with pytest.raises(FileExistsError):
        archive.pack(first, index)


def test_changed_source_cannot_be_packed_and_failure_is_retained(tmp_path):
    archive = module("archive_qualification")
    source = tmp_path / "input"
    source.write_bytes(b"before")
    index = archive.make_index({"raw/input": source})
    source.write_bytes(b"after")
    target = tmp_path / "failed.tar.gz"
    with pytest.raises(ValueError, match="Source changed"):
        archive.pack(target, index)
    assert target.exists() and source.read_bytes() == b"after"


def test_archive_verification_rejects_wrong_or_duplicate_index(tmp_path):
    archive = module("archive_qualification")
    source = tmp_path / "input"
    source.write_bytes(b"preserved")
    index = archive.make_index({"raw/input": source})
    target = tmp_path / "archive.tar.gz"
    archive.pack(target, index)
    index["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content mismatch"):
        archive.verify(target, index)
    index["files"].append(index["files"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        archive.verify(target, index)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "raw/../escape", "raw//alias"]
)
def test_archive_member_paths_are_canonical(name):
    with pytest.raises(ValueError, match="Unsafe"):
        module("archive_qualification").safe_member(name)


def test_numerical_study_recomputes_differences_without_a_quality_oracle(tmp_path):
    study = module("extract_numerical_study")
    payload = {"temperature": 0, "seed": 42, "max_tokens": 96}
    for role, text in (("native", "甲"), ("candidate", "乙")):
        row = {
            "payload": payload,
            "http_status": 200,
            "passed": True,
            "response": {
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1},
            },
        }
        (tmp_path / f"{role}-round-0-natural.json").write_text(
            json.dumps([{**row, "case": 0}])
        )
        (tmp_path / f"{role}-isolated.json").write_text(
            json.dumps([{**row, "repeat": 0}])
        )
    summary = {
        role: {
            "concurrent_distinct_texts_by_case": {"0": 1},
            "isolated_cache_distinct_texts": 1,
        }
        for role in ("native", "candidate")
    }
    summary.update(
        cross_endpoint_round_exact_matches=[0], isolated_cross_endpoint_exact_matches=0
    )
    receipt = {
        "passed": True,
        "errors": [],
        "started_at_utc": "start",
        "finished_at_utc": "end",
        "scope": "synthetic fixture",
        "artifact_sources": {},
        "services": [],
        "summary": summary,
        "plan": {
            "endpoint_order": ["native", "candidate"],
            "concurrent_rounds": 1,
            "cases_per_round": 1,
            "isolated_cache_explanations": 1,
            **payload,
        },
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(receipt))
    for name in (
        "repeat_outputs.py",
        "helpers.py",
        "configuration.json",
        "native-service.log",
        "candidate-service.log",
    ):
        (tmp_path / name).write_text("synthetic fixture")
    result = study.reduce_study(tmp_path)
    assert result["nonempty_http_success_count"] == 4
    assert result["concurrent_exact_total"] == 0
    assert not any(row["token_ids_present"] for row in result["requests"])
    receipt["summary"]["cross_endpoint_round_exact_matches"] = [1]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Recomputed summary differs"):
        study.reduce_study(tmp_path)


def test_archive_cannot_include_an_active_qualification(tmp_path):
    (tmp_path / "qualification.json").write_text('{"passed": false}')
    with pytest.raises(ValueError, match="Active qualification"):
        module("archive_qualification").mapping(tmp_path, tmp_path, tmp_path, tmp_path)


def test_ci_inventory_rejects_live_and_changed_sources(tmp_path):
    archive = module("archive_qualification")
    raw = tmp_path / "full.log"
    raw.write_bytes(b"one retained failure\n")
    row = {
        "path": str(raw),
        "bytes": raw.stat().st_size,
        "sha256": archive.digest(raw.read_bytes()),
        "live": False,
    }
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"status": "pending", "files": [row]}))
    with pytest.raises(ValueError, match="not terminal"):
        archive.ci_sources(inventory)
    inventory.write_text(
        json.dumps({"status": "terminal", "files": [{**row, "live": True}]})
    )
    with pytest.raises(ValueError, match="source is live"):
        archive.ci_sources(inventory)
    inventory.write_text(json.dumps({"status": "terminal", "files": [row]}))
    assert len(archive.ci_sources(inventory)) == 2
    raw.write_bytes(b"replacement success\n")
    with pytest.raises(ValueError, match="Frozen CPU/CI source differs"):
        archive.ci_sources(inventory)


def test_publication_archive_preserves_terminal_bytes_without_overwrite(tmp_path):
    archive = module("archive_publication")
    source = tmp_path / "readback.json"
    source.write_bytes(b'{"passed":false}\n')
    row = {
        "member": "readback/failed.json",
        "source": str(source),
        "bytes": source.stat().st_size,
        "sha256": archive.digest(source.read_bytes()),
    }
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"status": "terminal", "scope": "synthetic", "files": [row]})
    )
    first, second = tmp_path / "first", tmp_path / "second"
    archive.pack(inventory, first)
    archive.pack(inventory, second)
    name = "muxpd-website-publication-20261025.tar.gz"
    assert (first / name).read_bytes() == (second / name).read_bytes()
    with pytest.raises(FileExistsError):
        archive.pack(inventory, first)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen source changed"):
        archive.pack(inventory, tmp_path / "changed")
    index = json.loads((first / "website-publication-archive-index.json").read_text())
    assert archive.verify(first / name, index)["files"] == 2
    index["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Changed member"):
        archive.verify(first / name, index)
