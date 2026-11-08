"""Capture orchestration fixtures, not real model execution evidence."""

import argparse
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "duplex_capture",
    Path(__file__).resolve().parents[1] / "scripts/run_full_duplex_original_capture.py",
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def arguments(tmp_path):
    return argparse.Namespace(
        omni_root=tmp_path / "omni",
        dataset_root=tmp_path / "data",
        output=tmp_path / "capture",
        python="python",
        url="ws://127.0.0.1:19480/v1/realtime",
        dataset_revision_v10="fixture-v10",
        dataset_revision_v15="fixture-v15",
        model="fixture/model",
        model_revision="fixture",
        runtime="fixture-not-real",
    )


def test_plan_keeps_all_categories_original_ids_and_native_defaults(tmp_path):
    args = arguments(tmp_path)
    rows = ENTRY.plan(args)
    assert len(rows) == 27
    assert sum(1 if r["version"] == "v1.0" else 2 for r in rows) == 39
    assert rows[12]["sample_id"] == "icc_backchannel/0"
    assert len({(r["version"], r["sample_id"].split("/")[0]) for r in rows}) == 9
    assert not args.output.exists()
    for row in rows:
        assert "--timeout" not in row["command"]
        assert "--legacy-tail" not in row["command"]
        assert "--max-per-subset" not in row["command"]
        assert row["command"][-2:] == ["--sample-id", row["sample_id"]]


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost/v1/realtime",
        "ws://user:secret@localhost/v1/realtime",  # pragma: allowlist secret - rejection fixture
        "ws://localhost/v1/realtime?token=x",
        "ws://localhost/v1/chat/completions",
    ],
)
def test_plan_refuses_credentials_or_non_native_protocol(tmp_path, url):
    args = arguments(tmp_path)
    args.url = url
    with pytest.raises(ValueError, match="credential-free"):
        ENTRY.plan(args)


def test_mismatching_original_cli_fails_before_any_model_call(tmp_path):
    args = arguments(tmp_path)
    path = args.omni_root / "benchmarks/eval/benchmark_duplex_v10.py"
    path.parent.mkdir(parents=True)
    path.write_text("invented fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="original capture CLI differs"):
        ENTRY.execute(args, ENTRY.plan(args))
    assert not args.output.exists()


def test_failed_original_case_is_retained_without_replacement(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.omni_root.mkdir()
    monkeypatch.setattr(ENTRY, "verify_inputs", lambda args: args.omni_root)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        return argparse.Namespace(returncode=7)

    monkeypatch.setattr(ENTRY.subprocess, "run", fail)
    assert ENTRY.execute(args, ENTRY.plan(args)) == 7
    assert len(calls) == 1
    assert (args.output / "capture-progress-public.json").is_file()
    with pytest.raises(FileExistsError):
        ENTRY.execute(args, ENTRY.plan(args))
