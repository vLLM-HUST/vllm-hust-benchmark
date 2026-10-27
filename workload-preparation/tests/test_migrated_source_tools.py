"""Offline software fixtures for migrated helpers; not real dataset evidence."""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "source_tools"


@contextmanager
def load_tool(name):
    """Mirror direct script imports without requiring a research installation."""
    prior = list(sys.path)
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "migration_" + name, TOOLS / (name + ".py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = prior


def test_source_helpers_do_not_import_research_or_execute_native_tasks():
    import ast

    for path in TOOLS.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any(name.startswith("llm_serving_workloads") for name in names)
        assert not names & {"torch", "openai", "httpx", "subprocess"}


def test_download_inspection_does_not_download_or_write(monkeypatch, tmp_path, capsys):
    with load_tool("download_dataset") as tool:
        catalogs = tmp_path / "data/catalogs"
        catalogs.mkdir(parents=True)
        (catalogs / "authoritative_datasets_v1.json").write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "source_id": "GSM8K",
                            "platform": "hf",
                            "repo_id": "fixture/data",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(tool, "ROOT", tmp_path)
        monkeypatch.setattr(
            tool,
            "fetch_json",
            lambda url: {
                "sha": "reviewed",
                "gated": False,
                "siblings": [{"rfilename": "test.jsonl", "size": 1}],
            },
        )
        monkeypatch.setattr(
            tool, "source_spec", lambda *args: {"files": [], "revision": None}
        )
        monkeypatch.setattr(
            tool, "acquire", lambda *args: pytest.fail("inspect must not acquire")
        )
        monkeypatch.setattr(sys, "argv", ["download_dataset.py", "GSM8K"])
        tool.main()
        result = json.loads(capsys.readouterr().out)
        assert result["revision"] == "reviewed"
        assert not (tmp_path / ".cache").exists()
        assert not (tmp_path / "results").exists()


def test_download_requires_explicit_receipt_before_network(monkeypatch):
    with load_tool("download_dataset") as tool:
        monkeypatch.setattr(
            tool, "fetch_json", lambda *args: pytest.fail("network forbidden")
        )
        monkeypatch.setattr(sys, "argv", ["download_dataset.py", "GSM8K", "--download"])
        with pytest.raises(SystemExit) as error:
            tool.main()
        assert error.value.code == 2


def test_existing_release_bytes_are_verified_not_downloaded(monkeypatch, tmp_path):
    with load_tool("acquire_release_datasets") as tool:
        raw = b"column_a,column_b\n1,2\n"
        folder = tmp_path / ".cache/public_datasets/BURSTGPT/vfixture"
        folder.mkdir(parents=True)
        (folder / "fixture.csv").write_bytes(raw)
        monkeypatch.setattr(tool, "ROOT", tmp_path)
        monkeypatch.setattr(
            tool, "urlopen", lambda *args, **kwargs: pytest.fail("network forbidden")
        )
        source = {
            "source_id": "BURSTGPT",
            "repo_id": "fixture/data",
            "tag": "vfixture",
            "asset_kind": "metadata",
            "unit": "request_records",
            "scope": "software fixture",
            "files": [
                {
                    "name": "fixture.csv",
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            ],
        }
        result = tool.acquire(source)
        assert result["row_count"] == 1
        assert result["quality_or_runtime_evaluation"] == "NOT_RUN"
        source["files"][0]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="refusing to overwrite"):
            tool.acquire(source)
        assert (folder / "fixture.csv").read_bytes() == raw


def test_release_external_data_root_reuses_original_bytes(tmp_path, monkeypatch):
    with load_tool("acquire_release_datasets") as tool:
        component = tmp_path / "component"
        component.mkdir()
        monkeypatch.setattr(tool, "ROOT", component)
        root = tmp_path / "server-private-data"
        folder = root / "BURSTGPT/vfixture"
        folder.mkdir(parents=True)
        raw = b"column_a,column_b\n1,2\n"
        (folder / "fixture.csv").write_bytes(raw)
        monkeypatch.setattr(
            tool, "urlopen", lambda *args, **kwargs: pytest.fail("network forbidden")
        )
        source = {
            "source_id": "BURSTGPT",
            "repo_id": "fixture/data",
            "tag": "vfixture",
            "asset_kind": "metadata",
            "unit": "request_records",
            "scope": "fixture",
            "files": [
                {
                    "name": "fixture.csv",
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            ],
        }
        result = tool.acquire(source, data_root=root)
        assert (
            result["files"][0]["relative_local_path"] == "BURSTGPT/vfixture/fixture.csv"
        )
        assert result["files"][0]["path_relative_to"] == "explicit_data_root"
        assert not (component / ".cache").exists()
        with pytest.raises(ValueError, match="outside the component"):
            tool.acquire(source, data_root=component / "data")
        assert not (component / "data").exists()


def test_release_failures_expose_cause_types_without_private_details():
    from urllib.error import HTTPError, URLError

    with load_tool("acquire_release_datasets") as tool:
        assert tool.failure_metadata(URLError(TimeoutError("PRIVATE_SENTINEL"))) == {
            "error_type": "URLError",
            "reason_type": "TimeoutError",
        }
        assert tool.failure_metadata(
            HTTPError("https://secret.invalid", 403, "PRIVATE_SENTINEL", {}, None)
        ) == {
            "error_type": "HTTPError",
            "http_status": 403,
        }
        assert tool.failure_metadata(ValueError("PRIVATE_SENTINEL")) == {
            "error_type": "ValueError"
        }


def test_release_jsonl_gzip_and_unsafe_selection(tmp_path, monkeypatch):
    import gzip

    with load_tool("acquire_release_datasets") as tool:
        path = tmp_path / "fixture.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write('{"input":"invented fixture"}\n')
        assert tool.inspect(path)["row_count"] == 1
        with pytest.raises(ValueError, match="safety bound"):
            tool.inspect(path, max_uncompressed_bytes=1)
        monkeypatch.setattr(tool, "ROOT", tmp_path)
        with pytest.raises(ValueError, match="unsafe release path"):
            tool.acquire({"source_id": "../outside", "tag": "vfixture", "files": []})


def test_embedded_image_presence_is_not_a_visual_task(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    with load_tool("validate_embedded_media") as tool:
        path = tmp_path / "fixture.parquet"
        images = pa.array(
            [
                {"bytes": b"not-a-real-image", "path": None},
                {"bytes": None, "path": "missing.png"},
            ],
            type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
        )
        pq.write_table(pa.table({"image": images}), path)
        result = tool.validate_file(path)
        assert result["rows"] == 2
        assert result["nonempty_embedded_images"] == 1
        assert result["all_rows_have_embedded_image"] is False


def test_per_file_validation_preserves_outcome_separation(tmp_path, monkeypatch):
    from vllm_hust_workload_data.records import WorkloadRecord

    with load_tool("validate_source_adapters") as tool:
        path = tmp_path / "fixture.jsonl"
        path.write_text(
            '{"question":"fixture","answer":"REFERENCE_SENTINEL"}\n', encoding="utf-8"
        )
        spec = {
            "source_id": "GSM8K",
            "revision": "software-only",
            "files": [{"path": path.name, "sha256": tool.sha256(path), "role": "data"}],
        }
        monkeypatch.setattr(tool, "source_spec", lambda sid: spec)
        row = WorkloadRecord(
            source={"source_id": "GSM8K"},
            task={"input": {"question": "fixture"}},
            runtime={"kind": "task_dataset"},
            replay={"unmapped_field_names": []},
            outcome={"answer": "REFERENCE_SENTINEL"},
        )
        monkeypatch.setattr(tool, "iter_workload", lambda *args, **kwargs: iter([row]))
        result = tool.validate("GSM8K", tmp_path, limit_per_file=1)
        assert result["rows_adapted"] == 1
        assert result["model_calls"] == 0
        assert result["files"][0]["scope"] == "BOUNDED_PREFIX_NOT_FULL_FILE"
        assert "REFERENCE_SENTINEL" not in json.dumps(result)
