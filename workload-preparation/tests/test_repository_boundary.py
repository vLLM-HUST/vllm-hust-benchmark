import importlib.util
from pathlib import Path
import subprocess
import sys
import pytest

import vllm_hust_workload_data as package

ROOT = Path(__file__).resolve().parents[1]


def test_no_tracked_raw_enterprise_or_public_payloads():
    git_root = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if git_root.returncode:
        pytest.skip(
            "Git tracking guard applies to checkout; archives are checked separately"
        )
    repository = Path(git_root.stdout.strip()).resolve()
    relative = ROOT.relative_to(repository).as_posix()
    patterns = [
        f"{relative}/{suffix}" if relative != "." else suffix
        for suffix in ("*.jsonl", "*.parquet", "*.csv")
    ]
    result = subprocess.run(
        ["git", "ls-files", "--", *patterns],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_old_generator_modules_are_absent_and_not_imported():
    for name in [
        "families",
        "presets",
        "descriptors",
        "shared_workload_smoke",
        "metadata",
        "contracts",
        "trace_schema",
        "real_workload_replay",
    ]:
        assert importlib.util.find_spec("vllm_hust_workload_data." + name) is None
        assert "vllm_hust_workload_data." + name not in sys.modules
    assert not hasattr(package, "generate_repo_local_workload_requests")


def test_core_import_has_no_optional_media_or_external_runtime_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;import vllm_hust_workload_data;assert not any(n.startswith(('pyarrow','torch','vllm_quality_bounded_inference')) for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_makefile_does_not_start_models_or_downstream_plugins():
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "managed-plugin" not in text
    assert "MODEL" not in text
    assert "LLM_SERVING_WORKLOADS_SRC" not in text
