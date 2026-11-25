import json
import os
import subprocess
import sys
from pathlib import Path


def invoke(tmp_path, *args):
    env = dict(os.environ, VLLM_HUST_WORKSPACE_ROOT=str(tmp_path))
    env.pop("DEEPLINKENGINE_ROOT", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-m", "vllm_hust_benchmark.cli", "experiment", *args],
        cwd=tmp_path,
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )


def test_forwards_unmodified_arguments_cwd_and_status(tmp_path):
    workspace = tmp_path / "workspace with spaces"
    engine = workspace / "DeepLinkEngine"
    engine.mkdir(parents=True)
    (engine / "deeplink.py").write_text(
        "import json, os, sys\nprint(json.dumps([os.getcwd(), sys.argv[1:]]))\nsys.exit(17)\n"
    )
    args = [
        "--site",
        "my site.json",
        "--controllers",
        "kv-control,instance-scaling",
        "--dry-run",
    ]
    result = invoke(workspace, *args)
    assert result.returncode == 17
    assert json.loads(result.stdout) == [str(workspace.resolve()), ["benchmark", *args]]


def test_missing_engine_checkout(tmp_path):
    result = invoke(tmp_path, "--help")
    assert result.returncode == 2
    assert "DEEPLINKENGINE_ROOT" in result.stderr
