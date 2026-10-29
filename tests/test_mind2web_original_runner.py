import importlib.util
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "mind2web_runner", ROOT / "scripts/run_mind2web_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def command(output, endpoint="http://127.0.0.1:19467/v1"):
    return [
        sys.executable,
        str(ROOT / "scripts/run_mind2web_original.py"),
        "--author-root",
        "absent",
        "--tasks-root",
        "absent",
        "--scores",
        "absent",
        "--tokenizer",
        "absent",
        "--output",
        str(output),
        "--endpoint",
        endpoint,
    ]


def test_inert_plan_does_not_require_data_or_model(tmp_path):
    output = tmp_path / "absent"
    report = json.loads(subprocess.check_output(command(output)))
    assert not report["executed"]
    assert report["max_new_tokens"] == report["top_k"] == 50
    assert not output.exists()


def test_endpoint_credentials_rejected(tmp_path):
    bad = "http://user:secret@localhost/v1"  # pragma: allowlist secret
    result = subprocess.run(
        command(tmp_path / "absent", bad),
        capture_output=True,
    )
    assert result.returncode != 0
    assert b"without embedded credentials" in result.stderr


def test_repository_raw_output_rejected():
    result = subprocess.run(
        command(ROOT / "must-not-create-native-output"), capture_output=True
    )
    assert result.returncode != 0
    assert b"outside the checkout" in result.stderr
    assert not (ROOT / "must-not-create-native-output").exists()


def test_candidate_pickle_disallows_unreviewed_callable():
    with pytest.raises(ValueError, match="unreviewed"):
        runner.RestrictedScores(io.BytesIO(pickle.dumps(len))).load()
    assert runner.RestrictedScores(
        io.BytesIO(pickle.dumps({"ranks": {"sample": 1}}))
    ).load() == {"ranks": {"sample": 1}}
