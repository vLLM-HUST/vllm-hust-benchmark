"""Independent-port source fixtures only, not scientific or Agent validation."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_mlagentbench_messages_agent as prepare
import run_mlagentbench_messages_agent as runner


@pytest.fixture
def args(tmp_path, monkeypatch):
    source = tmp_path / "author"
    (source / "MLAgentBench").mkdir(parents=True)
    files = {
        "MLAgentBench/LLM.py": b"# pinned original LLM\n",
        "MLAgentBench/runner.py": prepare.chat.OPTIONAL_IMPORT
        + b"\n"
        + prepare.chat.RUN
        + b"\n",
        "original-task.txt": b"original task and references",
    }
    for name, raw in files.items():
        (source / name).write_bytes(raw)
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps(
            dict(
                files=[
                    dict(path=name, sha256=hashlib.sha256(raw).hexdigest())
                    for name, raw in files.items()
                ]
            )
        )
    )
    monkeypatch.setattr(prepare, "PINS", pins)
    monkeypatch.setattr(prepare.chat, "PINS", pins)
    monkeypatch.setattr(prepare.chat, "verify_source", lambda *_: source)
    return SimpleNamespace(
        action="prepare", author_root=source, output=tmp_path / "messages"
    )


def test_plan_and_runner_do_not_read_or_execute():
    assert (
        prepare.prepare(SimpleNamespace(action="plan"))["source_checkbox_eligible"]
        is False
    )
    assert runner.run(SimpleNamespace(execute=False))["native_agent_tasks_run"] == 0


def test_new_port_keeps_originals_and_historical_protocol(args):
    before = (args.author_root / "MLAgentBench/LLM.py").read_bytes()
    result = prepare.prepare(args)
    assert result["profile"]["max_tokens_to_sample"] == 2000
    assert (args.author_root / "MLAgentBench/LLM.py").read_bytes() == before
    assert (args.output / "originals/MLAgentBench/LLM.py.source").read_bytes() == before
    assert (
        b"complete_original_messages"
        in (args.output / "MLAgentBench/LLM.py").read_bytes()
    )
    assert (args.output / "mlagentbench_local_chat.py").is_file()
    prepare.verify_messages_port(args.output)
    with pytest.raises(FileExistsError):
        prepare.prepare(args)


@pytest.mark.parametrize(
    "path",
    [
        "original-task.txt",
        "MLAgentBench/LLM.py",
        "mlagentbench_local_messages.py",
        "unexpected.py",
    ],
)
def test_no_task_or_transport_tampering(args, path):
    prepare.prepare(args)
    (args.output / path).write_text("modified fixture")
    with pytest.raises(ValueError):
        prepare.verify_messages_port(args.output)
