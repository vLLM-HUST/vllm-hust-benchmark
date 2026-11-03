"""Invented protocol controls; no original assets, Java search or model calls."""

import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "qrecc_native", ROOT / "scripts/run_qrecc_original_native.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def search_text():
    def encode(x):
        return base64.b64encode(x.encode()).decode()

    raw = json.dumps({"id": "fixture-doc", "contents": "Unchanged invented passage."})
    return "COUNT\t3\nHIT\t" + encode("fixture-doc") + "\t1.2\t" + encode(raw) + "\n"


def fixture(tmp_path, monkeypatch):
    data = tmp_path / "original"
    data.mkdir()
    rows = [
        {
            "Conversation_no": 1,
            "Turn_no": i + 1,
            "Question": "Unchanged question " + str(i),
            "Context": ["Unchanged prior turn"],
            "Rewrite": "DO_NOT_FEED_GOLD_REWRITE",
            "Answer": "DO_NOT_FEED_GOLD_ANSWER",
            "Answer_URL": "DO_NOT_FEED_GOLD_URL",
        }
        for i in range(3)
    ]
    data = data / "test.json"
    data.write_text(json.dumps(rows))
    monkeypatch.setattr(
        module, "TEST_SHA", hashlib.sha256(data.read_bytes()).hexdigest()
    )
    jar = tmp_path / "fixture.jar"
    jar.write_bytes(b"invented-nonexecutable-jar")
    monkeypatch.setattr(
        module, "JAR_SHA256", hashlib.sha256(jar.read_bytes()).hexdigest()
    )
    index = tmp_path / "index"
    index.mkdir()
    (index / "segments_1").write_bytes(b"invented")

    def fake_run(args, **kwargs):
        if args[-1] == "-version":
            return SimpleNamespace(
                stdout="", stderr='openjdk version "11.0.fixture"', returncode=0
            )
        assert args[-3] == str(index) and args[-1] == "100"
        assert Path(args[-2]).read_text() == "predicted query"
        return SimpleNamespace(stdout=search_text(), stderr="", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return dict(
        data_file=data,
        index=index,
        java="fixture-java",
        jar=jar,
        output=tmp_path / "native",
        endpoint="http://127.0.0.1:19468/v1",
        model="fixture-no-model-run",
        expected_documents=3,
    )


@pytest.mark.parametrize("failed_row", [None, 1])
def test_complete_pipeline_or_retained_original_failure(
    tmp_path, monkeypatch, failed_row
):
    kwargs = fixture(tmp_path, monkeypatch)
    calls = []

    def model_call(**args):
        calls.append(args)
        wire = json.dumps(args["messages"])
        assert "DO_NOT_FEED_GOLD" not in wire
        if args["output"].name == "rewrite":
            if args["output"].parent.name == "row-" + str(failed_row):
                raise RuntimeError("invented failure; not a different question")
            return "predicted query"
        payload = json.loads(args["messages"][1]["content"])
        assert (
            payload["retrieved_passages"][0]["content"] == "Unchanged invented passage."
        )
        assert args["budget"] == 512
        return "predicted answer"

    monkeypatch.setattr(module, "model_call", model_call)
    result = module.run(**kwargs)
    assert (
        result["status"]
        == "ORIGINAL_NATIVE_ATTEMPTS_RETAINED_PUBLISHER_SCORING_REQUIRED"
    )
    assert result["failed_original_positions"] == ([] if failed_row is None else [1])
    assert result["completed_original_positions"] == (
        [0, 1, 2] if failed_row is None else [0, 2]
    )
    assert not result["source_checkbox_eligible"]
    predictions = json.loads(
        (kwargs["output"] / "predictions-private.json").read_text()
    )
    assert [x["original_row_index"] for x in predictions] == [0, 1, 2]
    if failed_row is not None:
        assert (
            predictions[1]["prediction"] == ""
            and (kwargs["output"] / "row-1/failure-private.json").exists()
        )
    assert len(calls) == (6 if failed_row is None else 5)


def test_input_labels_never_selected(tmp_path, monkeypatch):
    kwargs = fixture(tmp_path, monkeypatch)
    tasks = module.selected_inputs(kwargs["data_file"], 0, 3)
    assert all(
        set(x)
        == {"original_row_index", "Conversation_no", "Turn_no", "Question", "Context"}
        for x in tasks
    )
    with pytest.raises(ValueError, match="exceeds"):
        module.selected_inputs(kwargs["data_file"], 1, 3)
    kwargs["data_file"].write_text("[]")
    with pytest.raises(ValueError, match="unchanged"):
        module.selected_inputs(kwargs["data_file"], 0, 1)


def test_index_count_and_actual_unique_documents_required():
    hits = module.parse_hits(search_text(), 3)
    assert len(hits) == 1 and hits[0]["rank"] == 1
    with pytest.raises(ValueError, match="document count"):
        module.parse_hits(search_text(), 4)
    with pytest.raises(ValueError, match="distinct"):
        module.parse_hits(search_text() + search_text().splitlines(True)[1], 3)


@pytest.mark.parametrize("http_failure", [False, True])
def test_model_empty_or_http_failure_retained_without_retry(
    tmp_path, monkeypatch, http_failure
):
    body = b'{"choices":[{"message":{"content":""}}]}'
    calls = []

    class Response(io.BytesIO):
        code = 200

    def opening(*args, **kwargs):
        calls.append((args, kwargs))
        if http_failure:
            raise HTTPError(
                "http://127.0.0.1/v1",
                500,
                "invented",
                {},
                io.BytesIO(b'{"error":"invented"}'),
            )
        return Response(body)

    monkeypatch.setattr(module, "urlopen", opening)
    with pytest.raises((ValueError, RuntimeError)):
        module.model_call(
            endpoint="http://127.0.0.1/v1",
            model="fixture",
            messages=[],
            budget=128,
            timeout=1800,
            output=tmp_path,
        )
    assert len(calls) == 1 and (tmp_path / "response-private.json").is_file()
    assert json.loads((tmp_path / "http-status-private.json").read_text())[
        "status"
    ] == (500 if http_failure else 200)


def test_default_plan_and_remote_rejection(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_qrecc_original_native.py"])
    module.main()
    assert json.loads(capsys.readouterr().out)["status"] == "PLAN_ONLY"
    with pytest.raises(ValueError, match="loopback"):
        module.run(
            data_file="unused",
            index="unused",
            java="unused",
            jar="unused",
            output="unused",
            endpoint="https://public.example",
            model="fixture",
        )
