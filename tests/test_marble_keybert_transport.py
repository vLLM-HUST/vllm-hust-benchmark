"""Synthetic queue tests only; no neural or external tool execution."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import marble_keybert_transport as entry  # noqa: E402


def response():
    return {
        "status": "KEYBERT_TOOL_RETURNED_NOT_AGENT_RESULT",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",  # pragma: allowlist secret - original public model revision
        "cpu_neural_fallback_used": False,
        "keywords": [["original term", 0.125]],
    }


def bind_worker(monkeypatch, tmp_path, value):
    original = entry.save_new

    def worker(path, payload):
        original(path, payload)
        target = tmp_path / path.name.replace(".request.json", ".response.json")
        original(target, json.dumps(value).encode())

    monkeypatch.setattr(entry, "save_new", worker)


def call(queue, query="  原始 query\n", **options):
    return entry.QueuedKeyBERT(queue, timeout=0.01).extract_keywords(
        query, keyphrase_ngram_range=(1, 3), stop_words="english", **options
    )


def test_roundtrip_preserves_query_and_original_tuple_pairs(tmp_path, monkeypatch):
    bind_worker(monkeypatch, tmp_path, response())
    assert call(tmp_path) == [("original term", 0.125)]
    saved = next(tmp_path.glob("*.request.json"))
    assert json.loads(saved.read_bytes()) == {"query": "  原始 query\n"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "FAILED"),
        ("cpu_neural_fallback_used", True),
        ("model", "different"),
        ("revision", "different"),
        ("keywords", [["term", float("nan")]]),
        ("keywords", [["term", True]]),
        ("keywords", [["term"]]),
    ],
)
def test_unqualified_response_rejected(tmp_path, monkeypatch, field, value):
    payload = response()
    payload[field] = value
    bind_worker(monkeypatch, tmp_path, payload)
    with pytest.raises(ValueError):
        call(tmp_path)


def test_empty_original_keywords_remain_empty(tmp_path, monkeypatch):
    payload = response()
    payload["keywords"] = []
    bind_worker(monkeypatch, tmp_path, payload)
    assert call(tmp_path, "") == []


def test_timeout_retains_one_request_without_retry(tmp_path):
    with pytest.raises(TimeoutError, match="no resubmission"):
        call(tmp_path)
    assert len(list(tmp_path.glob("*.request.json"))) == 1
    assert not list(tmp_path.glob("*.response.json"))


def test_non_original_options_rejected_before_request(tmp_path):
    with pytest.raises(ValueError):
        entry.QueuedKeyBERT(tmp_path).extract_keywords(
            "query", keyphrase_ngram_range=(1, 1), stop_words="english"
        )
    assert not list(tmp_path.iterdir())


def test_original_factory_restored_after_failure(tmp_path):
    sentinel = object()
    collector = SimpleNamespace(KeyBERT=sentinel)
    with pytest.raises(RuntimeError):
        with entry.keyword_transport(collector, tmp_path):
            assert isinstance(collector.KeyBERT(), entry.QueuedKeyBERT)
            raise RuntimeError("fixture")
    assert collector.KeyBERT is sentinel
