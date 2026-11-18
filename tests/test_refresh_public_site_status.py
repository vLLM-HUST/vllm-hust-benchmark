from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "site_status", ROOT / "scripts/refresh_public_site_status.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
NOW = "2026-09-03T06:00:00Z"


def issue_reader(url):
    number = int(url.rsplit("/", 1)[1])
    if "/issues/" in url:
        return dict(
            number=number,
            state="closed",
            state_reason="completed",
            closed_at=NOW,
            updated_at=NOW,
        )
    return dict(
        number=number,
        state="closed",
        draft=False,
        merged_at=NOW,
        updated_at=NOW,
        html_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/{number}",
        title="Merged change",
        head={"ref": "feature"},
    )


def test_refresh_preserves_original_acceptance_and_does_not_mutate_input():
    old = json.loads(
        (ROOT / "website-publication/data/issues.json").read_text(encoding="utf-8")
    )
    before = copy.deepcopy(old)
    new = MODULE.refresh_issues(old, issue_reader, NOW)
    assert old == before
    for item, original in zip(new["issues"], old["issues"]):
        assert item["state"] == "closed"
        assert item["status"] == "completed"
        assert item["pr"]["merged_at"] == NOW
        assert item["progress_summary"] == original["progress_summary"]
        assert item["acceptance_criteria"] == original["acceptance_criteria"]
    assert new["last_updated"] == NOW
    assert new["curated_at"] == old["curated_at"]


def test_failure_does_not_replace_known_data():
    old = json.loads(
        (ROOT / "website-publication/data/issues.json").read_text(encoding="utf-8")
    )
    before = copy.deepcopy(old)

    def unavailable(url):
        raise OSError("API unavailable")

    with pytest.raises(OSError):
        MODULE.refresh_issues(old, unavailable, NOW)
    assert old == before


def repository_reader(url):
    if "pypi.org" in url:
        return {"info": {"version": "99.0"}, "urls": [{"upload_time_iso_8601": NOW}]}
    if url.endswith("/commits/main"):
        return {"sha": "a" * 40, "commit": {"committer": {"date": NOW}}}
    name = url.rsplit("/", 1)[1]
    return dict(
        full_name=f"vLLM-HUST/{name}",
        private=False,
        archived=False,
        default_branch="main",
        html_url=f"https://github.com/vLLM-HUST/{name}",
    )


def test_exact_source_identity_and_registry_approval_are_separate():
    result = MODULE.refresh_versions({}, repository_reader, NOW)
    assert len(result["packages"]) == 7
    assert sum(p["group"] == "infrastructure" for p in result["packages"]) == 4
    for pkg in result["packages"]:
        assert pkg["version"] == "main@aaaaaaaa"
        assert pkg["source_commit_url"].endswith("/commit/" + "a" * 40)
    assert result["registry"]["version"] == "99.0"
    assert result["registry"]["approved_for_current_stack"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("private", True),
        ("archived", True),
        ("default_branch", "dev"),
        ("full_name", "other/repo"),
    ],
)
def test_noncanonical_repository_fails_closed(field, value):
    def read(url):
        result = repository_reader(url)
        if "/commits/" not in url:
            result[field] = value
        return result

    with pytest.raises(ValueError):
        MODULE.refresh_versions({}, read, NOW)


@pytest.mark.parametrize(
    "timestamp,stale",
    [
        (NOW, False),
        ("2026-08-01T00:00:00Z", True),
        ("2026-10-01T00:00:00Z", True),
        (None, True),
        ("bad", True),
    ],
)
def test_freshness(timestamp, stale):
    assert MODULE.is_stale(timestamp, NOW) is stale


def test_json_reader_retries_transport_errors(monkeypatch):
    import io
    from urllib.error import URLError

    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise URLError("transient transport failure")
        return io.StringIO('{"verified": true}')

    monkeypatch.setattr(MODULE, "urlopen", request)
    monkeypatch.setattr(MODULE.time, "sleep", lambda seconds: None)
    assert MODULE.read_json("https://pypi.org/pypi/example/json") == {"verified": True}
    assert len(calls) == 3


def test_json_reader_does_not_retry_missing_records(monkeypatch):
    from urllib.error import HTTPError

    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.org", 404, "missing", {}, None)

    monkeypatch.setattr(MODULE, "urlopen", request)
    with pytest.raises(HTTPError):
        MODULE.read_json("https://example.org")
    assert len(calls) == 1
