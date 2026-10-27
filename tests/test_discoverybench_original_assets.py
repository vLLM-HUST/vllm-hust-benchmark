"""Tiny synthetic filesystem fixtures; not original DiscoveryBench evidence."""

import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "discovery_assets", ROOT / "scripts/prepare_discoverybench_original_assets.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_default_plan_has_no_data_or_model_activity(tmp_path):
    target = tmp_path / "absent"
    result = MODULE.prepare(target)
    assert result["status"] == "PLAN_ONLY" and not target.exists()
    assert result["model_calls"] == 0 and result["source_checkbox_eligible"] is False


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b"])
def test_unsafe_original_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        MODULE.safe_path(tmp_path, name)


def test_offline_git_blob_and_file_set_verification(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    file = source / "tiny.csv"
    file.write_bytes(b"original fixture,not benchmark data\n")
    tree = {
        "truncated": False,
        "tree": [
            {
                "type": "blob",
                "mode": "100644",
                "path": "tiny.csv",
                "size": file.stat().st_size,
                "sha": MODULE.blob_identity(file),
            }
        ],
    }
    (tmp_path / "publisher-tree-private.json").write_text(json.dumps(tree))
    result = MODULE.prepare(tmp_path, verify_only=True)
    assert result["files_verified"] == 1 and result["benchmark_executed"] is False
    (source / "extra.csv").write_text("not original")
    with pytest.raises(ValueError, match="file set"):
        MODULE.prepare(tmp_path, verify_only=True)
    (source / "extra.csv").unlink()
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        MODULE.prepare(tmp_path, verify_only=True)


def test_existing_data_never_overwritten(tmp_path):
    with pytest.raises(ValueError, match="fresh external"):
        MODULE.prepare(tmp_path, execute=True)


@pytest.mark.parametrize(
    "endpoint", ["https://huggingface.co", "https://hf-mirror.com"]
)
def test_official_index_plan_and_exact_readonly_reuse(tmp_path, monkeypatch, endpoint):
    target = tmp_path / "publisher-index"
    assert MODULE.prepare_publisher_index(target)["status"] == "PLAN_ONLY"
    assert not target.exists()
    payloads = {
        "README.md": b"fixture card\n",
        "train_relevant.csv": b"qid,question\n0,original fixture\n",
        "test_relevant.csv": b"qid,question\n1,fixed first\n2,fixed second\n",
    }
    identities = {
        name: (
            len(data),
            MODULE.hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
        )
        for name, data in payloads.items()
    }
    monkeypatch.setattr(MODULE, "HF_INDEX_FILES", identities)
    monkeypatch.setattr(
        MODULE,
        "urlopen",
        lambda request, **kwargs: io.BytesIO(
            payloads[request.full_url.rsplit("/", 1)[1]]
        ),
    )
    result = MODULE.prepare_publisher_index(target, execute=True, hf_endpoint=endpoint)
    assert result["transport_endpoint"] == endpoint
    assert result["files_verified"] == 3
    assert result["original_csv_structure"]["test_relevant.csv"]["original_rows"] == 2
    assert result["source_checkbox_eligible"] is False
    monkeypatch.setattr(
        MODULE,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("read-only reuse requested network"),
    )
    assert (
        MODULE.prepare_publisher_index(target, verify_only=True)["files_verified"] == 3
    )
    (target / "test_relevant.csv").write_text("changed")
    with pytest.raises(ValueError, match="blob/size mismatch"):
        MODULE.prepare_publisher_index(target, verify_only=True)


def test_official_index_rejects_existing_destination(tmp_path):
    with pytest.raises(ValueError, match="fresh external"):
        MODULE.prepare_publisher_index(tmp_path, execute=True)


def test_official_index_rejects_arbitrary_transport(tmp_path):
    with pytest.raises(ValueError, match="fixed public"):
        MODULE.prepare_publisher_index(
            tmp_path / "absent", hf_endpoint="http://other.invalid"
        )


def test_official_answer_keys_stay_reference_assets(tmp_path, monkeypatch):
    payload = b"dataset,metadata_id,qid,gold\nfixture,0,0,reference\n"
    identity = MODULE.hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload
    ).hexdigest()
    monkeypatch.setattr(
        MODULE, "HF_ANSWER_FILES", {"answer_key/tiny.csv": (len(payload), identity)}
    )
    monkeypatch.setattr(MODULE, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    result = MODULE.prepare_publisher_index(
        tmp_path / "keys", execute=True, answer_keys=True
    )
    assert (
        result["release"] == "publisher-answer-keys" and result["files_verified"] == 1
    )
    assert result["original_csv_structure"]["answer_key/tiny.csv"]["original_rows"] == 1
    assert result["source_checkbox_eligible"] is False and result["model_calls"] == 0


def test_reuse_non_utf8_original_bytes_records_parse_failure_not_fake_readiness(
    tmp_path, monkeypatch
):
    payload = b"id,gold\n0,original\x92reference\n"
    identity = MODULE.hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload
    ).hexdigest()
    monkeypatch.setattr(
        MODULE, "HF_ANSWER_FILES", {"answer_key/tiny.csv": (len(payload), identity)}
    )
    previous = tmp_path / "previous"
    (previous / "answer_key").mkdir(parents=True)
    (previous / "answer_key/tiny.csv").write_bytes(payload)
    monkeypatch.setattr(
        MODULE,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("valid asset redownloaded"),
    )
    output = tmp_path / "fresh"
    result = MODULE.prepare_publisher_index(
        output, execute=True, answer_keys=True, reuse_assets=previous
    )
    assert result["reused_original_files"] == ["answer_key/tiny.csv"]
    assert result["files_verified"] == 1 and len(result["csv_parse_errors"]) == 1
    assert (
        result["status"]
        == "ORIGINAL_PUBLISHER_FILES_VERIFIED_WITH_PARSE_ERRORS_NOT_NATIVE_EXECUTION"
    )
    assert (output / "answer_key/tiny.csv").read_bytes() == payload
    assert (
        result["original_csv_structure"] == {}
        and result["source_checkbox_eligible"] is False
    )


@pytest.mark.parametrize("wrong_member_size", [False, True])
def test_download_extract_identity_and_size_guard_with_tiny_fixture(
    tmp_path, monkeypatch, wrong_member_size
):
    payload = b"fixture only\n"
    identity = MODULE.hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload
    ).hexdigest()
    tree = {
        "truncated": False,
        "tree": [
            {
                "type": "blob",
                "mode": "100644",
                "path": "tiny.csv",
                "size": len(payload) + int(wrong_member_size),
                "sha": identity,
            }
        ],
    }
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        member = tarfile.TarInfo("fixture-root/tiny.csv")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    monkeypatch.setattr(
        MODULE,
        "urlopen",
        lambda request, **kwargs: io.BytesIO(
            json.dumps(tree).encode()
            if request.full_url == MODULE.API
            else archive.getvalue()
        ),
    )
    output = tmp_path / "external"
    if wrong_member_size:
        with pytest.raises(ValueError, match="file size differs"):
            MODULE.prepare(output, execute=True)
        result = json.loads((output / "preparation-public.json").read_text())
        assert result["status"] == "FAILED_NOT_COMPLETE"
        assert not (output / "source/tiny.csv").exists()
    else:
        result = MODULE.prepare(output, execute=True)
        assert result["files_verified"] == 1
        assert (output / "source/tiny.csv").read_bytes() == payload
    assert result["source_checkbox_eligible"] is False
