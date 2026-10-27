"""Software-only fixtures; no model, original task or native quality validation."""

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "dureader_prepare", ROOT / "scripts/prepare_dureader_official_scorer.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_existing_output_is_not_overwritten(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="fresh output"):
        module.prepare(tmp_path, fetch=lambda _: pytest.fail("network not allowed"))
    assert marker.read_text("utf-8") == "existing"


@pytest.mark.parametrize("corrupt", [False, True])
def test_publisher_api_validates_git_blob(monkeypatch, corrupt):
    payload = b"original public software fixture\n"
    metadata = {
        "type": "file",
        "path": "utils/fixture.py",
        "encoding": "base64",
        "content": base64.b64encode(payload).decode(),
        "size": len(payload),
        "sha": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
    }
    if corrupt:
        metadata["size"] += 1

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, bound):
            return json.dumps(metadata).encode()

    requests = []
    monkeypatch.setattr(
        module,
        "urlopen",
        lambda request, **kw: requests.append(request.full_url) or Response(),
    )
    url = (
        "https://raw.githubusercontent.com/fixture/publisher/"
        + "a" * 40
        + "/utils/fixture.py"
    )
    if corrupt:
        with pytest.raises(ValueError, match="bytes mismatch"):
            module.github_fetch(url)
    else:
        assert module.github_fetch(url) == payload
    assert requests == [
        "https://api.github.com/repos/fixture/publisher/contents/utils/fixture.py?ref="
        + "a" * 40
    ]


def test_scoring_compatibility_failures_are_retained(monkeypatch, tmp_path):
    def metric(predictions, references, task, source):
        assert source == "both"
        assert len(predictions[3]["answers"]) == 1
        assert len(references[3]["answers"]) == 3
        if task in {"all", "yesno"}:
            raise AssertionError("dictionary insertion ordering fixture")
        return {"F1": 1.0, "Rouge-L": 1.0}

    monkeypatch.setattr(
        module.importlib, "import_module", lambda _: SimpleNamespace(get_metrics=metric)
    )
    scores = module.fixture_check(tmp_path)
    assert scores["main"]["Rouge-L"] == 1
    assert scores["all"]["software_compatibility_error"] == "AssertionError"
    assert scores["yesno"]["detail"] == "dictionary insertion ordering fixture"


def test_pinned_software_has_no_dataset_download():
    assert len(module.FILES) == 10
    assert all("data/demo" not in entry[2] for entry in module.FILES.values())
    assert {entry[1] for entry in module.FILES.values()} == {
        module.DUREADER_REVISION,
        module.COCO_REVISION,
    }
