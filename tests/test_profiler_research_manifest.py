import json
from pathlib import Path


def test_profiler_research_custody_manifest():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root / "archive/mod-targeted-20261023/profiler-research/manifest.json"
        ).read_text()
    )
    # Public source commit, not a credential.
    source = "3861e7caa276d6696b9354c1850a7c8ebf9b3ba3"  # pragma: allowlist secret
    assert manifest["source_commit"] == source
    assert manifest["source_visibility"] == "public"
    rows = manifest["files"]
    assert len(rows) == len({row["path"] for row in rows}) == 48
    assert sum(row["bytes"] for row in rows) == 428181
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert (
        manifest["archive"]["sha256"]
        == "ce5d45f2f8b30bbcdcd1de03d9cf79f81c9830efcf498fc6ae5db098e0f15bad"  # pragma: allowlist secret
    )
    assert manifest["archive"]["bytes"] == 140199
    paths = {row["path"] for row in rows}
    assert (
        "paper/request_lifecycle_causal_profiler/request_lifecycle_causal_profiler.pdf"
        in paths
    )
    assert "docs/g4_execution_result.md" in paths
