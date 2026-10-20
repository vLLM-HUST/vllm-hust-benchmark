import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"


def receipt():
    return json.loads((REPORT / "b1-mod-source-audit-20261020.json").read_text())


def test_source_receipts_are_pinned_and_complete():
    data = receipt()
    assert len(data["repositories"]) == 22
    assert len({row["repository"] for row in data["repositories"]}) == 22
    for row in data["repositories"]:
        assert len(row["commit"]) == 40
        assert row["default_branch"]
        assert any(source["path"] == "README.md" for source in row["sources"])
        for source in row["sources"]:
            assert (
                hashlib.sha256(source["text"].encode()).hexdigest() == source["sha256"]
            )
            assert f"/blob/{row['commit']}/" in source["url"]
    for source in data["measured_host_sources"]:
        assert hashlib.sha256(source["text"].encode()).hexdigest() == source["sha256"]


def test_hybrid_fastpath_rejects_current_mtp2_before_device_access():
    row = next(
        row
        for row in receipt()["repositories"]
        if row["repository"].endswith("/vllm-hust-hybrid-spec")
    )
    source = next(
        s["text"]
        for s in row["sources"]
        if s["path"] == "tools/gdn_prepare_fastpath.py"
    )
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "try_build"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "pinned-fastpath", "exec"
        ),
        namespace,
    )
    builder = SimpleNamespace(vllm_config=SimpleNamespace(), num_spec=2)
    # Deliberately no tensor API: rejection must precede every device access.
    assert namespace["try_build"](builder, 0, None, None, None, api=None) is None


def test_original_source_receipt_remains_distinct():
    old = json.loads((REPORT / "b1-mod-source-audit.json").read_text())
    assert len(old["repositories"]) == 9
    assert old["captured_at"] != receipt()["captured_at"]
