from pathlib import Path
import hashlib
import json
import runpy
import subprocess
import sys
import tarfile
import zipfile

import pytest


@pytest.mark.parametrize(
    "variant", ["complete", "missing", "wrong_query", "duplicate", "wrong_shape"]
)
def test_fac_input_gate_preserves_empty_answers_and_rejects_mismatch(tmp_path, variant):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/check_stabletoolbench_fac_inputs.py"
    )
    keys = ["588", "608", "1073"]
    rows = [{"query_id": int(key), "query": key} for key in keys]
    answers = {key: {"query": key, "answer": {"final_answer": ""}} for key in keys}
    if variant == "missing":
        del answers["608"]
    elif variant == "wrong_query":
        answers["608"]["query"] = "modified"
    elif variant == "wrong_shape":
        answers = keys
    args = [sys.executable, str(script)]
    for name, data in (
        ("instructions", rows),
        ("ids", dict.fromkeys(keys, 0)),
        ("answers", answers),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        if name == "answers" and variant == "duplicate":
            path.write_text('{"608": {}, "608": {}}', encoding="utf-8")
        args.extend([f"--{name}", str(path)])
    result = subprocess.run(
        args, capture_output=True, text=True, check=False, timeout=10
    )
    if variant == "complete":
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["empty_final_answer_ids"] == keys
    else:
        assert result.returncode == 2


def test_rejects_wrong_asset_without_creating_output(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/prepare_stabletoolbench_fixed_g1.py"
    )
    asset = tmp_path / "wrong"
    asset.write_bytes(b"wrong original bytes")
    output = tmp_path / "prepared"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--instructions",
            str(asset),
            "--toolenv",
            str(asset),
            "--cache",
            str(asset),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert "unexpected original asset bytes" in result.stderr
    assert not output.exists()


def test_fac_map_keeps_fixed_order_and_original_values(tmp_path, monkeypatch):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/prepare_stabletoolbench_fixed_g1.py"
    )
    main = runpy.run_path(str(script))["main"]
    original = tmp_path / "original.json"
    original.write_text(json.dumps([{"query_id": i} for i in [588, 608, 1073, 5]]))
    toolenv = tmp_path / "tools.tar"
    with tarfile.open(toolenv, "w") as archive:
        member = tarfile.TarInfo("toolenv2404_filtered/Data")
        member.type = tarfile.DIRTYPE
        archive.addfile(member)
    cache = tmp_path / "cache.zip"
    with zipfile.ZipFile(cache, "w") as archive:
        for name in main.__globals__["SUPPLEMENTS"]:
            archive.writestr(f"tools/Data/{name}", "{}")
    ids = tmp_path / "official.json"
    ids.write_text('{"5":0,"1073":3,"608":2,"588":1}')

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setitem(
        main.__globals__, "HASHES", tuple(map(digest, (original, toolenv, cache)))
    )
    monkeypatch.setitem(main.__globals__, "ID_MAP_HASH", digest(ids))
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--instructions",
            str(original),
            "--toolenv",
            str(toolenv),
            "--cache",
            str(cache),
            "--output",
            str(output),
            "--official-ids",
            str(ids),
        ],
    )
    main()
    assert list(json.loads((output / "fixed-g1-fac-ids.json").read_text()).items()) == [
        ("588", 1),
        ("608", 2),
        ("1073", 3),
    ]


def test_converter_null_function_call_patch_preserves_real_calls():
    patch = (
        Path(__file__).resolve().parents[1]
        / "configs/workload-native/stabletoolbench-null-function-call.patch"
    ).read_text()
    additions = [
        line[1:].strip()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert additions == ["if message.get('function_call') is not None:"]
    predicate = additions[0][3:-1]
    for message, expected in [
        ({"role": "assistant", "content": "text"}, False),
        ({"function_call": None, "tool_calls": [{"id": "native-tool"}]}, False),
        ({"function_call": {"name": "Finish", "arguments": "{}"}}, True),
    ]:
        assert eval(predicate, {"__builtins__": {}}, {"message": message}) is expected
