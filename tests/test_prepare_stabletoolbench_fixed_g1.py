from pathlib import Path
import hashlib
import json
import runpy
import subprocess
import sys
import tarfile
import zipfile


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
