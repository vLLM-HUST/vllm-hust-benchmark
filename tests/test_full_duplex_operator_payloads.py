import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "duplex_operator_payload", ROOT / "scripts/prepare_full_duplex_operator_payloads.py"
)
ENTRY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ENTRY)


def tar_payload(name="packages/vendors/fixture/lib.so", link=None):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo(name)
        data = b"non-neural operator file fixture"
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        if link:
            alias = tarfile.TarInfo("packages/vendors/fixture/alias.so")
            alias.type = tarfile.SYMTYPE
            alias.linkname = link
            archive.addfile(alias)
    return stream.getvalue()


def wrapper(payload):
    header = (
        "#!/bin/bash\n# This script was generated using Makeself 2.5.0\n"
        f'SHA="{hashlib.sha256(payload).hexdigest()}"\n'
        f'skip="6"\nfilesizes="{len(payload)}"\ntotalsize="{len(payload)}"\n'
    )
    return header.encode() + payload


def test_literal_header_is_parsed_without_running_shell():
    payload = tar_payload()
    assert ENTRY.embedded_tar(wrapper(payload)) == payload


@pytest.mark.parametrize("change", ["checksum", "truncated", "duplicate", "version"])
def test_rejects_wrong_wrapper_contract(change):
    data = wrapper(tar_payload())
    if change == "checksum":
        data = data[:-1] + bytes([data[-1] ^ 1])
    elif change == "truncated":
        data = data[:-1]
    elif change == "duplicate":
        data = data.replace(b'skip="6"', b'skip="6"\nskip="6"')
    else:
        data = data.replace(b"Makeself 2.5.0", b"Makeself unknown")
    with pytest.raises(ValueError):
        ENTRY.embedded_tar(data)


def test_regular_payload_is_extracted_and_hash_bound(tmp_path):
    output = tmp_path / "out"
    files = ENTRY.extract_regular_payload(tar_payload(), output)
    assert len(files) == 1
    target = output / files[0]["path"]
    assert ENTRY.digest(target) == files[0]["sha256"]


@pytest.mark.parametrize(
    "name,link",
    [
        ("../escape", None),
        ("/absolute", None),
        ("safe", "../../outside"),
        ("safe", "/outside"),
        ("safe", "missing"),
    ],
)
def test_rejects_escaping_or_unresolved_members_before_creating_output(
    tmp_path, name, link
):
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        ENTRY.extract_regular_payload(tar_payload(name, link), output)
    assert not output.exists()


def test_contained_library_link_preserved_on_linux(tmp_path):
    if sys.platform == "win32":
        pytest.skip("real symlink creation is qualified on server/Linux CI")
    output = tmp_path / "out"
    files = ENTRY.extract_regular_payload(tar_payload(link="lib.so"), output)
    alias = output / "packages/vendors/fixture/alias.so"
    assert alias.is_symlink()
    assert alias.read_bytes() == b"non-neural operator file fixture"
    assert len(files) == 2
