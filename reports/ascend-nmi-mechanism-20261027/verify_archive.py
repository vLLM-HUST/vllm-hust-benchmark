"""Verify all archived bytes before optional extraction. Does not execute the archive."""
import argparse, base64, hashlib, io, json, tarfile
from pathlib import Path
def digest(data):
    return hashlib.sha256(data).hexdigest()
def verify(root, extract=None):
    manifest=json.loads((root/"manifest.json").read_text())
    encoded=(root/manifest["archive"]["encoded_path"]).read_bytes()
    assert digest(encoded)==manifest["archive"]["encoded_sha256"],"encoded capsule changed"
    payload=base64.b64decode(encoded,validate=False)
    assert len(payload)==manifest["archive"]["bytes"]
    assert digest(payload)==manifest["archive"]["sha256"],"capsule changed"
    expected={x["path"]:x for x in manifest["members"]}
    seen=set()
    with tarfile.open(fileobj=io.BytesIO(payload),mode="r:gz") as tar:
        for member in tar:
            assert member.isfile(),"only regular files permitted"
            path=Path(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert member.name in expected and member.name not in seen
            data=tar.extractfile(member).read()
            spec=expected[member.name]
            assert len(data)==spec["bytes"] and digest(data)==spec["sha256"],member.name
            seen.add(member.name)
            if extract is not None:
                target=extract/path
                target.parent.mkdir(parents=True,exist_ok=True)
                assert not target.exists(),"refuse to overwrite extracted evidence"
                target.write_bytes(data)
    assert seen==set(expected),"incomplete archive"
    for file in manifest["derived_files"]:
        p=root/file["path"]
        assert digest(p.read_bytes())==file["sha256"],str(p)
    return {"verified_members":len(seen),"archive_sha256":manifest["archive"]["sha256"]}
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract",type=Path)
    args=parser.parse_args()
    if args.extract is not None:
        assert not args.extract.exists(),"fresh extraction directory required"
    print(json.dumps(verify(Path(__file__).resolve().parent,args.extract),indent=2))
