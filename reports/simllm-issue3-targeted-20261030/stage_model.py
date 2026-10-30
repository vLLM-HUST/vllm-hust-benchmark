import concurrent.futures
import hashlib
import json
import shutil
from pathlib import Path

source = Path("/data/jxd/Qwen3.5-35B-A3B-W8A8")
target = Path("/tmp/simllm-issue3-quality-oct30/model-local")
files = sorted(p for p in source.iterdir() if p.is_file())
assert not target.exists(), "Refusing to overwrite an existing model directory"
required = sum(p.stat().st_size for p in files)
assert shutil.disk_usage(target.parent).free > required + 20 * 1024**3
print("Staging bytes", required, "files", len(files), flush=True)
target.mkdir()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_one(path):
    before = path.stat()
    h = hashlib.sha256()
    out = target / path.name
    with path.open("rb") as src, out.open("xb") as dst:
        for chunk in iter(lambda: src.read(16 * 1024**2), b""):
            h.update(chunk)
            dst.write(chunk)
    assert (before.st_size, before.st_mtime_ns) == (
        path.stat().st_size,
        path.stat().st_mtime_ns,
    )
    assert sha(out) == h.hexdigest(), path.name
    result = {"name": path.name, "bytes": before.st_size, "sha256": h.hexdigest()}
    print("Verified", path.name, flush=True)
    return result


with concurrent.futures.ThreadPoolExecutor(8) as pool:
    entries = list(pool.map(copy_one, files))
manifest = {
    "source": str(source),
    "destination": str(target),
    "total_bytes": required,
    "files": entries,
    "verified": True,
}
Path("/tmp/simllm-issue3-quality-oct30/model-staging.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print("LOCAL_MODEL_VERIFIED", flush=True)
