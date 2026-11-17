"""SHA-256 helpers: files, directory trees, evidence bundles and model directories."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

CHUNK = 1 << 20

SMALL_EXT = (".json", ".txt", ".model", ".jinja", ".py", ".md")
BIG_EXT = (".safetensors", ".bin", ".pt", ".pth")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hash_tree(root, exclude_top: Iterable[str] = (), exclude_files: Iterable[str] = ()) -> Dict[str, dict]:
    """{relative_posix_path: {"sha256", "bytes"}} for every file under root.

    exclude_top: top-level directory names to skip entirely (e.g. "results").
    exclude_files: exact relative paths to skip (e.g. the seal files themselves).
    """
    root = Path(root)
    skip_top = set(exclude_top)
    skip_files = set(exclude_files)
    out: Dict[str, dict] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.split("/")[0] in skip_top or rel in skip_files:
            continue
        out[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def bundle_hash(files: Dict[str, dict]) -> str:
    """One hash over a file listing: sha256 of lines '<sha256>  <path>\\n' sorted by path."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(("%s  %s\n" % (files[rel]["sha256"], rel)).encode())
    return h.hexdigest()


def hash_model_dir(model_dir, small_only: bool = False, expected_file: Optional[str] = None,
                   progress: Optional[Callable[[str, str, float], None]] = None) -> dict:
    """Hash a model directory; optionally compare with an expected-hash file.

    expected_file format: {"files": [{"file": name, "sha256": hex}, ...]} (vllm-hust-benchmark model-hashes.json).
    small_only: hash config/tokenizer/index files only and record weights as skipped (seconds instead of hours).
    """
    model_dir = str(model_dir)
    names = sorted(n for n in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, n)))
    files: Dict[str, dict] = {}
    for n in names:
        p = os.path.join(model_dir, n)
        big = n.endswith(BIG_EXT)
        if big and small_only:
            files[n] = {"sha256": None, "bytes": os.path.getsize(p), "skipped": "small-only"}
            continue
        if not big and not n.endswith(SMALL_EXT):
            continue
        t0 = time.time()
        files[n] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
        if progress:
            progress(n, files[n]["sha256"], time.time() - t0)
    result = {"model_dir": model_dir, "small_only": small_only, "files": files}
    if expected_file:
        with open(expected_file) as fh:
            exp = {f["file"]: f["sha256"] for f in json.load(fh)["files"]}
        cmp_ = {}
        for n, want in exp.items():
            got = files.get(n, {}).get("sha256")
            cmp_[n] = "not_hashed" if got is None else ("match" if got == want else "MISMATCH")
        result["compare_with_expected"] = cmp_
        result["all_expected_matched"] = all(v == "match" for v in cmp_.values())
    return result
