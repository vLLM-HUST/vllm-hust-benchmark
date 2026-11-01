"""Original MARBLE KeyBERT operation on a pinned local NPU auxiliary model.

This is a tool backend, not a native Agent runner or benchmark score. Default
is plan-only. With --execute, stdin is exactly {"query": <original query>}.
The caller must allocate an idle NPU and coordinate the shared inference lease.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/marble-minilm-original-assets.json"


def verify_assets(model_root):
    folder = Path(model_root).absolute()
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("existing non-symlink model directory required")
    manifest = json.loads(PROFILE.read_text("utf-8"))
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe original model path")
        path = folder / relative
        if any(p.is_symlink() for p in (path, *path.parents) if p != folder.parent):
            raise ValueError("model asset symlinks forbidden")
        raw = path.read_bytes()
        if len(raw) != entry["bytes"]:
            raise ValueError("original model asset size mismatch")
        if entry["lfs_sha256"]:
            actual = hashlib.sha256(raw).hexdigest()
            expected = entry["lfs_sha256"]
        else:
            actual = hashlib.sha1(
                b"blob " + str(len(raw)).encode() + b"\0" + raw
            ).hexdigest()
            expected = entry["git_blob"]
        if actual != expected:
            raise ValueError("original model asset identity mismatch")
    return manifest


def read_query(stream):
    raw = stream.read(8 * 1024**2 + 1)
    if len(raw) > 8 * 1024**2:
        raise ValueError("request exceeds transport bound; not truncated")
    request = json.loads(raw)
    if not isinstance(request, dict) or set(request) != {"query"}:
        raise ValueError("exact original query field required")
    if not isinstance(request["query"], str):
        raise ValueError("original query must be a string")
    return request["query"]


def extract(model_root, query):
    manifest = verify_assets(model_root)
    import torch
    import torch_npu  # noqa: F401 - register the vendor device backend
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    torch.npu.set_device(0)
    model = SentenceTransformer(
        str(Path(model_root).absolute()),
        device="npu:0",
        local_files_only=True,
        trust_remote_code=False,
    )
    if not all(p.device.type == "npu" for p in model.parameters()):
        raise RuntimeError("CPU neural parameters forbidden")
    forwards = []

    def check(module, arguments, kwargs):
        tensors = []

        def collect(value):
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(arguments)
        collect(kwargs)
        if not tensors or any(t.device.type != "npu" for t in tensors):
            raise RuntimeError("CPU neural forward input forbidden")
        if any(p.device.type != "npu" for p in module.parameters()):
            raise RuntimeError("neural parameters moved off NPU")
        forwards.append(1)

    handle = model[0].auto_model.register_forward_pre_hook(check, with_kwargs=True)
    try:
        keywords = KeyBERT(model=model).extract_keywords(
            query, keyphrase_ngram_range=(1, 3), stop_words="english"
        )
        torch.npu.synchronize()
    finally:
        handle.remove()
    return {
        "status": "KEYBERT_TOOL_RETURNED_NOT_AGENT_RESULT",
        "model": manifest["model"],
        "revision": manifest["revision"],
        "keywords": keywords,
        "neural_forward_calls": len(forwards),
        "cpu_neural_fallback_used": False,
        "source_checkbox_eligible": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return 0
    query = read_query(sys.stdin.buffer)
    print(json.dumps(extract(args.model_root, query), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
