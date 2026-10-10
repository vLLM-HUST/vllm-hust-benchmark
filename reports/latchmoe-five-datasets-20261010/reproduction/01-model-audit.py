"""Read-only source/model audit; no model construction or NPU allocation."""

import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

ROOT = Path("/root/latchmoe-five-datasets-20261010")
MODEL = Path("/root/qwen35-discovery-20261009/model")
LOCK = MODEL.parent / "model_lock.json"
lock = json.loads(LOCK.read_text())
config = json.loads((MODEL / "config.json").read_text())
totals = {}
tensor_names = set()
files_by_name = {}


def verify_file(entry):
    path = MODEL / entry["name"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    verified = digest == entry["sha256"] and path.stat().st_size == entry["size"]
    item = dict(
        name=entry["name"], bytes=path.stat().st_size, sha256=digest, verified=verified
    )
    return item


with ThreadPoolExecutor(max_workers=4) as pool:
    for future in as_completed(
        [pool.submit(verify_file, entry) for entry in lock["files"]]
    ):
        item = future.result()
        files_by_name[item["name"]] = item
        print(json.dumps(item), flush=True)

files = [files_by_name[entry["name"]] for entry in lock["files"]]
prefix_bytes = {}
for entry in lock["files"]:
    path = MODEL / entry["name"]
    if path.suffix != ".safetensors":
        continue
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_size < 32 * 1024 * 1024:
            raise ValueError("Invalid safetensors header")
        header = json.loads(stream.read(header_size))
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if name in tensor_names:
            raise ValueError("Duplicate tensor across shards")
        tensor_names.add(name)
        prefix = ".".join(name.split(".")[:2])
        prefix_bytes[prefix] = prefix_bytes.get(prefix, 0) + (
            tensor["data_offsets"][1] - tensor["data_offsets"][0]
        )
        group = (
            "vision"
            if name.startswith("model.visual.")
            else ("mtp" if ".mtp." in name else "language_including_unclassified")
        )
        totals[group] = totals.get(group, 0) + (
            tensor["data_offsets"][1] - tensor["data_offsets"][0]
        )

package = Path(
    "/root/latchmoe-env/lib/python3.12/site-packages/vllm_moe_offload_ascend"
)
capability_file = package / "moe_offload/capabilities.py"
spec = importlib.util.spec_from_file_location("audited_capabilities", capability_file)
cap = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cap
spec.loader.exec_module(cap)
descriptor = cap.describe_checkpoint_config(config["text_config"])
# Static TP2 guard reproduction; router fields deliberately stay unresolved.
# This is not a materialized-layer qualification or a model inference result.
tp2 = dataclasses.replace(descriptor, parallel_mode="multi_npu")
output = {
    "kind": "read_only_model_and_capability_audit",
    "model": lock["repo"],
    "revision": lock["revision"],
    "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
    "all_files_verified": all(x["verified"] for x in files),
    "files": files,
    "tensor_payload_bytes": totals,
    "tensor_prefix_payload_bytes": prefix_bytes,
    "tensor_names": len(tensor_names),
    "architectures": config["architectures"],
    "config": config["text_config"],
    "capability_source_sha256": hashlib.sha256(
        capability_file.read_bytes()
    ).hexdigest(),
    "checkpoint_descriptor": descriptor.to_jsonable(),
    "checkpoint_support_unresolved": cap.evaluate_support(descriptor).to_jsonable(),
    "static_tp2_guard": cap.evaluate_support(tp2).to_jsonable(),
    "diagnostic_only": True,
    "official_quality_scores": False,
    "no_model_constructed": True,
}
(ROOT / "evidence/model-audit.json").write_text(json.dumps(output, indent=2) + "\n")
print(
    json.dumps({k: v for k, v in output.items() if k not in ("files", "config")}),
    flush=True,
)
if not output["all_files_verified"]:
    raise SystemExit(1)
