"""Build a runtime-contract.json: a description of the serving setup a plan or run was made against.

It can be embedded into a plan with `prepare --solver-runtime-contract` (stored under `runtime`, with its
SHA-256), so the plan records which model revision, engine versions, command and environment it was meant for.

For a run without any compression method, `method_config_sha256` is the SHA-256 of the profile file (the thing
that defines the treatment-free configuration); that is written into the file itself.

The profile is only DATA describing the server somebody else runs. This module never starts anything.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Optional

from . import config_check
from .hashing import sha256_file

PACKAGES = ["vllm", "vllm_ascend", "vllm-hust-ext", "torch", "torch_npu", "transformers", "pyarrow", "triton-ascend"]


def build(profile_path, out_path, image_id: str = "NOT_PROVIDED", npu_smi_text: Optional[str] = None) -> dict:
    out = Path(out_path)
    if out.exists():
        raise FileExistsError("refusing to overwrite %s" % out)
    prof = json.loads(Path(profile_path).read_text(encoding="utf8"))
    problems = config_check.profile_problems(prof)
    if problems:
        raise ValueError("profile has undecided fields: %s" % problems)
    pk = {}
    for p in PACKAGES:
        try:
            pk[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pk[p] = "NOT_INSTALLED"
    if npu_smi_text is None:
        try:
            npu_smi_text = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30).stdout
        except Exception as e:  # noqa: BLE001
            npu_smi_text = "UNAVAILABLE: %r" % (e,)
    contract = {
        "note": "Built from profile %s for a treatment-free (no KV compression) run. method_config_sha256 = "
                "sha256(profile file), NOT a compression config. Package versions are those of the machine running "
                "THIS command; run it on the machine that hosts the server or they describe the wrong host."
                % prof.get("profile_id"),
        "model_revision": prof["model"].get("expected_revision") or "UNVERIFIED",
        "model_revision_verified": False,
        "container_image_digest": image_id,
        "sources": {"vllm": pk["vllm"], "vllm_ascend": pk["vllm_ascend"], "profile_id": prof.get("profile_id")},
        "packages": pk,
        "command": list(prof.get("launcher", [])) + [prof["model"]["path"]] + list(prof.get("serve_args", [])),
        "hardware": {"npu_smi_sha256": hashlib.sha256(npu_smi_text.encode()).hexdigest(),
                     "npu_smi_head": npu_smi_text[:600]},
        "method_config_sha256": sha256_file(profile_path),
        "environment": prof.get("env") or {"NONE": "none"},
    }
    out.write_text(json.dumps(contract, indent=1, ensure_ascii=False), encoding="utf8")
    return contract
