"""Exercise real model config through autoconfig without loading the model."""

import hashlib
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

from vllm_moe_offload_ascend.moe_offload import autoconfig

os.environ["VLLM_ASCEND_MOE_OFFLOAD_GB"] = "14"
args = SimpleNamespace(
    model="/root/qwen35-discovery-20261009/model",
    cpu_offload_gb=0,
    offload_backend="auto",
)
result = {
    "diagnostic_only": True,
    "official_quality_scores": False,
    "no_model_constructed": True,
    "source_path": autoconfig.__file__,
    "source_sha256": hashlib.sha256(Path(autoconfig.__file__).read_bytes()).hexdigest(),
}
try:
    result["applied"] = autoconfig.apply_moe_offload_defaults(args)
    result["plan"] = args._ascend_moe_offload_autoconfig_plan
    result["status"] = "config-planning-passed-not-npu-qualified"
except Exception as exc:
    result.update(
        status="config-planning-failed",
        error_type=type(exc).__name__,
        error=str(exc),
        traceback=traceback.format_exc(),
    )
print(json.dumps(result, indent=2))
