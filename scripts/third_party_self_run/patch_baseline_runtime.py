"""Apply the CANN 9.1 host-header rename to the isolated 3.2.0 compiler helper."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = (
    ROOT
    / "runtime-b0/lib/python3.11/site-packages/triton/backends/ascend/npu_utils.cpp"
)
old = '\t{"WARP_STACK_SIZE", rtLimitType_t::RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE},\n'
alias = (
    '\t{"WARP_STACK_SIZE", rtLimitType_t::RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE},\n'
)
header = Path("/usr/local/Ascend/cann-9.1.0/pkg_inc/runtime/runtime/base.h")
text = path.read_text()
if "RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE = 2" not in header.read_text():
    raise ValueError("Unexpected CANN API; this patch is not applicable")
backup = path.with_suffix(".cpp.original-cann91")
if not backup.exists():
    backup.write_bytes(path.read_bytes())
original = backup.read_text()
if (
    hashlib.sha256(backup.read_bytes()).hexdigest()
    != "2d6e58513486c70e47a8a4149418f750f59568ffa9feb2c45ba3479016554f53"
):
    raise ValueError("Unexpected original compiler helper; freeze a new patch version")
if original.count(old) != 1:
    raise ValueError("Expected one unsupported legacy enum entry")
repaired = original.replace(old, "")
if text not in [original, original.replace(old, alias), repaired]:
    raise ValueError("Compiler helper has unrelated changes")
path.write_text(repaired)
receipt = {
    "file": str(path),
    "before_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
    "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
    "change": "Remove unsupported legacy WARP_STACK_SIZE selector; preserve the existing DVG_WARP_STACK_SIZE mapping. No inference arithmetic or metrics changed.",
}
(ROOT / "baseline-cann91-patch.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
