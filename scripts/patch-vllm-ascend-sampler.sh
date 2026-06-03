# Patch sampler.py to gracefully fall back to PyTorch implementation when the
# custom C++ operator (npu_apply_top_k_top_p) is not available.
# This happens when vllm-ascend C++ extensions have not been compiled.
patch_vllm_ascend_sampler_for_missing_cpp_ops() {
  local sampler_py="$OFFICIAL_VLLM_ASCEND_WORKTREE/vllm_ascend/sample/sampler.py"
  if [[ ! -f "$sampler_py" ]]; then
    return 0
  fi

  # Check if already patched
  if grep -q "_apply_top_k_top_p_ascendc_available" "$sampler_py" 2>/dev/null; then
    echo "[official-env] sampler.py already patched for missing C++ ops"
    return 0
  fi

  # Use Python to patch the file reliably
  python3 - "$sampler_py" <<'PATCH_SCRIPT'
import sys
import re

sampler_path = sys.argv[1]
with open(sampler_path, "r") as f:
    content = f.read()

# Find and replace the apply_top_k_top_p selection logic
old_pattern = r"""apply_top_k_top_p = \(
    _apply_top_k_top_p_ascendc
    if get_ascend_device_type\(\) in \[AscendDeviceType\.A2, AscendDeviceType\.A3\]
    else _apply_top_k_top_p_pytorch
\)"""

new_code = """_apply_top_k_top_p_ascendc_available = (
    get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
    and hasattr(torch.ops._C_ascend, "npu_apply_top_k_top_p")
)

apply_top_k_top_p = (
    _apply_top_k_top_p_ascendc
    if _apply_top_k_top_p_ascendc_available
    else _apply_top_k_top_p_pytorch
)"""

if re.search(old_pattern, content):
    content = re.sub(old_pattern, new_code, content)
    with open(sampler_path, "w") as f:
        f.write(content)
    print("[official-env] patched sampler.py to check for npu_apply_top_k_top_p availability")
else:
    print("[official-env] warning: could not find pattern to patch in sampler.py", file=sys.stderr)
PATCH_SCRIPT
}
