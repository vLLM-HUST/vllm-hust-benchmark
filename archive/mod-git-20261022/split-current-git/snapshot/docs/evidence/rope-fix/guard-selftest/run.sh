#!/usr/bin/env bash
# Three-way self-test of the drift guard, WITHOUT touching the host tree.
#
# The guard reads the host sources through VLLM_ASCEND_HUST_ROOT (or the
# editable-install path).  Here the two host files are copied into /tmp and
# mutated, so the expected failure wording can be observed live:
#
#   fixed    -> inject "upstream already fixed the defect" into all three anchors
#   drifted  -> rename/move the anchors instead
#   no-host  -> make the host tree undiscoverable: the guard must FAIL, not skip
#
# Usage: bash run.sh        (from this directory; needs the repo's pytest/ruff)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/vllm-workspace/vllm-ascend-split-batch-hust
HOST=/vllm-workspace/vllm-ascend-hust/vllm_ascend
FIXED=/tmp/ropefix-drift-sim
DRIFTED=/tmp/ropefix-drift-sim2
NOHOST=/tmp/ropefix-nohost

seed_tree() {
  local root=$1
  rm -rf "$root"
  mkdir -p "$root/vllm_ascend/ops"
  cp "$HOST/utils.py" "$root/vllm_ascend/utils.py"
  cp "$HOST/ops/rotary_embedding.py" "$root/vllm_ascend/ops/rotary_embedding.py"
}

seed_tree "$FIXED"
python - "$FIXED" <<'PY'
import sys
from pathlib import Path
sim = Path(sys.argv[1]) / "vllm_ascend"
u = sim / "utils.py"
t = u.read_text()
old = '        "YaRNScalingRotaryEmbedding": AscendYaRNRotaryEmbedding,'
assert old in t
u.write_text(t.replace(old, old + '\n        "Llama3RotaryEmbedding": AscendYaRNRotaryEmbedding,', 1))
r = sim / "ops" / "rotary_embedding.py"
t = r.read_text()
old2 = "            self.mrope_interleaved,\n        )"
assert old2 in t
t = t.replace(old2, "            self.mrope_interleaved,\n            self.is_neox_style,\n        )", 1)
old3 = "        truncate: bool = False,"
assert old3 in t
r.write_text(t.replace(old3, "        truncate: bool = True,", 1))
PY

seed_tree "$DRIFTED"
python - "$DRIFTED" <<'PY'
import sys
from pathlib import Path
sim = Path(sys.argv[1]) / "vllm_ascend"
u = sim / "utils.py"
t = u.read_text()
old = "    REGISTERED_ASCEND_OPS = {"
assert old in t
u.write_text(t.replace(old, "    _REGISTERED = {", 1))
r = sim / "ops" / "rotary_embedding.py"
t = r.read_text()
old1 = "class AscendMRotaryEmbedding(MRotaryEmbedding):"
old2 = "        truncate: bool = False,\n    ) -> None:"
assert old1 in t and old2 in t
t = t.replace(old1, "class AscendMRotaryEmbedding2(MRotaryEmbedding):", 1)
r.write_text(t.replace(old2, "    ) -> None:", 1))
PY

cd "$REPO" || exit 1
echo "### case=fixed  VLLM_ASCEND_HUST_ROOT=$FIXED"
VLLM_ASCEND_HUST_ROOT=$FIXED python -m pytest tests/test_rope_fix_drift.py -q \
  > "$HERE/selftest-fixed.txt" 2>&1
grep -E "upstream fixed|anchor drifted|passed|failed" "$HERE/selftest-fixed.txt"

echo "### case=drifted  VLLM_ASCEND_HUST_ROOT=$DRIFTED"
VLLM_ASCEND_HUST_ROOT=$DRIFTED python -m pytest tests/test_rope_fix_drift.py -q \
  > "$HERE/selftest-drifted.txt" 2>&1
grep -E "upstream fixed|anchor drifted|passed|failed" "$HERE/selftest-drifted.txt"

# Missing evidence must be red: copy the guard next to *no* host tree and hide
# the editable install from find_spec (a sitecustomize on PYTHONPATH), so the
# guard cannot locate the anchors.  A skip here would look like a passing guard.
echo "### case=no-host  (host tree undiscoverable)"
rm -rf "$NOHOST"
mkdir -p "$NOHOST/tests"
cp "$REPO/tests/test_rope_fix_drift.py" "$NOHOST/tests/"
cat > "$NOHOST/sitecustomize.py" <<'PY'
import importlib.util

_orig = importlib.util.find_spec


def find_spec(name, *args, **kwargs):
    if name == "vllm_ascend":
        raise ModuleNotFoundError("simulated: no vllm-ascend checkout on this box")
    return _orig(name, *args, **kwargs)


importlib.util.find_spec = find_spec
PY
PYTHONPATH=$NOHOST python -m pytest "$NOHOST/tests/test_rope_fix_drift.py" -q -p no:cacheprovider \
  > "$HERE/selftest-no-host.txt" 2>&1
grep -E "source tree not found|skipped|passed|failed|error" "$HERE/selftest-no-host.txt" | head -5
