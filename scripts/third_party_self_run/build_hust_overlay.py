"""Build an isolated, source-bound GDN shape compatibility patch."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = Path(
    "/data/home/vllm-hust-codex-21rc/home/vllm-ascend-hust/vllm_ascend/ops/gdn.py"
)
folder = ROOT / "hust-compat"
folder.mkdir(exist_ok=True)
text = source.read_text()
if (
    hashlib.sha256(source.read_bytes()).hexdigest()
    != "819a424fa97a383d8f9c914f0ca40c186230e0b63f6289c7bd9e4b2e6fa82abc"
):
    raise ValueError("HUST GDN source checksum changed; refusing an unreviewed patch")
old = "z = z.reshape(z.size(0), -1, self.head_v_dim)"
new = "z = z.reshape(-1, self.num_v_heads // self.tp_size, self.head_v_dim)"
if text.count(old) != 2:
    raise ValueError("HUST GDN source changed; refusing an unreviewed replacement")
patched = text.replace(old, new)
(folder / "gdn.py").write_text(patched)
(folder / "sitecustomize.py").write_text("""import importlib.abc
import importlib.util
from pathlib import Path
import sys

class GDNCompatibilityFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm_ascend.ops.gdn":
            return importlib.util.spec_from_file_location(fullname, Path(__file__).with_name("gdn.py"))
        return None

sys.meta_path.insert(0, GDNCompatibilityFinder())
""")
receipt = {
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "patched_sha256": hashlib.sha256((folder / "gdn.py").read_bytes()).hexdigest(),
    "replacements": 2,
    "old": old,
    "new": new,
    "scope": "Equivalent [tokens, local_value_heads, head_dim] reshape; graph mode and arithmetic retained",
    "qualification": "requires component equivalence and full real-inference validation; not a passed benchmark result",
}
(folder / "patch.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
