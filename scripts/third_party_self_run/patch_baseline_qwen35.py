"""Keep Qwen3.5's compiled text-model arguments consistent in vLLM 0.18."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = (
    ROOT
    / "runtime-b0/lib/python3.11/site-packages/vllm/model_executor/models/qwen3_5.py"
)
backup = path.with_suffix(".py.original-text-graph")
original_hash = "76eeda13ad5d104b74ce8352e4a9766a0a93f29230ba359e9f7727aef4173f1d"
old = """        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
"""
new = """        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            # Normalize text-only execution to the embedding-based graph warmup.
            inputs_embeds = self.language_model.model.embed_input_ids(input_ids)
        if inputs_embeds is not None:
            input_ids = None

        hidden_states = self.language_model.model(
"""
current = path.read_text()
if not backup.exists():
    if hashlib.sha256(path.read_bytes()).hexdigest() != original_hash:
        raise ValueError("Unexpected Qwen3.5 source; refusing an unreviewed patch")
    backup.write_bytes(path.read_bytes())
original = backup.read_text()
if (
    hashlib.sha256(backup.read_bytes()).hexdigest() != original_hash
    or original.count(old) != 1
):
    raise ValueError("Unexpected original Qwen3.5 source")
patched = original.replace(old, new)
if current not in [original, patched]:
    raise ValueError("Qwen3.5 source has unrelated edits")
path.write_text(patched)
receipt = {
    "file": str(path),
    "before_sha256": original_hash,
    "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "change": "Materialize missing text embeddings before the compiled language model, using its original embedding layer; normalize input_ids to None when embeddings are present. Graph execution and model arithmetic retained.",
    "qualification": "Compatibility-modified B0; not an untouched upstream benchmark. Requires full native validation.",
}
(ROOT / "baseline-qwen35-patch.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
