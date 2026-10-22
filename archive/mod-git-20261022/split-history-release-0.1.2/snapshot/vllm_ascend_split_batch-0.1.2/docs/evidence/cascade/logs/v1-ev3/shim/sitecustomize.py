"""HARNESS-SIDE compatibility shim for evidence #3 (v1 baseline), NOT shipped code.

The cascade graph path is BROKEN on the new baseline: the plugin's monkeypatch
wrappers are written against the 0.23.0rc1 host signatures and three of them no
longer match vllm-hust v1 / vllm-ascend-hust main.  Observed failures, in the
order the engine hits them:

  1. cascade_runner_patch._capture_cudagraphs(self, batch_descriptors,
     cudagraph_runtime_mode)
        host: _capture_cudagraphs(..., profiler=None)
        -> TypeError: got an unexpected keyword argument 'profiler'
     (smoke_on_b64.log, sweep_on_graph_b64_r1.log)
  2. cascade_runner_patch._model_forward -> self._update_full_graph_params_if_needed(
        forward_context, num_tokens_padded, positions)
        host: _update_full_graph_params_if_needed(forward_context,
        num_tokens_padded)   [positions arg dropped]
        -> TypeError: takes 3 positional arguments but 4 were given
     (sweep_on_graph_shim_b64_a.log)
  3. cascade_graph_plugin.update_graph_params(..., num_dcp_pcp_tokens=None, ...)
        host: update_graph_params(update_stream, forward_context, num_tokens,
        vllm_config, speculative_config=None, draft_attn_metadatas=None)
        [num_dcp_pcp_tokens dropped]

This file is imported automatically in every interpreter (including the
EngineCore subprocess) via PYTHONPATH.  It rewrites ONLY those call signatures
in the plugin source *in memory*; the plugin tree on disk is untouched and no
host source is modified.  It exists so that the OFF-vs-ON performance
comparison can still be measured; numbers produced with it are labelled
"ON (shimmed)".

If you are reading a number produced with this shim, label it
"ON (shimmed signatures)".  The unshimmed ON leg is the defect.
"""
import re
import sys
from importlib.machinery import SourceFileLoader

RUNNER = "vllm_ascend_split_batch.cascade_runner_patch"
GRAPH = "vllm_ascend_split_batch.cascade_graph_plugin"

# --- 1. _capture_cudagraphs(profiler) -------------------------------------
R1_OLD_SIG = (
    "def _capture_cudagraphs(self, batch_descriptors, cudagraph_runtime_mode):"
)
R1_NEW_SIG = (
    "def _capture_cudagraphs(self, batch_descriptors, cudagraph_runtime_mode, "
    "profiler=None):"
)
R1_OLD_CALL = "orig(self, batch_descriptors, cudagraph_runtime_mode)"
R1_NEW_CALL = (
    "orig(self, batch_descriptors, cudagraph_runtime_mode, profiler=profiler)"
)

# --- 2. _update_full_graph_params_if_needed(positions) --------------------
R2_OLD = (
    "self._update_full_graph_params_if_needed(\n"
    "                forward_context, num_tokens_padded, positions\n"
    "            )"
)
R2_NEW = (
    "self._update_full_graph_params_if_needed(\n"
    "                forward_context, num_tokens_padded\n"
    "            )"
)

# --- 3. update_graph_params(num_dcp_pcp_tokens) ---------------------------
# Drop the whole line wherever it is the standalone parameter
# (def default, or a positional argument at either call site).
R3_RE = re.compile(r"^[ \t]*num_dcp_pcp_tokens(?:=None)?,\n", re.MULTILINE)


class _ShimLoader(SourceFileLoader):
    def get_code(self, fullname):
        # Bypass __pycache__: a stale .pyc silently restores the unshimmed
        # signature (observed on the first attempt).
        return self.source_to_code(self.get_source(fullname),
                                   self.get_filename(fullname))

    def source_to_code(self, data, path, *, _optimize=-1):
        src = data.decode("utf-8") if isinstance(data, bytes) else data
        n = 0
        if fullname_endswith(path, "cascade_runner_patch.py"):
            for old, new in ((R1_OLD_SIG, R1_NEW_SIG),
                             (R1_OLD_CALL, R1_NEW_CALL),
                             (R2_OLD, R2_NEW)):
                if old in src:
                    src = src.replace(old, new)
                    n += 1
        elif fullname_endswith(path, "cascade_graph_plugin.py"):
            src, k = R3_RE.subn("", src)
            n += k
        if n:
            print(f"[ev3-shim] applied {n} signature fix(es) to {path}",
                  flush=True)
        return compile(src, path, "exec", dont_inherit=True, optimize=_optimize)


def fullname_endswith(path, suffix):
    return str(path).endswith(suffix)


from importlib.machinery import PathFinder  # noqa: E402


class _ShimFinder(PathFinder):
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname not in (RUNNER, GRAPH):
            return None
        spec = super().find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        if isinstance(loader, SourceFileLoader):
            spec.loader = _ShimLoader(loader.name, loader.path)
        return spec


if not any(getattr(f, "__name__", "") == "_ShimFinder" for f in sys.meta_path):
    sys.meta_path.insert(0, _ShimFinder)
