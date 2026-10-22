"""Source-level (AST) signature audit: plugin monkeypatch targets vs host.

Existence checks alone missed parameter drift; this compares the actual
parameter lists of every method the cascade plugin wraps.
"""
import ast
from pathlib import Path

HOST_VLLM = Path("/vllm-workspace/vllm-hust/vllm")
HOST_ASCEND = Path("/vllm-workspace/vllm-ascend-hust/vllm_ascend")
PLUGIN = Path("/vllm-workspace/vllm-ascend-split-batch-hust/src/vllm_ascend_split_batch")


def find_defs(path: Path, name: str):
    """Yield (lineno, params) for every `def name(...)` in the file."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            params = []
            for a in node.args.args:
                params.append(a.arg)
            if node.args.vararg:
                params.append("*" + node.args.vararg.arg)
            for a in node.args.kwonlyargs:
                params.append(a.arg)
            if node.args.kwarg:
                params.append("**" + node.args.kwarg.arg)
            out.append((node.lineno, params))
    return out


def show(label: str, path: Path, name: str):
    for lineno, params in find_defs(path, name):
        print(f"  {label}:{lineno}  {name}({', '.join(params)})")


print("=== HOST (vllm-hust v1 + vllm-ascend-hust main) ===")
show("gpu_model_runner", HOST_VLLM / "v1/worker/gpu_model_runner.py", "_capture_cudagraphs")
show("gpu_model_runner", HOST_VLLM / "v1/worker/gpu_model_runner.py", "_warmup_and_capture")
show("gpu_model_runner", HOST_VLLM / "v1/worker/gpu_model_runner.py", "_determine_batch_execution_and_padding")
show("gpu_model_runner", HOST_VLLM / "v1/worker/gpu_model_runner.py", "_model_forward")
show("model_runner_v1", HOST_ASCEND / "worker/model_runner_v1.py", "_update_full_graph_params_if_needed")
show("attention_v1", HOST_ASCEND / "attention/attention_v1.py", "update_graph_params")
show("acl_graph", HOST_ASCEND / "compilation/acl_graph.py", "update_full_graph_params")

print("\n=== PLUGIN (cascade wrappers) ===")
show("cascade_runner_patch", PLUGIN / "cascade_runner_patch.py", "_capture_cudagraphs")
show("cascade_runner_patch", PLUGIN / "cascade_runner_patch.py", "_determine_batch_execution_and_padding")
show("cascade_runner_patch", PLUGIN / "cascade_runner_patch.py", "_model_forward")
show("cascade_graph_plugin", PLUGIN / "cascade_graph_plugin.py", "update_graph_params")

print("\n=== HOST call sites (how many positional args are passed) ===")
def call_sites(path: Path, name: str):
    src = path.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            target = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
            if target == name:
                kw = [k.arg for k in node.keywords]
                print(f"  {path.name}:{node.lineno}  {name}(pos={len(node.args)}, kw={kw})")

call_sites(HOST_VLLM / "v1/worker/gpu_model_runner.py", "_capture_cudagraphs")
call_sites(HOST_VLLM / "v1/worker/gpu_model_runner.py", "_update_full_graph_params_if_needed")
call_sites(HOST_ASCEND / "worker/model_runner_v1.py", "_update_full_graph_params_if_needed")
call_sites(HOST_ASCEND / "compilation/acl_graph.py", "update_graph_params")
