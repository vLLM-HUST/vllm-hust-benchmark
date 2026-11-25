"""Compare an expected serving configuration with what can be observed from the CLIENT side.

Nothing here launches a server. Inputs are things a client can obtain:
  * a profile JSON (the configuration somebody INTENDED; used as the expected values),
  * optionally the server's actual argv (e.g. /proc/<pid>/cmdline saved by whoever runs the server),
  * optionally the server's startup log text,
  * optionally the response of GET /v1/models.

Per field the status is
  verified     argv and log both agree with the expectation
  log_only     argv not provided; the log agrees
  argv_only    log did not expose the field; argv agrees (a command-line fact, not proof of the engine's value)
  unverified   neither source exposed the field
  mismatch     a source disagrees with the expectation

Log patterns were calibrated on one real vLLM 0.28.1.post1.dev143 / vllm-ascend startup log (2026-10-12).
Other versions may print differently: fields then show up as argv_only / unverified, never as verified.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def parse_flags(args: List[str]) -> Dict[str, Any]:
    """['--a','1','--b','--c','x','y'] -> {'a':'1','b':True,'c':['x','y']}. Tokens before the first flag are ignored."""
    d: Dict[str, Any] = {}
    i = 0
    while i < len(args):
        t = args[i]
        if t.startswith("--"):
            k = t[2:]
            if "=" in k:
                k, v = k.split("=", 1)
                d[k] = v
                i += 1
                continue
            vals = []
            j = i + 1
            while j < len(args) and not args[j].startswith("--"):
                vals.append(args[j])
                j += 1
            d[k] = True if not vals else (vals[0] if len(vals) == 1 else vals)
            i = j
        else:
            i += 1
    return d


def _bool_flag(d, name):
    if ("no-" + name) in d:
        return False
    if name in d:
        return True
    return None


def _int(v, default=None):
    return int(v) if v not in (None, True) else default


def effective_from_argv(args: List[str]) -> Dict[str, Any]:
    d = parse_flags(args)
    comp = json.loads(d["compilation-config"]) if isinstance(d.get("compilation-config"), str) else {}
    sizes = d.get("cudagraph-capture-sizes")
    if sizes is None:
        sizes = comp.get("cudagraph_capture_sizes")
    if isinstance(sizes, str):
        sizes = [sizes]
    sizes = [int(x) for x in sizes] if sizes is not None else None
    spec = json.loads(d["speculative-config"]) if isinstance(d.get("speculative-config"), str) else None
    return {
        "dtype": d.get("dtype"),
        "tensor_parallel_size": _int(d.get("tensor-parallel-size"), 1),
        "pipeline_parallel_size": _int(d.get("pipeline-parallel-size"), 1),
        "data_parallel_size": _int(d.get("data-parallel-size"), 1),
        "enable_expert_parallel": bool(_bool_flag(d, "enable-expert-parallel")),
        "decode_context_parallel_size": _int(d.get("decode-context-parallel-size")),
        "disable_custom_all_reduce": bool(_bool_flag(d, "disable-custom-all-reduce")),
        "gpu_memory_utilization": float(d["gpu-memory-utilization"]) if "gpu-memory-utilization" in d else None,
        "max_model_len": _int(d.get("max-model-len")),
        "max_num_seqs": _int(d.get("max-num-seqs")),
        "max_num_batched_tokens": _int(d.get("max-num-batched-tokens")),
        "enable_prefix_caching": _bool_flag(d, "enable-prefix-caching"),
        "enable_chunked_prefill": _bool_flag(d, "enable-chunked-prefill"),
        "speculative_config": spec,
        "enforce_eager": _bool_flag(d, "enforce-eager"),
        "compilation_mode": comp.get("mode"),
        "cudagraph_mode": comp.get("cudagraph_mode"),
        "cudagraph_capture_sizes": sizes,
    }


_LOG_PATTERNS = {
    "dtype": [r"\bdtype=(?:torch\.)?(\w+)"],
    "tensor_parallel_size": [r"\btensor_parallel_size=(\d+)"],
    "pipeline_parallel_size": [r"\bpipeline_parallel_size=(\d+)"],
    "data_parallel_size": [r"\bdata_parallel_size=(\d+)"],
    "decode_context_parallel_size": [r"\bdecode_context_parallel_size=(\d+)"],
    "enable_expert_parallel": [r"\benable_expert_parallel=(True|False)", r"'enable_expert_parallel':\s*(True|False)"],
    "disable_custom_all_reduce": [r"\bdisable_custom_all_reduce=(True|False)"],
    "gpu_memory_utilization": [r"\bgpu_memory_utilization=([\d.]+)", r"Desired GPU memory utilization is \(([\d.]+),",
                               r"'gpu_memory_utilization':\s*([\d.]+)"],
    "max_model_len": [r"\b(?:max_seq_len|max_model_len)=(\d+)"],
    "max_num_seqs": [r"\bmax_num_seqs=(\d+)", r"'max_num_seqs':\s*(\d+)"],
    "max_num_batched_tokens": [r"\bmax_num_batched_tokens=(\d+)"],
    "enable_prefix_caching": [r"\benable_prefix_caching=(True|False)"],
    "enable_chunked_prefill": [r"\b(?:enable_chunked_prefill|chunked_prefill_enabled)=(True|False)"],
    "enforce_eager": [r"\benforce_eager=(True|False)"],
    "compilation_mode": [r"'mode':\s*<CompilationMode\.\w+:\s*(\d+)>"],
    "cudagraph_mode": [r"\bcudagraph_mode\W{1,4}(?:CUDAGraphMode\.)?([A-Z_]+)"],
    "cudagraph_capture_sizes": [r"\bcudagraph_capture_sizes\W{1,4}\[([\d,\s]+)\]"],
    "speculative_config": [r"\bspeculative_config=(\S+?)[,)]"],
}
_BOOL_KEYS = ("enable_expert_parallel", "disable_custom_all_reduce", "enable_prefix_caching",
              "enable_chunked_prefill", "enforce_eager")
_INT_KEYS = ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "max_model_len",
             "max_num_seqs", "max_num_batched_tokens", "decode_context_parallel_size", "compilation_mode")


def _norm_log(key, raw):
    if key in _BOOL_KEYS:
        return raw == "True"
    if key in _INT_KEYS:
        return int(raw)
    if key == "gpu_memory_utilization":
        return float(raw)
    if key == "cudagraph_capture_sizes":
        return [int(x) for x in re.findall(r"\d+", raw)]
    if key == "speculative_config":
        return None if raw == "None" else raw
    return raw


def effective_from_log(text: str) -> Dict[str, tuple]:
    """{key: (value, matched_snippet)} for each key found in the log; the first matching pattern wins."""
    out = {}
    for key, pats in _LOG_PATTERNS.items():
        for pat in pats:
            m = re.search(pat, text)
            if m:
                out[key] = (_norm_log(key, m.group(1)), m.group(0)[:160])
                break
    if "enable_expert_parallel" not in out and re.search(r"Expert parallelism is enabled", text):
        out["enable_expert_parallel"] = (True, "Expert parallelism is enabled")
    return out


def _eq(key, a, b):
    if key == "decode_context_parallel_size":  # unset on the command line == engine default 1
        return (1 if a is None else a) == (1 if b is None else b)
    if key == "gpu_memory_utilization" and a is not None and b is not None:
        return abs(float(a) - float(b)) < 1e-9
    if key == "cudagraph_capture_sizes" and a is not None and b is not None:
        return sorted(a) == sorted(b)
    return a == b


def load_argv(path: str) -> List[str]:
    """Read an argv saved as a JSON list, or as text separated by NULs / newlines (/proc/<pid>/cmdline)."""
    with open(path, "rb") as f:
        raw = f.read().decode("utf8", "replace")
    s = raw.strip()
    if s.startswith("["):
        return [str(x) for x in json.loads(s)]
    sep = "\0" if "\0" in raw else "\n"
    return [x for x in raw.split(sep) if x]


def profile_problems(profile: dict) -> List[str]:
    bad = []
    for k, v in profile.get("expect", {}).items():
        if v is None and k not in ("decode_context_parallel_size", "speculative_config"):
            bad.append("expect.%s is null (undecided)" % k)
    return bad


def verify(profile: dict, argv: Optional[List[str]] = None, startup_log_text: Optional[str] = None,
           models_json: Optional[dict] = None) -> dict:
    expect = profile["expect"]
    argv_eff = effective_from_argv(argv) if argv else None
    log_eff = effective_from_log(startup_log_text) if startup_log_text else {}
    fields = {}
    for key, exp in expect.items():
        entry: Dict[str, Any] = {"expected": exp}
        ok_argv: Optional[bool] = None
        if argv_eff is not None:
            a = argv_eff.get(key)
            entry["argv"] = a
            ok_argv = _eq(key, a, exp)
            if key in ("enable_prefix_caching", "enable_chunked_prefill", "enforce_eager") and a is None:
                ok_argv = False  # an engine default is not an explicit setting
        else:
            entry["argv"] = "not_provided"
        ok_log: Optional[bool] = None
        if key in log_eff:
            lv, snip = log_eff[key]
            entry["log"], entry["log_snippet"] = lv, snip
            ok_log = _eq(key, lv, exp)
        else:
            entry["log"] = "not_found" if startup_log_text else "not_provided"
        if ok_argv is False or ok_log is False:
            status = "mismatch"
        elif ok_argv and ok_log:
            status = "verified"
        elif ok_log:
            status = "log_only"
        elif ok_argv:
            status = "argv_only"
        else:
            status = "unverified"
        entry["status"] = status
        fields[key] = entry

    api: Dict[str, Any] = {}
    if models_json:
        try:
            m = models_json["data"][0]
            api = {"served_model_name": m.get("id"), "max_model_len": m.get("max_model_len"), "root": m.get("root")}
            if expect.get("max_model_len") is not None and api["max_model_len"] is not None:
                api["max_model_len_matches"] = api["max_model_len"] == expect["max_model_len"]
            sn = profile.get("model", {}).get("served_name")
            if sn:
                api["served_name_matches"] = api["served_model_name"] == sn
        except Exception as e:  # noqa: BLE001
            api["error"] = repr(e)
    mism = [k for k, v in fields.items() if v["status"] == "mismatch"]
    if api.get("max_model_len_matches") is False:
        mism.append("api.max_model_len")
    if api.get("served_name_matches") is False:
        mism.append("api.served_name")
    by = lambda s: [k for k, v in fields.items() if v["status"] == s]  # noqa: E731
    return {"ok": not mism, "mismatches": mism, "verified": by("verified"), "log_only": by("log_only"),
            "argv_only": by("argv_only"), "unverified": by("unverified"), "fields": fields, "api_models": api}
