"""Approved profile selection must preserve the original strict baseline gate."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shared_profile", ROOT / "tools/shared_test_profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_baseline_bytes_and_default_gate_remain_frozen():
    assert hashlib.sha256(profile.PROFILE.read_bytes()).hexdigest() == "5792631172af883cf247e05b214444c02a106831aa50162b8c6f3a0e63f24661"
    pressure = profile.load("pressure-256-v1")
    with pytest.raises(ValueError, match="max_num_seqs differs"):
        profile.validate_engine(pressure["engine"], pressure["packages"])


def test_explicit_approved_pressure_profile_changes_only_authorized_fields():
    baseline = profile.load()
    pressure = profile.load("pressure-256-v1")
    proof = profile.validate_engine(pressure["engine"], pressure["packages"], "pressure-256-v1")
    assert proof["shared_profile_id"] == pressure["id"]
    assert pressure["authorization"]["instruction"] == "允许另冻压力配置"
    engine = copy.deepcopy(pressure["engine"])
    assert engine["max_num_seqs"] == 256
    assert engine["compilation_config"]["max_cudagraph_capture_size"] == 256
    engine["max_num_seqs"] = baseline["engine"]["max_num_seqs"]
    for name in ["max_cudagraph_capture_size", "cudagraph_capture_sizes"]:
        engine["compilation_config"][name] = baseline["engine"]["compilation_config"][name]
    assert engine == baseline["engine"]
    assert pressure["packages"] == baseline["packages"]
    changed = copy.deepcopy(pressure["engine"])
    changed["kv_cache_memory_bytes"] //= 2
    with pytest.raises(ValueError, match="kv_cache_memory_bytes differs"):
        profile.validate_engine(changed, pressure["packages"], "pressure-256-v1")


def test_local_profile_override_cannot_bypass_central_registry():
    with pytest.raises(ValueError, match="Unknown central shared profile"):
        profile.load("/tmp/private-override.json")


def test_frozen_bootstrap_cached_old_module_cannot_shadow_registered_selector(monkeypatch):
    import sys
    import types
    sys.path.insert(0, str(ROOT / "experiments/workload"))
    import swe_bench_pro_runtime as runtime
    old = types.ModuleType("shared_test_profile")
    old.load = lambda: {"id": "cached-old-module"}
    monkeypatch.setitem(sys.modules, "shared_test_profile", old)
    selected = runtime.shared_profile_module()
    value = selected.load("pressure-256-v1")
    assert value["engine"]["max_num_seqs"] == 256
    assert selected.validate_engine(value["engine"], value["packages"], "pressure-256-v1")["shared_profile_id"] == value["id"]
