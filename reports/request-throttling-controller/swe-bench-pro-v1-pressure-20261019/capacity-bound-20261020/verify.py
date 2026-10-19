from pathlib import Path
import hashlib
import json
import math

root = Path(__file__).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
r = json.loads((root / "report.json").read_text())
b = json.loads((root / "baseline-profile.json").read_text())
c = json.loads((root / "shared-pressure-profile-candidate.json").read_text())
g = json.loads((root / "candidate-gate-check.json").read_text())
o, d = r["observed"], r["derived"]
assert not r["hardware_saturation_test_run"]
assert (
    r["baseline_profile_sha256"]
    == hashlib.sha256((root / "baseline-profile.json").read_bytes()).hexdigest()
)
assert o["pool_blocks"] == 809 and o["usable_blocks"] == 808
assert math.isclose(o["pool_blocks"] / o["max_length_capacity_requests"], 22)
assert d["unique_per_full_length_request_block_cost"] == 22
assert int(o["max_length_capacity_requests"] * o["max_model_len"]) == 1204968
assert d["max_live_request_blocks_at_16"] == 16 * 22
assert math.isclose(
    d["reported_usage_bound_regular_standalone_percent"], 352 / 808 * 100
)
assert d["minimum_full_length_unshared_requests_for_90pct"] == math.ceil(0.9 * 808 / 22)
assert "NOT_APPROVED_NOT_DEPLOYED" in c["status"]
assert (
    g["status"] == "EXPECTED_REJECTION_BY_CURRENT_SHARED_GATE"
    and not g["candidate_deployed"]
)
be, ce = b["engine"], c["engine"]
assert be["max_num_seqs"] == 16 and ce["max_num_seqs"] == 256
assert ce["compilation_config"]["cudagraph_capture_sizes"] == [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
]
assert ce["compilation_config"]["max_cudagraph_capture_size"] == 256
ce["max_num_seqs"] = be["max_num_seqs"]
for key in ["cudagraph_capture_sizes", "max_cudagraph_capture_size"]:
    ce["compilation_config"][key] = be["compilation_config"][key]
assert ce == be
assert len(r["source_hashes"]) >= 5 and r["conditions"] and r["limits"]
print(
    "PASS: public arithmetic/config consistency only; candidate not deployed; no hardware saturation or performance claim"
)
