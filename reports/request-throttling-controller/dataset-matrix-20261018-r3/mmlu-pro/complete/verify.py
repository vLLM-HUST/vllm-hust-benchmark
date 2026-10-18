import hashlib
import json
import pathlib
import statistics

r = pathlib.Path(__file__).parent
for line in (r / "SHA256SUMS").read_text().splitlines():
    h, n = line.split("  ", 1)
    assert pathlib.Path(n).name == n
    assert hashlib.sha256((r / n).read_bytes()).hexdigest() == h
p = json.loads((r / "report.json").read_text())
assert p["status"] == "DATASET_PAIR_COMPLETE"
q = p["quality_pair"]
a = q["arms"]["native"]
b = q["arms"]["controller"]
for k in q["pair_input_check"]:
    assert q["pair_input_check"][k] and a[k] == b[k]
for arm, v in [("native", a), ("controller", b)]:
    assert v == json.loads(
        (r.parent / (arm + "-quality-checkpoint") / "checkpoint.json").read_text()
    )
    assert (
        v["summary"]["cases"] == 12032
        and v["summary"]["accuracy"] == v["summary"]["correct"] / 12032
    )
assert q["quality_retention"] == b["summary"]["correct"] / a["summary"]["correct"]
assert q["quality_budget_passed"] == (
    q["quality_retention"] >= q["quality_retention_budget"]
)
f = p["performance"]
assert f == json.loads((r.parent / "performance/checkpoint.json").read_text())
assert len(f["records"]) == 8
for arm in ["native", "controller"]:
    rows = [x for x in f["records"] if x["arm"] == arm and x["repetition"] != "warmup"]
    assert len(rows) == 3
    for k, v in f["aggregate"][arm].items():
        assert v == statistics.median(x[k] for x in rows)
print(
    "PASS complete matched quality and independent performance; public evidence consistency only"
)
