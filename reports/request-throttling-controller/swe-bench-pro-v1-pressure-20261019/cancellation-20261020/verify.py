from pathlib import Path
import hashlib
import json

root = Path(__file__).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
r = json.loads((root / "report.json").read_text())
assert r["status"] == "ISOLATED_NON_STREAMING_CANCELLATION_ACCEPTED"
assert not r["benefit_claim"] and not r["real_agent_comparison_rerun"]
assert not r["production_proxy_replaced"] and not r["model_restarted"]
assert r["fixed_upstream_files_verified"] == 17
for arm in ["native", "controller"]:
    before, after = r["before"][arm], r["after"][arm]
    assert before["running_first_sample_after_timeout"] == 1
    assert before["release"]["released"]
    assert 0 < after["first_idle_sample_after_timeout_seconds"] < 2
    assert 0 <= after["generation_token_delta"] < 512
    assert after["abort_delta"] == 0
    assert after["normal_response"]["status"] == 200
    assert after["normal_response"]["usage"]["completion_tokens"] == 16
for name, expected in r["source_hashes"].items():
    assert (
        hashlib.sha256(
            (root.parent / "sources" / "cancellation-20261020" / name).read_bytes()
        ).hexdigest()
        == expected
    )
print(
    "PASS: public integrity and limited transport-acceptance consistency; no private re-audit or performance claim"
)
