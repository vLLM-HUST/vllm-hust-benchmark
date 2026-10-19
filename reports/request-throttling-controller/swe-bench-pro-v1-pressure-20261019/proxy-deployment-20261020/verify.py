from pathlib import Path
import hashlib
import json

root = Path(__file__).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
r = json.loads((root / "report.json").read_text())
assert r["status"] == "FIXED_PROXY_DEPLOYMENT_AND_REQUEST_PATH_PASS"
assert r["one_click_status"] == "PASS" and r["payload_files_checked"] == 42
assert r["official_source_files"] == 17 and r["cards_retained"] == 2
assert not r["model_restarted"] and not r["performance_benefit_claim"]
assert not r["valid_agent_performance_result"]
for arm in ["native", "controller"]:
    a = r["arms"][arm]
    assert a["output_tokens"] == 16 and a["release"]
    assert a["scheduling_enabled"] == (arm == "controller")
    assert not any(a["engine_after"].values())
    diversity = r["initial_prompt_diversity"][arm]
    assert diversity["trajectories"] == 256
    assert diversity["unique_initial_message_hashes"] == 8
    assert diversity["multiplicity_histogram"] == {"32": 8}
assert r["failed_deployment_attempt"]["model_calls"] == 0
print(
    "PASS: deployment/request-path consistency only; no Agent performance claim or private re-audit"
)
