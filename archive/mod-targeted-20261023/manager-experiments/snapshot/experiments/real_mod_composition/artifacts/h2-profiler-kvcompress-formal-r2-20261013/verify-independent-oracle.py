#!/usr/bin/env python3
"""Independently verify the revision-2 source-bound ownership oracle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "campaign.json"
ORACLE = HERE / "oracle.json"
RECEIPT = HERE / "oracle-receipt.json"
GATE_ORACLE = HERE / "oracle-gate.json"
PRIOR_RECEIPT = Path(
    "/root/ecpa-h2-real-pair-20261013/independent-oracle-receipt.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> int:
    campaign_bytes = CAMPAIGN.read_bytes()
    campaign = json.loads(campaign_bytes)
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    gate_oracle = json.loads(GATE_ORACLE.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_RECEIPT.read_text(encoding="utf-8"))

    require(sha256(CAMPAIGN) == oracle["campaign"]["sha256"], "campaign SHA")
    require(sha256(ORACLE) == receipt["oracle_sha256"], "oracle SHA receipt")
    require(sha256(Path(__file__)) == receipt["verifier_sha256"], "verifier SHA")
    require(gate_oracle["schema"] == "ecpa-h2-real-mod-oracle/v1",
            "gate wrapper schema")
    require(gate_oracle["campaign_sha256"] == f"sha256:{sha256(CAMPAIGN)}",
            "gate wrapper campaign binding")
    require(gate_oracle["status"] == "automated-independent-reviewed",
            "gate wrapper status")
    require(gate_oracle["review"]["kind"] == "automated-independent",
            "gate wrapper review kind")
    require(gate_oracle["review"]["verdict"] == "PASS",
            "gate wrapper review verdict")
    require(gate_oracle["review"]["ownership_kind"] == "physical-ownership",
            "gate wrapper ownership kind")
    require(gate_oracle["review"]["ownership_verdict"]
            == oracle["verdict"] == "admit", "gate wrapper ownership verdict")
    require(gate_oracle["labels"] == [{
        "case_id": oracle["case_id"],
        "decision": "admit",
        "scope": "physical-ownership",
    }], "gate wrapper admit label")
    require(gate_oracle["bindings"]["detailed_oracle_sha256"]
            == f"sha256:{sha256(ORACLE)}", "gate wrapper detailed oracle binding")
    require(gate_oracle["bindings"]["independent_receipt_sha256"]
            == f"sha256:{sha256(RECEIPT)}", "gate wrapper receipt binding")
    require(gate_oracle["review"]["detailed_oracle_sha256"]
            == gate_oracle["bindings"]["detailed_oracle_sha256"],
            "review detailed oracle binding")
    require(gate_oracle["review"]["independent_receipt_sha256"]
            == gate_oracle["bindings"]["independent_receipt_sha256"],
            "review receipt binding")
    require(
        sha256(PRIOR_RECEIPT) == oracle["prior_independent_receipt"]["sha256"],
        "prior independent receipt SHA",
    )
    require(campaign["campaign_revision"] == 2, "campaign revision")
    old_runner_sha = oracle["campaign_digest_update"]["previous_runner_sha256"].encode()
    new_runner_sha = oracle["campaign_digest_update"]["current_runner_sha256"].encode()
    require(campaign_bytes.count(new_runner_sha) == 1,
            "new runner digest is not a single campaign field")
    reverted = campaign_bytes.replace(new_runner_sha, old_runner_sha)
    require(hashlib.sha256(reverted).hexdigest()
            == oracle["campaign_digest_update"]["previous_campaign_sha256"],
            "campaign changed beyond the single runner.sha256 field")
    require(campaign["formal_runs"] == [], "campaign was not frozen before runs")
    require(campaign["predecessor_campaign"]["sha256"] == prior["campaign_sha256"],
            "predecessor is not the previously reviewed campaign")
    require(campaign["predecessor_campaign"]["status"] == "excluded-fail-closed",
            "predecessor is not excluded fail-closed")
    attempts = campaign["excluded_attempts"]
    require(len(attempts) == 2, "expected two excluded workload-drift attempts")
    require(all(row["score_eligible"] is False for row in attempts),
            "excluded attempt became score eligible")
    require(all("request.model" in row["reason"] and "served_name_template" in row["reason"]
                for row in attempts), "excluded attempt reason changed")

    expected_orders = {
        "profiler-then-kvcompress": ["lifecycle_profiler", "kvcompress"],
        "kvcompress-then-profiler": ["kvcompress", "lifecycle_profiler"],
    }
    require(
        {row["order_id"]: row["pip_order"] for row in campaign["installation_orders"]}
        == expected_orders,
        "installation orders",
    )
    require(campaign["model"] == oracle["frozen_runtime_inputs"]["model"], "model")
    require(campaign["workload"] == oracle["frozen_runtime_inputs"]["workload"],
            "workload")
    require(campaign["host"]["tensor_parallel_size"] == 2, "TP size")
    require(campaign["kvcompress"]["compression_threshold_tokens"] == 9216,
            "compression threshold")
    require(campaign["kvcompress"]["workload_expected_to_cross_threshold"] is False,
            "workload threshold classification")

    runner = Path(campaign["runner"]["path"])
    require(sha256(runner) == campaign["runner"]["sha256"], "runner SHA")
    runner_text = runner.read_text(encoding="utf-8")
    runner_contracts = (
        'served_template = str(',
        'served = served_template.format(order_id=args.order)',
        'frozen_workload.get(',
        '"model": served,',
        '"--served-model-name",',
        'str(tensor_parallel_size)',
        'str(max_model_len)',
        'campaign.get("runner", {}).get("sha256") != sha256(Path(__file__))',
        'raise RuntimeError("formal campaign runner digest mismatch")',
    )
    require(all(token in runner_text for token in runner_contracts),
            "runner no longer derives frozen request/server inputs")

    campaign_members = {row["id"]: row for row in campaign["members"]}
    for name, source in oracle["sources"].items():
        commit = source["commit"]
        require(git(source["repo"], "rev-parse", f"{commit}^{{commit}}") == commit,
                f"{name} commit")
        require(git(source["repo"], "rev-parse", f"{commit}^{{tree}}") == source["tree"],
                f"{name} tree")
        for file_name, blob in source["files"].items():
            require(git(source["repo"], "rev-parse", f"{commit}:{file_name}") == blob,
                    f"{name} blob {file_name}")
    require(campaign_members["lifecycle_profiler"]["commit"]
            == oracle["sources"]["lifecycle_profiler"]["commit"], "profiler campaign commit")
    require(campaign_members["kvcompress"]["commit"]
            == oracle["sources"]["kvcompress"]["commit"], "KVCompress campaign commit")
    require(campaign["host"]["core"]["commit"] == oracle["sources"]["host_core"]["commit"],
            "core campaign commit")
    require(campaign["host"]["ascend"]["commit"]
            == oracle["sources"]["host_ascend"]["commit"], "Ascend campaign commit")
    require(oracle["verdict"] == "admit", "unexpected oracle verdict")
    print(f"PASS revision=2 verdict=admit oracle_sha256={sha256(ORACLE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
