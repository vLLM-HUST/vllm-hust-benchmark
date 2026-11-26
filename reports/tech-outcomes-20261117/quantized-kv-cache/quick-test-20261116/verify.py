"""Verify the sealed quick campaign and recompute observations; never run inference.

Python 3.11+, standard library only. --source reads the original local directory;
without it the public archive is verified and unpacked into a temporary directory.
--write writes derived JSON beside this script; default compares existing outputs.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import re
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def telemetry(root, start, end):
    samples = []
    for line in (root / "evidence/device-samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        if not start <= row["unix"] <= end:
            continue
        values = []
        card = None
        for text in row["npu"].splitlines():
            header = re.match(r"\|\s*(\d+)\s+910B2\s*\|", text)
            if header:
                card = int(header[1])
            # HBM is the last column inside the combined three-metric field.
            match = re.search(r"(\d+)\s*/\s*65536\s*\|\s*$", text)
            if match:
                assert card in (4, 7)
                values.append((card, int(match[1])))
        assert len(values) == 2 and {x[0] for x in values} == {4, 7}
        samples.append((row["unix"], dict(values)))
    assert len(samples) > 1
    return {
        "unit": "MB as printed by npu-smi; no MiB conversion asserted",
        "samples": len(samples),
        "first_unix": samples[0][0],
        "last_unix": samples[-1][0],
        "observed_span_fraction_of_batch": (samples[-1][0] - samples[0][0])
        / (end - start),
        "largest_sample_gap_s": max(b[0] - a[0] for a, b in zip(samples, samples[1:])),
        "per_card_observed_range_MB": {
            str(card): [
                min(x[1][card] for x in samples),
                max(x[1][card] for x in samples),
            ]
            for card in (4, 7)
        },
        "observed_max_two_card_sum_MB": max(sum(x[1].values()) for x in samples),
        "scope": "sampled occupied HBM only, partial B1 coverage; no matched-load memory-saving claim",
    }


def audit_stream(path, row, chat):
    content = ""
    first = None
    usage = None
    finish = None
    done = False
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if not event["line"].startswith("data: "):
            continue
        data = event["line"][6:]
        if data == "[DONE]":
            done = True
            continue
        payload = json.loads(data)
        assert not payload.get("error")
        usage = payload.get("usage") or usage
        for choice in payload.get("choices", []):
            piece = (
                (choice.get("delta", {}).get("content") or "")
                if chat
                else (choice.get("text") or "")
            )
            if piece:
                if first is None:
                    first = event["received_monotonic"]
                content += piece
            finish = choice.get("finish_reason") or finish
    assert row["success"] and row["http_status"] == 200 and row["error"] is None
    assert done and usage and row["content"] == content
    assert (
        row["finish_reason"] == finish
        and row["output_tokens"] == usage["completion_tokens"]
    )
    assert row["prompt_tokens"] == usage["prompt_tokens"]
    assert row["first_content_monotonic"] == first
    assert abs(row["ttft_s"] - (first - row["submitted_monotonic"])) < 1e-9
    assert (
        abs(
            row["tpot_s"]
            - (row["completed_monotonic"] - first) / (row["output_tokens"] - 1)
        )
        < 1e-9
    )


def recompute(root):
    plan = load(root / "execution-plan.json")
    for repo, identity in plan["sources"].items():
        for suffix in ("diff", "status"):
            assert (
                sha(root / "evidence" / (repo + "." + suffix))
                == identity[suffix + "_sha256"]
            )
    original = load(root / "paired-results.json")
    assert (
        sha(root / "inputs/scorer.py")
        == "c2237c69f848e2358abe71c68f54143b5c090a81bd4f7915088b19b8ae157801"
    )
    scorer = module(root / "inputs/scorer.py", "frozen_quick_scorer")
    report = module(root / "report.py", "original_quick_report")
    report.ROOT = root  # audit_arm only; original main reads current /root repos and overwrites originals.
    cases = load(root / "inputs/materialized200.json")
    frozen = load(root / "inputs/mmlu-fixed200.json")
    assert len(cases) == 200
    assert [x["case_id"] for x in cases] == [x["case_id"] for x in frozen["cases"]]
    for case, saved in zip(cases, frozen["cases"]):
        assert (
            hashlib.sha256(case["prompt"].encode()).hexdigest()
            == saved["prompt_sha256"]
            == case["prompt_sha256"]
        )
    for name in ("test", "validation"):
        assert (
            sha(root / "inputs" / (name + ".parquet"))
            == frozen["source_parquet_sha256"][name]
        )
    assert (
        sha(root / "inputs/standard.json")
        == "66c050b7c6d705853f10bc437e6ace1707a9b46d43b6bdb7159724045045b756"
    )
    assert (
        sha(root / "inputs/mmlu-fixed200.json")
        == "8124492d6c59f53817116e27b32cd86c7b06c5ce59a45d1d77df56c080772c87"
    )
    plans = {}
    arms = {}
    rows = {}
    for arm in ("B0", "B1"):
        data, rows[arm] = report.audit_arm(arm, scorer)
        for key, value in data.items():
            assert value == original["arms"][arm][key], (arm, key)
        for row in rows[arm]:
            audit_stream(
                root / arm / "mmlu/raw-sse" / (row["case_id"] + ".jsonl"), row, True
            )
        data["requests_per_s"] = data["successful"] / data["batch_wall_s"]
        data["HBM_reparsed"] = telemetry(root, data["start_unix"], data["end_unix"])
        capacity = re.search(
            r"NPU KV cache size: ([\d,]+) tokens",
            (root / arm / "server.log").read_text(),
        )
        data["reported_KV_cache_capacity_tokens"] = int(capacity[1].replace(",", ""))
        arms[arm] = data
        plans[arm] = load(root / arm / "mmlu/request-plan.json")
        assert len(plans[arm]) == 200
        preflight = load(root / arm / "mmlu/tokenize-preflight.json")
        for entry, case, token in zip(plans[arm], cases, preflight):
            assert entry["case_id"] == case["case_id"] == token["case_id"]
            payload = entry["payload"]
            assert payload["messages"] == [{"role": "user", "content": case["prompt"]}]
            assert (
                payload["max_tokens"] == 8192
                and payload["temperature"] == payload["seed"] == 0
            )
    assert load(root / "B0/mmlu/tokenize-preflight.json") == load(
        root / "B1/mmlu/tokenize-preflight.json"
    )
    for a, b in zip(plans["B0"], plans["B1"]):
        assert {k: v for k, v in a["payload"].items() if k != "cache_salt"} == {
            k: v for k, v in b["payload"].items() if k != "cache_salt"
        }
    for arm in ("B0", "B1", "B1-C8"):
        argv = load(root / arm / "command.json")["argv"]
        assert argv == plan["arms"]["B0" if arm == "B0" else "B1"]
    quality = []
    for case, a, b in zip(cases, rows["B0"], rows["B1"]):
        item = {
            "case_id": case["case_id"],
            "category": case["category"],
            "expected_answer": case["answer"],
            "prompt_in_archive": "inputs/materialized200.json",
            "prompt_sha256": case["prompt_sha256"],
        }
        for arm, row in [("B0", a), ("B1", b)]:
            item[arm] = {
                key: row[key]
                for key in (
                    "success",
                    "http_status",
                    "error",
                    "correct",
                    "parsed_answer",
                    "answer_in_range",
                    "finish_reason",
                    "output_tokens",
                    "prompt_tokens",
                    "ttft_s",
                    "tpot_s",
                )
            }
            item[arm]["full_output_in_archive"] = f"{arm}/mmlu/{case['case_id']}.json"
            item[arm]["raw_SSE_in_archive"] = (
                f"{arm}/mmlu/raw-sse/{case['case_id']}.jsonl"
            )
        item["quality_regression"] = a["correct"] and not b["correct"]
        item["quality_recovery"] = not a["correct"] and b["correct"]
        quality.append(item)
    b0, b1 = arms["B0"], arms["B1"]
    changes = {
        "output_tokens_per_s_percent": 100
        * (b1["output_tokens_per_s"] / b0["output_tokens_per_s"] - 1),
        "requests_per_s_percent": 100
        * (b1["requests_per_s"] / b0["requests_per_s"] - 1),
        "batch_wall_s_percent": 100 * (b1["batch_wall_s"] / b0["batch_wall_s"] - 1),
        "output_token_volume_percent": 100
        * (b1["accepted_output_tokens"] / b0["accepted_output_tokens"] - 1),
        "ttft_mean_increase_percent": 100
        * (b1["ttft_s"]["mean"] / b0["ttft_s"]["mean"] - 1),
    }
    c8 = {}
    for length in ("4096", "16384"):
        c8[length] = {}
        configs = []
        for arm, folder in [("B0", "B0"), ("B1", "B1-C8")]:
            group = root / folder / "continuous" / length
            config = load(group / "config.json")
            assert digest(config["prompt_token_ids"]) == config["token_ids_sha256"]
            configs.append(config)
            phases = {}
            for phase in ("warmup", "formal"):
                path = group / phase
                phase_rows = load(path / "requests.json")
                summary = load(path / "summary.json")
                for row in phase_rows:
                    audit_stream(path / (row["case_id"] + ".sse.jsonl"), row, False)
                    assert row["prompt_tokens"] == int(length)
                wall = max(x["completed_monotonic"] for x in phase_rows) - min(
                    x["submitted_monotonic"] for x in phase_rows
                )
                tokens = sum(x["output_tokens"] for x in phase_rows if x["success"])
                assert len(phase_rows) == summary["actual"] == summary["successful"]
                assert (
                    tokens == summary["accepted_output_tokens"]
                    and wall == summary["batch_wall_s"]
                )
                assert tokens / wall == summary["output_tokens_per_s"]
                assert summary["window_completed"] == sum(
                    x["completed_in_window"] for x in phase_rows
                )
                assert summary["length_stops"] == sum(
                    x["finish_reason"] == "length" for x in phase_rows
                )
                phases[phase] = summary
            c8[length][arm] = phases
        assert configs[0] == configs[1]
        c8[length]["output_tokens_per_s_change_percent_including_drain"] = 100 * (
            c8[length]["B1"]["formal"]["output_tokens_per_s"]
            / c8[length]["B0"]["formal"]["output_tokens_per_s"]
            - 1
        )
    regression = sum(x["quality_regression"] for x in quality)
    recovery = sum(x["quality_recovery"] for x in quality)
    assert regression == 21 and recovery == 24
    n = regression + recovery
    diff = (recovery - regression) / 200
    se = math.sqrt(((n - 200 * diff * diff) / 199) / 200)
    quality_summary = {
        "paired_cases": 200,
        "B0_correct_B1_wrong": regression,
        "B0_wrong_B1_correct": recovery,
        "exact_two_sided_McNemar_p": min(
            1,
            2
            * sum(math.comb(n, i) for i in range(min(regression, recovery) + 1))
            / 2**n,
        ),
        "paired_difference_approx_95CI_percentage_points": [
            100 * (diff - 1.96 * se),
            100 * (diff + 1.96 * se),
        ],
        "scope": "descriptive subset uncertainty only; no equivalence test or throughput repeat-run CI",
    }
    return {
        "schema": "int8-quick-audited/v1",
        "measurement_date_local": "2026-11-16 Asia/Shanghai",
        "experiment_operator": "hustcui",
        "publication_audit": "Codex assisted; offline recomputation only, no new inference",
        "claim_scope": "single-pair frozen MMLU-Pro 200 output-token-rate observation only; no historical A3 throughput, request-throughput gain, stable concurrency or quality-equivalence claim",
        "arms": arms,
        "changes": changes,
        "quality": quality_summary,
        "continuous_C8_engineering": c8,
        "source_identities": plan["sources"],
        "model_identity": "originals: evidence/model-manifest.json, post-run hashes only",
        "activation": {
            "actual_call_verified": original["activation"]["actual_call_verified"],
            "receipt_in_archive": "B1-activation/activation-receipt.json",
            "scope": "independent identical-config diagnostic, excluded from timed arms",
        },
        "limitations": original["limitations"]
        + load(HERE / "manifest.json")["source_limitations"],
        "correction": "Original HBM=0 is a parser defect, not a measured zero; corrected fields above come from original npu-smi output. Original report remains byte-identical.",
        "retained_failures": original["retained_setup_and_diagnostic_failures"],
    }, quality


def verify_members(root, manifest):
    for item in manifest["members"]:
        path = root / item["path"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], (
            item["path"]
        )
    listed = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha(root / name) == expected, name
        listed.add(name)
    assert listed | {"SHA256SUMS"} == {x["path"] for x in manifest["members"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest = load(HERE / "manifest.json")
    with tempfile.TemporaryDirectory(prefix="int8-quick-verify-") as temp:
        archive = Path(temp) / "originals.tar.gz"
        with archive.open("wb") as out:
            for item in manifest["parts"]:
                path = HERE / item["path"]
                assert (
                    path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
                )
                out.write(path.read_bytes())
        assert (
            archive.stat().st_size == manifest["archive_bytes"]
            and sha(archive) == manifest["archive_sha256"]
        )
        root = args.source or Path(temp) / "originals"
        if not args.source:
            with tarfile.open(archive) as tar:
                expected = {x["path"] for x in manifest["members"]}
                members = tar.getmembers()
                assert (
                    len(members) == len(expected)
                    and {x.name for x in members} == expected
                )
                for item in members:
                    assert (
                        item.isfile()
                        and not Path(item.name).is_absolute()
                        and ".." not in Path(item.name).parts
                    )
                tar.extractall(root, filter="data")
        verify_members(root, manifest)
        result, pairs = recompute(root)
        for name, value in [
            ("audited-results.json", result),
            ("quality-pairs.json", pairs),
        ]:
            if args.write:
                (HERE / name).write_text(
                    json.dumps(value, ensure_ascii=False, indent=2) + "\n"
                )
            else:
                assert load(HERE / name) == value, name
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "files": len(manifest["members"]),
                    "MMLU_paired_cases": len(pairs),
                    "all_raw_streams_checked": True,
                    "changes": result["changes"],
                    "HBM": {k: v["HBM_reparsed"] for k, v in result["arms"].items()},
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
