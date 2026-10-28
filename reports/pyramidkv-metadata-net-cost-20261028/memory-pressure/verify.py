"""Offline verification of archive bytes and matched service observations.

Never execute code or extract paths from the evidence archive.
"""

import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent


def main():
    members = json.loads((ROOT / "members.json").read_text())
    files = {}
    chunks = json.loads((ROOT / "chunks.json").read_text())
    payload = []
    for part in chunks["parts"]:
        name = PurePosixPath(part["path"])
        assert not name.is_absolute() and ".." not in name.parts
        data = (ROOT / part["path"]).read_bytes()
        assert len(data) == part["bytes"]
        assert hashlib.sha256(data).hexdigest() == part["sha256"]
        payload.append(data)
    combined = b"".join(payload)
    assert len(combined) == chunks["archive_bytes"]
    assert hashlib.sha256(combined).hexdigest() == chunks["archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(combined), mode="r:gz") as archive:
        for entry in archive:
            name = PurePosixPath(entry.name)
            assert entry.isfile() and not name.is_absolute() and ".." not in name.parts
            assert entry.name not in files
            data = archive.extractfile(entry).read()
            expected = members[entry.name]
            assert len(data) == expected["bytes"]
            assert hashlib.sha256(data).hexdigest() == expected["sha256"]
            files[entry.name] = data
    assert set(files) == set(members)

    def read(name):
        return json.loads(files[name])

    contract = read("paired-plan/manifest.json")
    for name, wanted in contract["files_sha256"].items():
        if name in files:
            assert hashlib.sha256(files[name]).hexdigest() == wanted
        else:
            assert name == "paired-plan/cases.json"  # Private QA input boundary.
    plan = {x["case_id"]: x for x in read("input-index.json")}
    expected_groups = read("paired-plan/groups.json")
    assert len({case["cache_salt"] for case in plan.values()}) == 149
    for case in read("synthetic-inputs.json"):
        assert len(case["prompt"]) == case["prompt_tokens"]
        encoded = json.dumps(case["prompt"], separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == case["prompt_sha256"]
    analysis = json.loads((ROOT / "analysis.json").read_text())
    assert analysis == read("analysis.json")
    stock_members = json.loads((ROOT.parent / "members.json").read_text())
    assert (
        hashlib.sha256(files["stock-on-diagnostic-results.json"]).hexdigest()
        == stock_members["profile-stock-run1/results.json"]["sha256"]
    )
    stock = {row["case_id"]: row for row in read("stock-on-diagnostic-results.json")}
    assert contract["kv_cache_bytes_per_worker"] == 2147483648
    assert analysis["primary_cell"] == "24576-c4"
    custody = read("related-custody.json")
    descriptor = (ROOT / ".." / "batched" / "chunks.json").read_bytes()
    assert hashlib.sha256(descriptor).hexdigest() == custody["descriptor_sha256"]
    assert (
        "metadata_probe"
        not in files["candidate/src/vllm_ascend_kvcompress/provider.py"].decode()
    )
    totals = {}
    transactions = {}
    for arm in contract["order"]:
        prefix = arm + "/evaluation/"
        rows = read(prefix + "results.json")
        lookup = {x["case_id"]: x for x in rows}
        assert len(rows) == len(lookup) == 149 and set(lookup) == set(plan)
        groups = read(prefix + "groups.json")
        assert len(groups) == len(expected_groups)
        log = files[arm + "/server.log"].decode()
        assert log.count("reserved 2.00 GiB for KV Cache") == 2
        # Recover only an explicit adjacent worker prefix split by one API log.
        # The immutable raw log remains unchanged; any recovery is counted explicitly.
        split_prefix = re.compile(
            r"(\(Worker_TP[01] pid=\d+\) )(\(APIServer pid=\d+\) INFO: [^\n]*\n)"
            r"(INFO [^\n]*\[provider.py:\d+\] KV compression worker commit acknowledged request_id=[^\n]+)"
        )
        recovered = list(split_prefix.finditer(log))
        assert len(recovered) == len(
            analysis["arms"][arm]["transactions"]["explicit_prefix_recoveries"]
        )
        log = split_prefix.sub(lambda m: m.group(2) + m.group(1) + m.group(3), log)
        telemetry = [
            json.loads(line)
            for line in files[prefix + "telemetry.jsonl"].decode().splitlines()
        ]
        assert all("error" not in row for row in telemetry)
        hbm = {}
        for row in telemetry:
            device = None
            for line in row.get("npu_smi", "").splitlines():
                match = re.search(r"\|\s*(\d+)\s+910B2", line)
                if match:
                    device = match[1]
                elif device is not None:
                    pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
                    if pairs:
                        used, capacity = map(int, pairs[-1])
                        assert capacity == 65536
                        hbm.setdefault(device, []).append(used)
                        device = None
        resources = analysis["arms"][arm]["resources"]
        assert resources["device_hbm_mib"] == {
            device: {"minimum": min(values), "peak": max(values)}
            for device, values in hbm.items()
        }
        assert resources["peak_kv_usage_fraction"] == max(
            row.get("metrics", {}).get("vllm:kv_cache_usage_perc", 0)
            for row in telemetry
        )
        for key in (
            "num_preemptions",
            "spec_decode_num_draft_tokens",
            "spec_decode_num_accepted_tokens",
        ):
            values = []
            for stage in ("before", "after"):
                metrics = files[prefix + f"metrics-{stage}.txt"].decode()
                found = re.findall(
                    rf"^vllm:{key}_total\{{[^\n]*\}} ([\d.eE+-]+)$",
                    metrics,
                    re.MULTILINE,
                )
                assert found
                values.append(sum(map(float, found)))
            assert analysis["arms"][arm]["counter_deltas"][key] == values[1] - values[0]
        matched_transactions = 0
        for row in rows:
            case = plan[row["case_id"]]
            assert row["success"]
            if arm.startswith("on") and case["kind"] == "quality":
                assert row["prompt_sha256"] == stock[row["case_id"]]["prompt_sha256"]
                assert row["output"] == stock[row["case_id"]]["output"]
            assert all(
                row[k] == case[k]
                for k in ("kind", "prompt_tokens", "prompt_sha256", "cache_salt")
            )
            assert row["usage"]["prompt_tokens"] == case["prompt_tokens"]
            assert row["usage"]["prompt_tokens_details"]["cached_tokens"] >= 0
            if case["ignore_eos"]:
                assert row["usage"]["completion_tokens"] == 128
            assert read(prefix + "requests/" + row["case_id"] + ".json") == row
            text = []
            usage = None
            raw = files[prefix + "requests/" + row["case_id"] + ".sse"].decode()
            for line in raw.splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                text.extend(c.get("text", "") for c in event.get("choices", []))
                if event.get("usage"):
                    usage = event["usage"]
            assert "".join(text) == row["output"] and usage == row["usage"]
            identity = re.escape(row["request_id"])
            commits = re.findall(
                rf"scheduler commit request_id=({identity}(?:-0)?(?:-[a-fA-F0-9]{{8}})?) semantic_tokens=(\d+) physical_tokens=(\d+) source_blocks=(\d+) destination_blocks=(\d+) released_blocks=(\d+)",
                log,
            )
            expected = arm.startswith("on") and case["prompt_tokens"] > 4096
            assert len(commits) == int(expected), (arm, row["case_id"], commits)
            for engine, semantic, physical, _, _, _ in commits:
                assert int(semantic) == case["prompt_tokens"] and int(physical) == 2048
                for rank in (0, 1):
                    ack = re.findall(
                        rf"Worker_TP{rank} [^\n]*worker commit acknowledged request_id={re.escape(engine)} semantic_tokens=(\d+) physical_tokens=(\d+)",
                        log,
                    )
                    assert ack == [(semantic, physical)]
                matched_transactions += 1
        assert not analysis["arms"][arm]["transactions"][
            "requires_recomputation_review"
        ]
        assert (
            sum(row["usage"]["prompt_tokens_details"]["cached_tokens"] for row in rows)
            == analysis["arms"][arm]["cached_tokens"]
        )
        if analysis["arms"][arm]["cached_tokens"]:
            assert analysis["arms"][arm]["counter_deltas"]["num_preemptions"] > 0
        assert matched_transactions == len(
            analysis["arms"][arm]["transactions"]["transactions"]
        )
        assert [
            row["case_id"]
            for row in rows
            if row["usage"]["prompt_tokens_details"]["cached_tokens"]
        ] == analysis["arms"][arm]["nonzero_cached_cases"]
        transactions[arm] = matched_transactions
        for group, expected in zip(groups, expected_groups, strict=True):
            assert all(
                group[k] == expected[k]
                for k in ("name", "kind", "concurrency", "case_ids")
            )
            assert group["failures"] == 0 and group["success"] == len(group["case_ids"])
            tokens = sum(
                lookup[k]["usage"]["completion_tokens"] for k in group["case_ids"]
            )
            assert math.isclose(
                tokens / group["wall_seconds"],
                group["output_tokens_per_second"],
                rel_tol=1e-12,
            )
        totals[arm] = {}
        for length in contract["inputs"]:
            for concurrency in contract["concurrencies"]:
                cell = f"{length}-c{concurrency}"
                selected = [
                    g for g in groups if g["name"].startswith("perf-" + cell + "-")
                ]
                assert len(selected) == 3
                tokens = sum(
                    lookup[k]["usage"]["completion_tokens"]
                    for g in selected
                    for k in g["case_ids"]
                )
                seconds = sum(g["wall_seconds"] for g in selected)
                totals[arm][cell] = (tokens, seconds)
                selected_rows = [lookup[k] for g in selected for k in g["case_ids"]]
                for key in ("ttft_seconds", "tpot_seconds", "e2e_seconds"):
                    assert math.isclose(
                        sum(row[key] for row in selected_rows) / len(selected_rows),
                        analysis["arms"][arm]["performance"][cell][key],
                        rel_tol=1e-12,
                    )
                assert math.isclose(
                    tokens / seconds,
                    analysis["arms"][arm]["performance"][cell][
                        "output_tokens_per_second"
                    ],
                    rel_tol=1e-12,
                )
    for cell in analysis["comparisons"]:
        pooled = {
            kind: sum(totals[f"{kind}-{i}"][cell][0] for i in (1, 2))
            / sum(totals[f"{kind}-{i}"][cell][1] for i in (1, 2))
            for kind in ("off", "on")
        }
        for kind, value in pooled.items():
            assert math.isclose(
                value,
                analysis["comparisons"][cell]["pooled_output_tokens_per_second"][kind],
                rel_tol=1e-12,
            )
        assert math.isclose(
            100 * (pooled["on"] / pooled["off"] - 1),
            analysis["comparisons"][cell]["on_vs_off_percent"],
            abs_tol=1e-9,
        )
        for i, actual in enumerate(analysis["comparisons"][cell]["paired_percent"], 1):
            on = totals[f"on-{i}"][cell]
            off = totals[f"off-{i}"][cell]
            assert math.isclose(
                100 * ((on[0] / on[1]) / (off[0] / off[1]) - 1), actual, abs_tol=1e-9
            )
    print(
        json.dumps(
            {
                "verified": True,
                "archive_members": len(files),
                "service_requests": 596,
                "matched_transactions": transactions,
            }
        )
    )


if __name__ == "__main__":
    main()
