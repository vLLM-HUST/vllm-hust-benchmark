"""Export numeric evidence from frozen archives, or recompute it without archives/NPU."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = {
    "full128": {
        "archive": "pressure-complete-and-capacity-preflight.tar.gz",
        "prefix": "pressure-measurements-v1",
        "arms": ["pressure-native", "pressure-default", "pressure-early"],
        "tasks": 128,
        "requests": 2215,
        "phases": 26,
        "reference": "analysis/full128-original-analysis.json",
    },
    "full32": {
        "archive": "swe32-complete-and-source-evidence.tar.gz",
        "prefix": "swe32-measurements-v1",
        "arms": [
            f"r{i}-{mode}" for i in range(1, 4) for mode in ("native", "working-set")
        ],
        "tasks": 32,
        "requests": 540,
        "phases": 22,
        "reference": "analysis/full32-original-analysis.json",
    },
}
# An explicit allowlist prevents publishing input token arrays or generated text.
REQUEST_FIELDS = (
    "key",
    "phase",
    "prefix_id",
    "task_id",
    "source_index",
    "prompt_tokens",
    "prompt_sha256",
    "messages_sha256",
    "source_query_sha256",
    "lcp_previous_tokens",
    "recorded_tool_seconds",
    "http_status",
    "ttft_s",
    "wall_s",
    "tpot_s",
    "usage",
    "finish_reason",
    "done",
    "valid",
    "input_sha256",
    "output_sha256",
)
PUBLIC_FIELDS = (*REQUEST_FIELDS, "first_chunk_s", "nonempty_chunk_count")
COUNTERS = (
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:request_prefill_kv_computed_tokens_sum",
    "vllm:prompt_tokens_total",
    "vllm:kv_offload_store_bytes_total",
    "vllm:kv_offload_load_bytes_total",
    "vllm:num_preemptions_total",
)
GAUGES = (
    "vllm:kv_cache_usage_perc",
    "vllm:kv_offload_cpu_cache_usage_perc",
    "vllm:kv_tiering_device_cache_usage",
    "vllm:kv_tiering_candidate_working_set_blocks",
    "vllm:kv_tiering_candidate_working_set_bytes",
    "vllm:kv_tiering_candidate_working_set_censored",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_waiting_by_reason",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_request(row):
    result = {key: row[key] for key in REQUEST_FIELDS}
    result["first_chunk_s"] = row["chunk_times_s"][0]
    result["nonempty_chunk_count"] = len(row["chunk_times_s"])
    return result


def encode(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def load(path):
    return json.loads(path.read_bytes())


def select_metrics(lines, names):
    result = {}
    for line in lines:
        if line and not line.startswith("#"):
            key, value, *_ = line.split()
            if key.split("{")[0] in names:
                result[key] = float(value)
    return result


def close(actual, expected, name):
    require(
        math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-7),
        f"Mismatch: {name}: {actual} != {expected}",
    )


def export(archive_dir, root=ROOT):
    """Return deterministic public bytes; never extract or execute archive members."""
    archives = {
        a["name"]: a for a in load(root / "raw-archive-manifest.json")["assets"]
    }
    manifest = {
        "schema": "kv-tiering-public-numeric-evidence/v1",
        "kind": "allowlisted derivatives of frozen originals; not new measurements or complete SSE",
        "groups": {},
        "files": {},
    }
    files = {}

    def publish(path, raw):
        # gzip mtime=0 makes this export byte-reproducible on the same runtime.
        payload = gzip.compress(raw, mtime=0)
        files[path] = payload
        manifest["files"][path] = {
            "bytes": len(payload),
            "sha256": sha(payload),
            "decoded_sha256": sha(raw),
        }

    for group, spec in GROUPS.items():
        path = archive_dir / spec["archive"]
        raw_sha = file_sha(path)
        require(
            raw_sha == archives[path.name]["sha256"]
            and path.stat().st_size == archives[path.name]["bytes"],
            f"Archive identity: {path.name}",
        )
        info = {"archive": path.name, "archive_sha256": raw_sha, "arms": {}}
        with tarfile.open(path) as tar:
            # One forward pass avoids repeated decompression of large input/SSE members.
            payloads = {
                member.name: tar.extractfile(member).read()
                for member in tar
                if member.isfile()
                and member.name.startswith(spec["prefix"] + "/")
                and (
                    member.name.endswith(
                        (
                            "/requests.jsonl",
                            "-summary.json",
                            ".prom",
                            "-client-wall.json",
                        )
                    )
                    or member.name
                    in (
                        spec["prefix"] + "/analysis.json",
                        spec["prefix"] + "/monitor.jsonl",
                    )
                )
            }

            def read(member, payloads=payloads):
                raw = payloads[member]
                return raw, {"member": member, "bytes": len(raw), "sha256": sha(raw)}

            reference, _ = read(spec["prefix"] + "/analysis.json")
            require(
                reference == (root / spec["reference"]).read_bytes(),
                "Frozen analysis differs",
            )
            members = list(payloads)
            for arm in spec["arms"]:
                prefix = spec["prefix"] + "/" + arm + "-client/"
                raw, source = read(prefix + "requests.jsonl")
                rows = [
                    project_request(row) for row in map(json.loads, raw.splitlines())
                ]
                publish(
                    f"public/{arm}-requests.jsonl.gz",
                    b"".join(encode(row) for row in rows),
                )
                phases, sources = [], {}
                for member in sorted(
                    n
                    for n in members
                    if n.startswith(prefix) and n.endswith("-summary.json")
                ):
                    raw, summary_source = read(member)
                    summary = json.loads(raw)
                    phase = {
                        key: summary[key]
                        for key in (
                            "name",
                            "wall_s",
                            "completed",
                            "output_tokens",
                            "output_tok_s",
                        )
                    }
                    phase_sources = {"summary": summary_source}
                    for when in ("before", "after"):
                        raw, metric_source = read(
                            prefix + summary["name"] + f"-{when}.prom"
                        )
                        phase["metrics_" + when] = select_metrics(
                            raw.decode().splitlines(), COUNTERS
                        )
                        phase_sources[when] = metric_source
                    phases.append(phase)
                    sources[summary["name"]] = phase_sources
                wall, wall_source = read(f"{spec['prefix']}/{arm}-client-wall.json")
                publish(
                    f"public/{arm}-phases.json.gz",
                    encode({"phases": phases, "client_wall": json.loads(wall)}),
                )
                info["arms"][arm] = {
                    "requests_source": source,
                    "phase_sources": sources,
                    "client_wall_source": wall_source,
                }
            raw, monitor_source = read(spec["prefix"] + "/monitor.jsonl")
            samples = []
            for line in raw.splitlines():
                row = json.loads(line)
                status = row.get("status") or {}
                if status.get("arm") in spec["arms"]:
                    samples.append(
                        {
                            "epoch": row["epoch"],
                            "arm": status["arm"],
                            "stage": status.get("stage"),
                            "sample_error": bool(row.get("error")),
                            "metrics": select_metrics(row.get("metrics", []), GAUGES),
                        }
                    )
            publish(
                f"public/{group}-samples.jsonl.gz",
                b"".join(encode(row) for row in samples),
            )
            info["monitor_source"] = monitor_source
        manifest["groups"][group] = info
    files["public/manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2).encode() + b"\n"
    )
    return files


def evaluate_arm(rows, evidence, spec, plugin_enabled=False):
    """Compute from per-request timings, phase wall and before/after counters."""
    require(len(rows) == spec["requests"], "Missing/extra requests")
    require(len({r["key"] for r in rows}) == len(rows), "Duplicate request identity")
    require(len({r["task_id"] for r in rows}) == spec["tasks"], "Task pool differs")
    by_phase = defaultdict(list)
    for row in rows:
        require(
            set(row) == set(PUBLIC_FIELDS), "Unexpected/missing public request fields"
        )
        require(
            row["valid"] is True
            and row["http_status"] == 200
            and row["done"] == 1
            and row["finish_reason"] == "length",
            "Unsuccessful request",
        )
        usage = row["usage"]
        require(
            usage["completion_tokens"] == 32
            and usage["prompt_tokens"] == row["prompt_tokens"]
            and usage["total_tokens"] == row["prompt_tokens"] + 32,
            "Usage differs",
        )
        require(row["input_sha256"] == row["prompt_sha256"], "Input digest differs")
        require(
            0 <= row["ttft_s"] <= row["wall_s"] and row["nonempty_chunk_count"] > 0,
            "Invalid timing",
        )
        close(row["ttft_s"], row["first_chunk_s"], "first nonempty chunk")
        close(row["tpot_s"], (row["wall_s"] - row["ttft_s"]) / 31, "TPOT")
        by_phase[row["phase"]].append(row)
    phases = evidence["phases"]
    require(
        len(phases) == spec["phases"]
        and len({p["name"] for p in phases}) == len(phases),
        "Missing/duplicate phase",
    )
    require(set(by_phase) == {p["name"] for p in phases}, "Request/phase mismatch")
    result_phases = {}
    totals = dict.fromkeys(COUNTERS, 0.0)
    for phase in phases:
        cohort = by_phase[phase["name"]]
        output = sum(row["usage"]["completion_tokens"] for row in cohort)
        require(
            phase["completed"] == len(cohort)
            and phase["output_tokens"] == output
            and phase["wall_s"] > 0,
            "Phase count/wall differs",
        )
        require(
            phase["wall_s"] >= max(r["wall_s"] for r in cohort),
            "Phase shorter than request",
        )
        close(phase["output_tok_s"], output / phase["wall_s"], "phase throughput")
        counters = dict.fromkeys(COUNTERS, 0.0)
        require(
            set(phase["metrics_before"]) <= set(phase["metrics_after"]),
            "Counter disappeared",
        )
        for key, end in phase["metrics_after"].items():
            metric = key.split("{")[0]
            require(metric in COUNTERS, "Unexpected metric")
            delta = end - phase["metrics_before"].get(key, 0)
            require(
                delta >= 0 and math.isfinite(delta), "Counter reset/nonfinite value"
            )
            counters[metric] += delta
        # Native exposes no plugin byte counters. Absence is not evidence of an FS read.
        required = set(COUNTERS) - {
            "vllm:kv_offload_store_bytes_total",
            "vllm:kv_offload_load_bytes_total",
        }
        if plugin_enabled:
            required = set(COUNTERS)
        require(
            required <= {key.split("{")[0] for key in phase["metrics_after"]},
            "Missing core metric",
        )
        close(
            counters["vllm:prompt_tokens_total"],
            sum(r["prompt_tokens"] for r in cohort),
            "server/client input tokens",
        )
        for key, value in counters.items():
            totals[key] += value
        result_phases[phase["name"]] = {
            "requests": len(cohort),
            "wall_s": phase["wall_s"],
            "output_tok_s": output / phase["wall_s"],
            "ttft_mean_s": statistics.mean(r["ttft_s"] for r in cohort),
            "tpot_mean_s": statistics.mean(r["tpot_s"] for r in cohort),
            "counters": counters,
        }
    wall = sum(p["wall_s"] for p in phases)
    client_wall = evidence["client_wall"]["seconds"]
    require(client_wall >= wall, "Client wall excludes measured rounds")
    output = sum(r["usage"]["completion_tokens"] for r in rows)
    return {
        "requests": len(rows),
        "logical_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "output_tokens": output,
        "phase_wall_s_sum": wall,
        "output_tok_s": output / wall,
        "whole_client_output_tok_s": output / client_wall,
        "ttft_mean_s": statistics.mean(r["ttft_s"] for r in rows),
        "tpot_mean_s": statistics.mean(r["tpot_s"] for r in rows),
        "counters": totals,
        "phases": result_phases,
    }


def recompute(root=ROOT):
    for path, expected in load(root / "preserved-members-sha256.json").items():
        require(
            sha((root / path).read_bytes()) == expected,
            "Frozen member changed: " + path,
        )
    manifest = load(root / "public/manifest.json")
    archives = {
        a["name"]: a for a in load(root / "raw-archive-manifest.json")["assets"]
    }
    decoded = {}
    for path, identity in manifest["files"].items():
        require(
            path.startswith("public/") and ".." not in Path(path).parts,
            "Unsafe manifest path",
        )
        raw = (root / path).read_bytes()
        require(
            len(raw) == identity["bytes"] and sha(raw) == identity["sha256"],
            f"Public file identity: {path}",
        )
        decoded[path] = gzip.decompress(raw)
        require(
            sha(decoded[path]) == identity["decoded_sha256"],
            f"Decoded identity: {path}",
        )
    output = {
        "scope": "offline numeric recomputation; not SSE/content/quality validation, a new run or an official score",
        "groups": {},
    }
    for group, spec in GROUPS.items():
        info = manifest["groups"][group]
        require(
            info["archive"] == spec["archive"]
            and info["archive_sha256"] == archives[spec["archive"]]["sha256"],
            "Original archive reference differs",
        )
        require(set(info["arms"]) == set(spec["arms"]), "Arm selection differs")
        reference = load(root / spec["reference"])
        arms, signatures = {}, []
        for arm in spec["arms"]:
            rows = list(
                map(json.loads, decoded[f"public/{arm}-requests.jsonl.gz"].splitlines())
            )
            evidence = json.loads(decoded[f"public/{arm}-phases.json.gz"])
            actual = evaluate_arm(
                rows, evidence, spec, plugin_enabled="native" not in arm
            )
            expected = reference["arms"][arm]
            for key in (
                "requests",
                "logical_prompt_tokens",
                "output_tokens",
                "phase_wall_s_sum",
                "output_tok_s",
                "ttft_mean_s",
                "tpot_mean_s",
            ):
                close(actual[key], expected[key], arm + ":" + key)
            for key in COUNTERS:
                close(
                    actual["counters"][key], expected["counters"][key], arm + ":" + key
                )
            for name, phase in actual["phases"].items():
                for key in ("requests", "wall_s", "ttft_mean_s", "tpot_mean_s"):
                    close(
                        phase[key],
                        expected["phases"][name][key],
                        arm + ":" + name + ":" + key,
                    )
                for key in COUNTERS:
                    close(
                        phase["counters"][key],
                        expected["phases"][name]["counters"][key],
                        arm + ":" + name + ":" + key,
                    )
            signatures.append(
                sorted(
                    (
                        r["key"],
                        r["task_id"],
                        r["source_index"],
                        r["input_sha256"],
                        r["prompt_tokens"],
                    )
                    for r in rows
                )
            )
            arms[arm] = actual
        require(
            all(s == signatures[0] for s in signatures), "Unmatched inputs between arms"
        )
        peaks = {arm: {} for arm in spec["arms"]}
        for line in decoded[f"public/{group}-samples.jsonl.gz"].splitlines():
            row = json.loads(line)
            for key, value in row["metrics"].items():
                peaks[row["arm"]][key] = max(peaks[row["arm"]].get(key, 0), value)
        require(peaks == reference["sampled_peaks"], "Sampled pressure peaks differ")
        pairs = (
            [
                ("pressure-native", "pressure-default"),
                ("pressure-native", "pressure-early"),
            ]
            if group == "full128"
            else [(f"r{i}-native", f"r{i}-working-set") for i in range(1, 4)]
        )
        comparisons = {}
        for native, candidate in pairs:
            comparisons[candidate] = {
                key + "_change_pct": (arms[candidate][key] / arms[native][key] - 1)
                * 100
                for key in (
                    "output_tok_s",
                    "whole_client_output_tok_s",
                    "ttft_mean_s",
                    "tpot_mean_s",
                )
            }
        output["groups"][group] = {
            "arms": arms,
            "comparisons": comparisons,
            "all_input_identities_match": True,
            "sampled_peaks": peaks,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Recompute public evidence with stdlib only",
    )
    mode.add_argument(
        "--export-from",
        type=Path,
        help="Create new allowlisted derivatives from verified local archives",
    )
    mode.add_argument(
        "--check-originals",
        type=Path,
        help="Compare every decoded public derivative with a fresh export of controlled originals",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional numeric recomputation output; never overwrites original analyses",
    )
    args = parser.parse_args()
    if args.export_from or args.check_originals:
        files = export(args.export_from or args.check_originals)
        for path, raw in files.items():
            target = ROOT / path
            if args.export_from:
                require(
                    not target.exists(),
                    "Export target exists; use a new version directory instead of replacing evidence",
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            elif path.endswith(".gz"):
                require(
                    gzip.decompress(target.read_bytes()) == gzip.decompress(raw),
                    "Controlled/public derivation differs: " + path,
                )
            else:
                old, fresh = load(target), json.loads(raw)
                require(
                    old["groups"] == fresh["groups"], "Original member custody differs"
                )
                require(
                    set(old["files"]) == set(fresh["files"]),
                    "Public file inventory differs",
                )
                for name in old["files"]:
                    require(
                        old["files"][name]["decoded_sha256"]
                        == fresh["files"][name]["decoded_sha256"],
                        "Decoded derivation manifest differs",
                    )
        print("Controlled archive export/derivation check passed")
    result = recompute()
    if args.output:
        require(
            args.output.resolve()
            not in [p.resolve() for p in ROOT.rglob("*") if p.is_file()],
            "Refusing to overwrite an existing report/evidence file",
        )
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for group, data in result["groups"].items():
        for arm, comparison in data["comparisons"].items():
            print(group, arm, json.dumps(comparison, sort_keys=True))
    print(
        "9 arms / 9,885 requests: public timing, input identity, phase, counter and pressure checks passed; no NPU used"
    )


if __name__ == "__main__":
    main()
