"""Rebuild synthetic continuous stress metrics from sealed timestamps and SSE."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

from audit_fixed018_retrieval import audit_activation
from recompute_fixed018_serving_cost import parse_npu


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def quantile(values, probability):
    """Nearest rank, including small samples; no interpolation or outlier deletion."""
    return sorted(values)[math.ceil(probability * len(values)) - 1] if values else None


def reconstruct(raw):
    first = last = usage = finish = None
    done = False
    prior = -math.inf
    for line in gzip.decompress(raw).splitlines():
        saved = json.loads(line)
        stamp = saved["clock"]["monotonic"]
        if stamp < prior:
            raise ValueError("Non-monotonic stream")
        prior = stamp
        wire = saved["raw"]
        if not wire.startswith("data:"):
            continue
        payload = wire[5:].strip()
        if payload == "[DONE]":
            done = True
            continue
        event = json.loads(payload)
        if event.get("error"):
            raise ValueError("Recorded stream error")
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if any(delta.get(k) for k in ("content", "reasoning", "reasoning_content")):
                first = stamp if first is None else first
                last = stamp
            finish = choice.get("finish_reason") or finish
    if not done or first is None or not usage or not finish:
        raise ValueError("Incomplete recorded stream")
    return first, last, usage, finish


def timing(rows):
    result = {"samples": len(rows)}
    for key in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [r[key] for r in rows]
        result[key] = {
            "mean": statistics.fmean(values) if values else None,
            "p95": quantile(values, 0.95),
            "p99": quantile(values, 0.99),
        }
    return result


def utilization(samples, start, end):
    """Timestamp at sample-call midpoint; left hold until next call, not physical CI."""
    totals = {"5": 0.0, "6": 0.0}
    covered = 0.0
    for current, following in zip(samples, samples[1:]):
        duration = max(0, min(end, following["time"]) - max(start, current["time"]))
        if duration and current["values"] is not None:
            covered += duration
            for card in totals:
                totals[card] += current["values"][card] * duration
    complete = math.isclose(covered, end - start, abs_tol=1e-5)
    values = {k: v / covered for k, v in totals.items()} if complete else None
    return dict(
        covered_seconds=covered,
        missing_seconds=max(0, end - start - covered),
        per_card_percent=values,
        mean_percent=statistics.fmean(values.values()) if values else None,
    )


def audit(root):
    def raw(name):
        return (root / name).read_bytes()

    def read(name):
        return json.loads(raw(name))

    manifest = {}
    for line in raw("SHA256SUMS").decode().splitlines():
        digest, name = line.split("  ", 1)
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe manifest member")
        if sha(raw(name)) != digest:
            raise ValueError("Changed sealed member: " + name)
        manifest[name] = digest
    receipt = read("receipt.json")
    if receipt["status"] != "completed" or not receipt["all_reserved_devices_released"]:
        raise ValueError("Arm not completed and released")
    identity = read("runtime-identity.json")
    if identity["role"] == "B0" and not (
        identity["vllm_version"] == identity["ascend_version"] == "0.18.0"
    ):
        raise ValueError("Not fixed018 B0")
    plan = read("plan.json")
    inputs = read("request-plan.json")
    if sha(raw("request-plan.json")) != plan["input_sha256"]:
        raise ValueError("Frozen input mismatch")
    preflight = read("tokenize-preflight.json")
    expected_tokens = [
        {k: c[k] for k in ("input_tokens", "token_ids_sha256")}
        for cases in inputs.values()
        for c in cases
    ]
    if preflight != expected_tokens:
        raise ValueError("Server tokenization mismatch")
    rows = []
    for name in sorted(manifest):
        if not name.endswith(".sse.jsonl.gz"):
            continue
        record = read(name.removesuffix(".sse.jsonl.gz") + ".json")
        first, last, usage, finish = reconstruct(raw(name))
        if not (
            record["success"]
            and record["http_status"] == 200
            and first == record["first_token"]["monotonic"]
            and last == record["last_token"]["monotonic"]
            and usage == record["usage"]
            and finish == record["finish_reason"] == "length"
            and usage["completion_tokens"] == 256
            and usage["prompt_tokens"] == record["input_tokens"]
        ):
            raise ValueError("Stream/scalar mismatch: " + name)
        start, end = record["start"]["monotonic"], record["end"]["monotonic"]
        if not start <= first <= last <= end:
            raise ValueError("Invalid timing order")
        rows.append(
            dict(
                id=record["id"],
                phase=record["phase"],
                start=start,
                end=end,
                input_tokens=usage["prompt_tokens"],
                output_tokens=usage["completion_tokens"],
                cached_tokens=usage.get("prompt_tokens_details", {}).get(
                    "cached_tokens"
                ),
                ttft_ms=(first - start) * 1000,
                tpot_ms=(end - first) / 255 * 1000,
                e2e_ms=(end - start) * 1000,
                raw_sse_sha256=manifest[name],
            )
        )
    samples = []
    call_durations = []
    for line in raw("device-samples.jsonl").splitlines():
        r = json.loads(line)
        if "error" in r:
            samples.append(dict(time=r["monotonic"], values=None))
            continue
        values = parse_npu(r["raw"])
        if set(values) != {"5", "6"}:
            raise ValueError("Unexpected sampled cards")
        call_durations.append(r["monotonic_end"] - r["monotonic_start"])
        samples.append(
            dict(time=(r["monotonic_end"] + r["monotonic_start"]) / 2, values=values)
        )
    phases = []
    for phase in read("continuous.json")["phases"]:
        cohort = [r for r in rows if r["phase"] == phase["name"]]
        start, end = phase["start"]["monotonic"], phase["end"]["monotonic"]
        deadline = phase["submission_deadline_monotonic"]
        if phase["failed"] or len(cohort) != phase["requests"]:
            raise ValueError("Phase cohort mismatch")
        if not all(start <= r["start"] < deadline and r["end"] <= end for r in cohort):
            raise ValueError("Requests outside declared phase")
        item = dict(
            name=phase["name"],
            requests=len(cohort),
            failures=0,
            submission_seconds=phase["seconds"],
            elapsed_including_drain=end - start,
            drain_seconds=end - deadline,
            output_tokens=sum(r["output_tokens"] for r in cohort),
            cohort_output_tokens_per_second=sum(r["output_tokens"] for r in cohort)
            / (end - start),
            timing=timing(cohort),
            utilization=utilization(samples, start, end),
            windows=[],
        )
        if phase["name"].endswith("-formal"):
            for offset in range(0, 300, 60):
                left, right = start + offset, start + offset + 60
                arrivals = [r for r in cohort if left <= r["start"] < right]
                inside = [r for r in cohort if left <= r["start"] and r["end"] <= right]
                crossing = [
                    r
                    for r in cohort
                    if r["start"] < right and r["end"] > left and r not in inside
                ]
                item["windows"].append(
                    dict(
                        offset_seconds=offset,
                        seconds=60,
                        arrival_cohort_timing_including_drain=timing(arrivals),
                        fully_contained_requests=len(inside),
                        boundary_crossing_requests=len(crossing),
                        fully_contained_output_tokens=sum(
                            r["output_tokens"] for r in inside
                        ),
                        fully_contained_request_accounting_tokens_per_second=sum(
                            r["output_tokens"] for r in inside
                        )
                        / 60,
                        utilization=utilization(samples, left, right),
                    )
                )
        phases.append(item)
    return dict(
        schema="fixed018-continuous-audit/v1",
        run=root.name,
        runtime=identity,
        source_manifest_sha256=sha(raw("SHA256SUMS")),
        verified_members=len(manifest),
        request_plan_sha256=sha(raw("request-plan.json")),
        receipt=receipt,
        phases=phases,
        rows=rows,
        sampling=dict(
            call_duration_max_seconds=max(call_durations),
            max_sample_gap_seconds=max(
                b["time"] - a["time"] for a, b in zip(samples, samples[1:])
            ),
            method="Monotonic midpoint, left-held samples; missing intervals not zero-filled; sampled estimate, not physical utilization CI.",
        ),
        scope="Synthetic C8 stress only. Warmups excluded from formal metrics. Formal throughput includes full drain in denominator; all formal requests in numerator. Window throughput counts only requests wholly inside each window, with crossings separate, not exact token-arrival throughput. Arrival-cohort latency includes drain; small-sample P99 retained. No five-dataset quality, fixed-arrival latency or business-cost acceptance inferred.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--capsule", action="store_true")
    args = parser.parse_args()
    value = audit_capsule(args.root) if args.capsule else audit(args.root)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in value.items()
                if k in ("run", "verified_members", "phases")
            },
            indent=2,
        )
    )


def audit_capsule(directory):
    """No extraction or private data required to reproduce the published pair status."""
    manifest = json.loads((directory / "continuous-capsule.json").read_text())
    path = directory / manifest["file"]
    if sha(path.read_bytes()) != manifest["sha256"]:
        raise ValueError("Capsule SHA mismatch")
    files = {}
    with tarfile.open(path) as archive:
        for member in archive:
            if (
                not member.isfile()
                or member.name in files
                or Path(member.name).is_absolute()
                or ".." in Path(member.name).parts
            ):
                raise ValueError("Unsafe archive member")
            files[member.name] = archive.extractfile(member).read()
    if len(files) != manifest["members"]:
        raise ValueError("Archive member count changed")

    class MemoryPath:
        def __init__(self, name):
            self.path = name
            self.name = name.rsplit("/", 1)[-1]

        def __truediv__(self, name):
            return MemoryPath(self.path + "/" + name)

        def read_bytes(self):
            return files[self.path]

    b0 = audit(MemoryPath("continuous-b0-r1"))
    failed = []
    for run in ("continuous-b1-r1", "continuous-b1-r2"):
        prefix = run + "/"
        for line in files[prefix + "SHA256SUMS"].decode().splitlines():
            digest, name = line.split("  ", 1)
            if sha(files[prefix + name]) != digest:
                raise ValueError("Failed-run evidence changed")
        receipt = json.loads(files[prefix + "receipt.json"])
        if (
            receipt["status"] != "failed"
            or not receipt["all_reserved_devices_released"]
        ):
            raise ValueError("Failed receipt changed")
        rows = []
        for name, raw in files.items():
            if not name.startswith(prefix) or not name.endswith(".sse.jsonl.gz"):
                continue
            scalar = json.loads(files[name.removesuffix(".sse.jsonl.gz") + ".json"])
            if scalar["success"]:
                first, last, usage, finish = reconstruct(raw)
                if (
                    usage != scalar["usage"]
                    or finish != scalar["finish_reason"]
                    or first != scalar["first_token"]["monotonic"]
                    or last != scalar["last_token"]["monotonic"]
                ):
                    raise ValueError("Warmup stream mismatch")
                error = None
            else:
                errors = []
                for line in gzip.decompress(raw).splitlines():
                    wire = json.loads(line)["raw"]
                    if wire.startswith("data:") and wire[5:].strip() != "[DONE]":
                        event = json.loads(wire[5:])
                        if event.get("error"):
                            errors.append(event["error"])
                if not errors:
                    raise ValueError("Failed request has no retained SSE error")
                error = errors
            rows.append(
                dict(
                    id=scalar["id"],
                    phase=scalar["phase"],
                    success=scalar["success"],
                    error=error,
                    source_sha256=sha(raw),
                )
            )
        item = dict(
            run=run,
            receipt=receipt,
            requests=len(rows),
            successful_warmup=sum(
                r["success"] and r["phase"].endswith("warmup") for r in rows
            ),
            successful_formal=sum(
                r["success"] and r["phase"].endswith("formal") for r in rows
            ),
            failed_requests=sum(not r["success"] for r in rows),
            rows=rows,
        )
        if rows:
            log = files[prefix + "service.log"].decode()
            needle = "torch.OutOfMemoryError: allocate_host_memory_slowpath:"
            if needle not in log or "aclrtMallocHostWithCfg" not in log:
                raise ValueError("Recorded root failure disappeared")
            item.update(
                runtime=json.loads(files[prefix + "runtime-identity.json"]),
                actual_failure="Pinned host-memory allocation failed in BetterScale state offload, followed by EngineDeadError and eight SSE failures. Not a proven disk-full or cgroup OOM-kill cause.",
                controller_followup="Receipt later reports Foreign NPU process after worker ancestry changed during failure. PID is the recorded own worker; this later observation is not the original engine failure.",
                activation=audit_activation(
                    files, prefix=prefix, expected_requests=len(rows)
                ),
                source_log_sha256=sha(files[prefix + "service.log"]),
                long16k_executed=False,
                formal_4k_completed=False,
            )
        failed.append(item)
    return dict(
        schema="fixed018-continuous-pair/v1",
        capsule_sha256=manifest["sha256"],
        b0=b0,
        b1_attempts=failed,
        status="B0_completed_B1_measured_host_allocation_failure",
        paired_continuous_stability_pass=False,
        paired_4k_throughput_gain_percent=None,
        paired_16k_throughput_gain_percent=None,
        scope="This candidate failed the prospective C8 synthetic continuous workload. No B1 16K run, no full paired throughput, no five-dataset overall verdict. Preserve the actual failure rather than labeling the entire attempt unmeasured.",
    )


if __name__ == "__main__":
    main()
