"""Replay archived real SSE/phase clocks; never starts a model or fills missing profiles."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path


def replay(capsule):
    with tarfile.open(capsule) as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    manifest = json.loads(files["manifest.json"])
    assert manifest["complete_five"] is False
    for n, v in manifest["files"].items():
        assert (
            len(files[n]) == v["bytes"]
            and hashlib.sha256(files[n]).hexdigest() == v["sha256"]
        ), n

    def read(n):
        return json.loads(files[n])

    def stream(n):
        first = last = usage = finish = None
        done = False
        prior = -math.inf
        for line in gzip.decompress(files[n]).splitlines():
            row = json.loads(line)
            stamp = row["clock"]["monotonic"]
            assert stamp >= prior
            prior = stamp
            if not row["raw"].startswith("data:"):
                continue
            payload = row["raw"][5:].strip()
            if payload == "[DONE]":
                done = True
                continue
            event = json.loads(payload)
            assert not event.get("error")
            usage = event.get("usage") or usage
            for c in event.get("choices", []):
                delta = c.get("delta", {})
                if any(
                    delta.get(k) for k in ["content", "reasoning", "reasoning_content"]
                ):
                    first = stamp if first is None else first
                    last = stamp
                finish = c.get("finish_reason") or finish
        assert done and first is not None and usage and finish
        return first, last, usage, finish

    def requests(label, dataset=False):
        root = manifest["original_arms"][label]["original"]
        prefix = label + ("/service/evaluation/requests/" if dataset else "/service/")
        out = []
        for name in sorted(files):
            if (
                not name.startswith(prefix)
                or not name.endswith(".json")
                or "/" in name[len(prefix) :]
            ):
                continue
            row = read(name)
            if dataset:
                assert row["status"] == "ok"
                wire = (
                    label
                    + "/"
                    + row["stream_capture"].rsplit("/" + Path(root).name + "/", 1)[1]
                )
                assert wire.startswith(label + "/service/transport/")
                assert hashlib.sha256(files[wire]).hexdigest() == row["stream_sha256"]
            else:
                if (
                    not isinstance(row, dict)
                    or not all(k in row for k in ["success", "start", "end"])
                    or not ("phase" in row or "case_id" in row)
                ):
                    continue
                assert row["success"]
                wire = name[:-5] + ".sse.jsonl.gz"
            first, last, usage, finish = stream(wire)
            assert (first, last, usage, finish) == (
                row["first_token"]["monotonic"],
                row["last_token"]["monotonic"],
                row["usage"],
                row["finish_reason"],
            )
            assert row["start"]["monotonic"] <= first <= last <= row["end"]["monotonic"]
            out.append(row)
        return out

    def metrics(label, records, start, end):
        assert end > start and records
        output = sum(x["usage"]["completion_tokens"] for x in records)
        ttft = [
            (x["first_token"]["monotonic"] - x["start"]["monotonic"]) * 1000
            for x in records
        ]
        tpot = [
            (x["end"]["monotonic"] - x["first_token"]["monotonic"])
            * 1000
            / (x["usage"]["completion_tokens"] - 1)
            for x in records
            if x["usage"]["completion_tokens"] > 1
        ]
        return dict(
            role="B0" if label.endswith("B0") else "r9",
            requests=len(records),
            failed=0,
            output_tokens=output,
            duration_seconds=end - start,
            output_tokens_per_second=output / (end - start),
            mean_ttft_ms=statistics.fmean(ttft),
            mean_tpot_ms=statistics.fmean(tpot),
            truncated=sum(x["finish_reason"] == "length" for x in records),
            source_seal_sha256=manifest["original_arms"][label]["seal_sha256"],
        )

    result = []
    for name, label, count in [
        ("mmlu", "MMLU200", 200),
        ("frontier", "Frontier160", 160),
    ]:
        arms = []
        for role in ["B0", "B1"]:
            tag = name + "-" + role
            records = requests(tag, dataset=name == "frontier")
            assert len(records) == count
            if name == "mmlu":
                phase = read(tag + "/service/joint.json")
                start, end = [phase[k]["monotonic"] for k in ["start", "end"]]
            else:
                start = min(x["start"]["monotonic"] for x in records)
                end = max(x["end"]["monotonic"] for x in records)
            arms.append(metrics(tag, records, start, end))
        result.append(
            dict(
                id=name,
                label=label,
                control="B0",
                candidate="r9",
                baseline_id="original018-cann8.5.1",
                arms=arms,
            )
        )
    continuous = {r: requests("C8-" + r) for r in ["B0", "B1"]}
    for phase_name in ["4096-formal", "16384-formal"]:
        arms = []
        for role in ["B0", "B1"]:
            tag = "C8-" + role
            phase = next(
                p
                for p in read(tag + "/service/continuous.json")["phases"]
                if p["name"] == phase_name
            )
            records = [x for x in continuous[role] if x["phase"] == phase_name]
            assert len(records) == phase["requests"] and phase["failed"] == 0
            arms.append(
                metrics(
                    tag, records, phase["start"]["monotonic"], phase["end"]["monotonic"]
                )
            )
        result.append(
            dict(
                id=phase_name,
                label=("4K" if phase_name.startswith("4096") else "16K") + " / C8",
                control="B0",
                candidate="r9",
                baseline_id="original018-cann8.5.1",
                arms=arms,
            )
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capsule", required=True, type=Path)
    p.add_argument("--check", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    rows = replay(a.capsule)
    if a.check:
        expected = json.loads(a.check.read_text())
        assert (
            hashlib.sha256(a.capsule.read_bytes()).hexdigest()
            == expected["raw_asset"]["sha256"]
        )
        for actual, wanted in zip(rows, expected["measurements"], strict=True):
            assert actual["id"] == wanted["id"]
            for x, y in zip(actual["arms"], wanted["arms"], strict=True):
                for k, v in x.items():
                    assert (
                        math.isclose(v, y[k], rel_tol=1e-12)
                        if isinstance(v, float)
                        else v == y[k]
                    ), (actual["id"], k)
    if a.out:
        a.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print("REPLAY_OK", len(rows), "paired profiles; zero new inference")
