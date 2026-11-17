"""Audit archived measurements without launching models or modifying evidence."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def prom(path):
    result = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            key, value, *_ = line.split()
            result[key] = float(value)
    return result


def audit(root, plugin):
    result = {
        "scope": "offline re-audit of complete historical runs; no new hardware test",
        "arms": {},
        "source_payload_files": 0,
    }
    identity = load(root / "campaign-final-identity.json")
    for name, expected in identity["plugin_sha256"].items():
        if name.startswith("src/"):
            assert (
                hashlib.sha256((plugin / name).read_bytes()).hexdigest() == expected
            ), name
            result["source_payload_files"] += 1
    assert result["source_payload_files"] == 18
    measured = root / "pressure-measurements-v1"
    reference = load(measured / "analysis.json")
    signatures = []
    for arm in ("pressure-native", "pressure-default", "pressure-early"):
        client = measured / (arm + "-client")
        rows = [
            json.loads(line)
            for line in (client / "requests.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 2215
        phases = sorted(client.glob("*-summary.json"))
        assert sum(load(p)["completed"] for p in phases) == len(rows)
        phase_wall = sum(load(p)["wall_s"] for p in phases)
        signature = []
        for row in rows:
            body = load(client / "requests" / (row["key"] + "-input.json"))
            prompt = body["prompt"]
            digest = hashlib.sha256(
                json.dumps(prompt, separators=(",", ":")).encode()
            ).hexdigest()
            assert digest == row["input_sha256"] == row["prompt_sha256"]
            assert len(prompt) == row["prompt_tokens"] == row["usage"]["prompt_tokens"]
            assert body["max_tokens"] == row["usage"]["completion_tokens"] == 32
            assert body["temperature"] == 0 and body["ignore_eos"] is True
            assert row["http_status"] == 200 and row["done"] == 1
            assert row["finish_reason"] == "length" and row["valid"] is True
            assert row["usage"]["total_tokens"] == len(prompt) + 32
            assert abs(row["ttft_s"] - row["chunk_times_s"][0]) < 1e-8
            assert abs(row["tpot_s"] - (row["wall_s"] - row["ttft_s"]) / 31) < 1e-8
            done, text, usage, finish = 0, "", None, None
            for line in (
                (client / "requests" / (row["key"] + ".sse")).read_bytes().splitlines()
            ):
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done += 1
                    continue
                event = json.loads(data)
                assert not event.get("error")
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    text += choice.get("text") or ""
                    finish = choice.get("finish_reason") or finish
            assert done == 1 and usage == row["usage"] and finish == "length"
            assert hashlib.sha256(text.encode()).hexdigest() == row["output_sha256"]
            signature.append((row["key"], row["task_id"], row["source_index"], digest))
        signatures.append(sorted(signature))
        counters = {}
        for phase in phases:
            name = load(phase)["name"]
            before, after = (
                prom(client / (name + "-before.prom")),
                prom(client / (name + "-after.prom")),
            )
            for key, end in after.items():
                metric = key.split("{")[0]
                if metric.endswith(("_total", "_sum", "_count")):
                    counters[metric] = (
                        counters.get(metric, 0) + end - before.get(key, 0)
                    )
        actual = {
            "requests": len(rows),
            "logical_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "output_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
            "phase_wall_s_sum": phase_wall,
            "ttft_mean_s": statistics.mean(r["ttft_s"] for r in rows),
            "tpot_mean_s": statistics.mean(r["tpot_s"] for r in rows),
            "counters": counters,
        }
        actual["output_tok_s"] = actual["output_tokens"] / phase_wall
        expected = reference["arms"][arm]
        for name in (
            "requests",
            "logical_prompt_tokens",
            "output_tokens",
            "phase_wall_s_sum",
            "ttft_mean_s",
            "tpot_mean_s",
            "output_tok_s",
        ):
            assert abs(actual[name] - expected[name]) < 1e-7, (arm, name)
        for name, value in expected["counters"].items():
            assert abs(counters.get(name, 0) - value) < 1e-6, (arm, name)
        result["arms"][arm] = actual
    assert signatures[0] == signatures[1] == signatures[2]
    result["all_inputs_identical"] = True
    result["all_requests_and_streams_valid"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.extracted_root, args.plugin)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Archived source, input, SSE, timing and metric audits passed")


if __name__ == "__main__":
    main()
