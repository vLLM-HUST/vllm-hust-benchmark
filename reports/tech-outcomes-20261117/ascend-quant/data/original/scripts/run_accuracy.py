#!/usr/bin/env python3
"""Run the frozen 12,032-case MMLU-Pro plan against a plain vLLM server.

This is a local reimplementation of the transport in the published
`reproduction/mmlu_pro_run.py.txt` (the downloaded copy is never executed).
Prompt construction, selection and scoring come from the project adapter
`vllm_hust_benchmark/mmlu_pro_pyramidkv.py`, unchanged, so the scorer identity
is the frozen `scorer_source_sha256`.

Protocol (identical to the published BF16 precheck):
  * /v1/completions, prompt = frozen token-ID list, max_tokens 2048
  * temperature 0, seed 17, ignore_eos false
  * concurrency 16, no retries; a transport failure is kept and scored wrong
  * per-case JSON kept for every request, resume supported

Example:
  python -I run_accuracy.py --plan plan --base-url http://127.0.0.1:8201 \
      --served-model qwen35-a3b-w8a8 --label w8a8 \
      --output results/w8a8-run1 --concurrency 16
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import importlib.util
import json
import pathlib
import sys
import threading
import time
import urllib.request

ADAPTER_PATH = pathlib.Path(
    "/data/jxd/whb/src/vllm_hust_benchmark/mmlu_pro_pyramidkv.py"
)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("mmlu_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_plan(path, adapter):
    contract = json.loads((path / "contract.json").read_text())
    if contract["scorer_source_sha256"] != sha_bytes(ADAPTER_PATH.read_bytes()):
        raise SystemExit("adapter changed after contract freeze")
    for name, digest in contract["files_sha256"].items():
        got = sha_bytes((path / name).read_bytes())
        if got != digest:
            raise SystemExit("plan file hash mismatch: %s" % name)
    cases = json.loads((path / "cases.json").read_text())
    if len(cases) != contract["tasks"] or len({c["case_id"] for c in cases}) != len(cases):
        raise SystemExit("task manifest mismatch")
    for c in cases:
        if len(c["prompt"]) != c["prompt_tokens"]:
            raise SystemExit("prompt token count mismatch: %s" % c["case_id"])
        digest = sha_bytes(json.dumps(c["prompt"], separators=(",", ":")).encode())
        if digest != c["prompt_sha256"]:
            raise SystemExit("prompt hash mismatch: %s" % c["case_id"])
    return contract, cases


def one_request(base_url, model, case, gen, timeout):
    body = {
        "model": model,
        "prompt": case["prompt"],
        "max_tokens": case["max_tokens"],
        "temperature": gen["temperature"],
        "seed": gen["seed"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": case["ignore_eos"],
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    rec = {
        "case_id": case["case_id"],
        "prompt_sha256": case["prompt_sha256"],
        "prompt_tokens": case["prompt_tokens"],
        "max_tokens": case["max_tokens"],
        "ignore_eos": case["ignore_eos"],
        "success": False,
        "output": "",
        "finish_reason": None,
        "error": None,
        "usage": None,
    }
    t0 = time.monotonic()
    t_first = None
    parts = []
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    rec["usage"] = chunk["usage"]
                for ch in chunk.get("choices", []):
                    if ch.get("text"):
                        if t_first is None:
                            t_first = time.monotonic()
                        parts.append(ch["text"])
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
        rec["output"] = "".join(parts)
        rec["success"] = rec["finish_reason"] is not None
        if not rec["success"]:
            rec["error"] = "stream ended without finish_reason"
    except Exception as exc:  # noqa: BLE001 - failures are kept and scored wrong
        rec["output"] = "".join(parts)
        rec["error"] = repr(exc)
    t1 = time.monotonic()
    rec["e2e_seconds"] = t1 - t0
    rec["ttft_seconds"] = (t_first - t0) if t_first else None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, type=pathlib.Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:8201")
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    adapter = load_adapter(ADAPTER_PATH)
    contract, cases = load_plan(a.plan, adapter)
    gen = contract["generation"]
    if a.limit:
        cases = cases[: a.limit]
    out = a.output
    if out.exists() and not a.resume:
        sys.exit("output exists; use a new directory (or --resume)")
    (out / "requests").mkdir(parents=True, exist_ok=True)

    meta = {
        "runner": "run_accuracy.py (local plain-server transport)",
        "runner_sha256": sha_bytes(pathlib.Path(__file__).read_bytes()),
        "adapter": str(ADAPTER_PATH),
        "adapter_sha256": sha_bytes(ADAPTER_PATH.read_bytes()),
        "contract_sha256": sha_bytes((a.plan / "contract.json").read_bytes()),
        "label": a.label,
        "base_url": a.base_url,
        "served_model": a.served_model,
        "concurrency": a.concurrency,
        "generation": gen,
        "n_cases": len(cases),
        "limit": a.limit,
        "resume": a.resume,
        "started_unix": time.time(),
        "deviations": ["concurrency 16 instead of contract 1"],
    }
    if a.limit:
        meta["WARNING"] = "smoke test only - not a score"
    (out / ("run.%d.json" % int(time.time()))).write_text(json.dumps(meta, indent=1))

    done = {}
    lock = threading.Lock()
    todo = []
    for c in cases:
        f = out / "requests" / (c["case_id"] + ".json")
        if a.resume and f.exists():
            r = json.loads(f.read_text())
            if r.get("success"):
                done[c["case_id"]] = r
                continue
        todo.append(c)
    print(
        "cases=%d already_done=%d to_run=%d concurrency=%d"
        % (len(cases), len(done), len(todo), a.concurrency),
        flush=True,
    )

    t_start = time.monotonic()

    def work(c):
        r = one_request(a.base_url, a.served_model, c, gen, a.timeout)
        (out / "requests" / (c["case_id"] + ".json")).write_text(
            json.dumps(r, ensure_ascii=False)
        )
        with lock:
            done[c["case_id"]] = r
            n = len(done)
            if n % 100 == 0 or n == len(cases):
                ok = sum(1 for x in done.values() if x["success"])
                print(
                    "%d/%d ok=%d elapsed=%.0fs"
                    % (n, len(cases), ok, time.monotonic() - t_start),
                    flush=True,
                )

    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(work, todo))
    wall = time.monotonic() - t_start

    results = [done[c["case_id"]] for c in cases]
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False))
    scores = adapter.score_results(results, cases)
    (out / "score.json").write_text(json.dumps(scores, indent=1, ensure_ascii=False))

    by_id = {c["case_id"]: c for c in cases}
    rec_by_id = {r["case_id"]: r for r in results}
    with open(out / "per-case-outcomes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "case_id",
                "category",
                "gold",
                "prediction",
                "valid",
                "correct",
                "transport_success",
                "finish_reason",
                "prompt_tokens",
                "completion_tokens",
            ]
        )
        for row in scores["outcomes"]:
            case = by_id[row["case_id"]]
            rec = rec_by_id[row["case_id"]]
            usage = rec.get("usage") or {}
            w.writerow(
                [
                    row["case_id"],
                    row["category"],
                    case["answer"],
                    row["prediction"],
                    int(row["valid"]),
                    int(row["correct"]),
                    int(row["transport_success"]),
                    rec.get("finish_reason"),
                    case["prompt_tokens"],
                    usage.get("completion_tokens"),
                ]
            )

    ok = [r for r in results if r["success"]]
    summary = {
        "label": a.label,
        "cases": len(cases),
        "correct": scores["correct"],
        "accuracy": scores["accuracy"],
        "invalid_answers": scores["invalid_answers"],
        "transport_failures": scores["transport_failures"],
        "length_limited": scores["length_limited"],
        "wall_seconds_this_invocation": wall,
        "mean_ttft_s": sum(r["ttft_seconds"] for r in ok if r["ttft_seconds"])
        / max(1, len(ok)),
        "mean_e2e_s": sum(r["e2e_seconds"] for r in ok) / max(1, len(ok)),
        "smoke_only": bool(a.limit),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if scores["transport_failures"]:
        sys.exit("transport failures present (kept, scored incorrect)")


if __name__ == "__main__":
    main()
