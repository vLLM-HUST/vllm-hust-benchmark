"""Open-loop serving benchmark with frozen plans and per-event SSE receipts.

Scenarios (all synthetic token-ID prompts, ignore_eos fixed output lengths):
  mixed   Poisson arrivals; short (in S_IN/out S_OUT) and long (L_IN/L_OUT) mix
  burst   groups of B requests arrive together every T seconds (mixed sizes)
  contend N steady decode streams + long prefills injected every T seconds
  short   Poisson short-only (decode heavy)

Usage:
  loadgen.py plan --scenario mixed --rate 1.0 --duration 120 --seed 1 --out plan.json
  loadgen.py run --plan plan.json --port 8300 --out result.json [--label x]
  loadgen.py report result.json [...] --ttft-slo 2.0 --tpot-slo 0.1 --gap-slo 0.5
"""

import argparse
import asyncio
import json
import math
import random
import statistics
import subprocess
import sys
import time

MODEL = "qwen3.5-35b"


def pct(values, p):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def make_plan(a):
    rng = random.Random(f"{a.scenario}:{a.seed}")
    reqs = []

    def add(t, kind, n_in, n_out):
        prompt = [rng.randrange(1000, 30000) for _ in range(n_in)]
        reqs.append({"id": f"r{len(reqs)}", "t": round(t, 4), "kind": kind,
                     "prompt": prompt, "max_tokens": n_out})

    def jitter(n):
        return max(16, int(n * rng.uniform(0.75, 1.25)))

    if a.scenario in ("mixed", "short"):
        t = 0.0
        while True:
            t += rng.expovariate(a.rate)
            if t >= a.duration:
                break
            if a.scenario == "mixed" and rng.random() < a.long_frac:
                add(t, "long", jitter(a.l_in), a.l_out)
            else:
                add(t, "short", jitter(a.s_in), jitter(a.s_out))
    elif a.scenario == "burst":
        t = 1.0
        while t < a.duration:
            for i in range(a.burst):
                if i < max(1, int(a.burst * a.long_frac)):
                    add(t, "long", jitter(a.l_in), a.l_out)
                else:
                    add(t, "short", jitter(a.s_in), jitter(a.s_out))
            t += a.interval
    elif a.scenario == "contend":
        # N decode streams long enough to cover the window, then periodic long P.
        out = int(a.duration * 40) + 64
        for i in range(a.streams):
            add(0.2 * i, "stream", jitter(a.s_in), out)
        t = 5.0
        while t < a.duration:
            add(t, "long", jitter(a.l_in), a.l_out)
            t += a.interval
    reqs.sort(key=lambda r: r["t"])
    return {"scenario": a.scenario, "params": {k: v for k, v in vars(a).items()
            if k not in ("cmd", "out")}, "requests": reqs}


async def one(client, port, r, t0, rows):
    import httpx  # noqa: F401
    await asyncio.sleep(max(0.0, t0 + r["t"] - time.perf_counter()))
    body = {"model": MODEL, "prompt": r["prompt"], "max_tokens": r["max_tokens"],
            "temperature": 0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    sent = time.perf_counter()
    events, usage, err = [], None, None
    try:
        async with client.stream("POST", f"http://127.0.0.1:{port}/v1/completions",
                                 json=body) as resp:
            if resp.status_code != 200:
                err = f"http {resp.status_code}: {(await resp.aread())[:200]!r}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[6:])
                    if d.get("usage"):
                        usage = d["usage"]
                    if any(c.get("text") for c in d.get("choices", [])):
                        events.append(time.perf_counter())
    except Exception as e:  # recorded, never retried
        err = repr(e)
    rows.append({"id": r["id"], "kind": r["kind"], "in": len(r["prompt"]),
                 "out": r["max_tokens"], "sched": t0 + r["t"], "sent": sent,
                 "events": events, "usage": usage, "error": err})


def npu_mem():
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                             timeout=20).stdout
        return [l for l in out.splitlines() if "VLLMWorker" in l]
    except Exception:
        return []


async def run(a):
    import httpx
    plan = json.load(open(a.plan))
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=1800, trust_env=False, limits=limits) as c:
        # warmup: one short and one long-ish request
        for n in (32, 2048):
            await c.post(f"http://127.0.0.1:{a.port}/v1/completions", json={
                "model": MODEL, "prompt": [1000 + i for i in range(n)],
                "max_tokens": 8, "temperature": 0, "ignore_eos": True})
        rows = []
        t0 = time.perf_counter() + 0.5
        await asyncio.gather(*(one(c, a.port, r, t0, rows) for r in plan["requests"]))
        end = time.perf_counter()
    json.dump({"label": a.label, "plan": a.plan, "scenario": plan["scenario"],
               "params": plan["params"], "t0": t0, "end": end,
               "npu_mem": npu_mem(), "rows": rows}, open(a.out, "w"))
    print(json.dumps(summarize(json.load(open(a.out)), a), indent=1))


def summarize(res, a):
    rows = res["rows"]
    ok = [r for r in rows if not r["error"] and r["events"]]
    errs = len(rows) - len(ok)
    ttft = [r["events"][0] - r["sched"] for r in ok]
    tpot, gaps_all, maxgap, good = [], [], [], 0
    by_kind = {}
    for r in ok:
        ev = r["events"]
        g = [y - x for x, y in zip(ev, ev[1:])]
        gaps_all += g
        n_out = (r["usage"] or {}).get("completion_tokens") or r["out"]
        tp = (ev[-1] - ev[0]) / (n_out - 1) if n_out > 1 else 0.0
        tpot.append(tp)
        mg = max(g) if g else 0.0
        maxgap.append(mg)
        t_ok = ev[0] - r["sched"] <= (a.ttft_slo_long if r["kind"] == "long" else a.ttft_slo)
        meets = t_ok and tp <= a.tpot_slo and mg <= a.gap_slo
        good += meets
        k = by_kind.setdefault(r["kind"], {"n": 0, "good": 0, "ttft": [], "tpot": [], "maxgap": []})
        k["n"] += 1
        k["good"] += meets
        k["ttft"].append(ev[0] - r["sched"])
        k["tpot"].append(tp)
        k["maxgap"].append(mg)
    span = max(r["events"][-1] for r in ok) - res["t0"] if ok else float("nan")
    out_tok = sum((r["usage"] or {}).get("completion_tokens") or 0 for r in ok)
    in_tok = sum(r["in"] for r in ok)
    lag = [r["sent"] - r["sched"] for r in rows]
    f = lambda x: None if x is None else round(x, 4)  # noqa: E731
    return {
        "label": res.get("label"), "scenario": res["scenario"], "requests": len(rows),
        "errors": errs, "span_s": f(span), "out_tok_s": f(out_tok / span),
        "total_tok_s": f((out_tok + in_tok) / span), "req_s": f(len(ok) / span),
        "goodput_req": good, "goodput_frac": f(good / len(rows)) if rows else None,
        "goodput_req_s": f(good / span),
        "ttft_p50": f(pct(ttft, 50)), "ttft_p90": f(pct(ttft, 90)), "ttft_p99": f(pct(ttft, 99)),
        "tpot_mean": f(statistics.mean(tpot)) if tpot else None,
        "tpot_p50": f(pct(tpot, 50)), "tpot_p99": f(pct(tpot, 99)),
        "itl_p50": f(pct(gaps_all, 50)), "itl_p99": f(pct(gaps_all, 99)),
        "itl_p999": f(pct(gaps_all, 99.9)), "worst_stall": f(max(gaps_all) if gaps_all else None),
        "maxgap_p50": f(pct(maxgap, 50)), "maxgap_p90": f(pct(maxgap, 90)),
        "send_lag_max": f(max(lag) if lag else None),
        "by_kind": {k: {"n": v["n"], "good": v["good"], "ttft_p50": f(pct(v["ttft"], 50)),
                        "ttft_p99": f(pct(v["ttft"], 99)), "tpot_p50": f(pct(v["tpot"], 50)),
                        "tpot_p99": f(pct(v["tpot"], 99)), "maxgap_p90": f(pct(v["maxgap"], 90))}
                    for k, v in by_kind.items()},
        "npu_mem": res.get("npu_mem"),
    }


def slo_args(p):
    p.add_argument("--ttft-slo", type=float, default=1.0)
    p.add_argument("--ttft-slo-long", type=float, default=3.0)
    p.add_argument("--tpot-slo", type=float, default=0.1)
    p.add_argument("--gap-slo", type=float, default=0.5)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--scenario", choices=["mixed", "burst", "contend", "short"], required=True)
    p.add_argument("--rate", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=120)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--long-frac", type=float, default=0.2)
    p.add_argument("--s-in", type=int, default=256)
    p.add_argument("--s-out", type=int, default=128)
    p.add_argument("--l-in", type=int, default=8000)
    p.add_argument("--l-out", type=int, default=32)
    p.add_argument("--burst", type=int, default=8)
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--streams", type=int, default=6)
    p.add_argument("--out", required=True)
    r = sp.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--label", default="")
    slo_args(r)
    rep = sp.add_parser("report")
    rep.add_argument("files", nargs="+")
    rep.add_argument("--full", action="store_true")
    slo_args(rep)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan = make_plan(a)
        json.dump(plan, open(a.out, "w"))
        kinds = {}
        for q in plan["requests"]:
            kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
        print(a.out, len(plan["requests"]), kinds)
    elif a.cmd == "run":
        asyncio.run(run(a))
    else:
        for fn in a.files:
            s = summarize(json.load(open(fn)), a)
            if not a.full:
                s.pop("by_kind"); s.pop("npu_mem")  # noqa: E702
            print(json.dumps(s))


if __name__ == "__main__":
    sys.exit(main())
