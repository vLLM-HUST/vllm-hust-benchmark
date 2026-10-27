"""Graph-mode engines with GPU APC, CPU cache and staged P/D handoff."""

from __future__ import annotations

import argparse
import heapq
import itertools
import math
from collections import Counter, OrderedDict, deque
from pathlib import Path

from pd_serial_simulator import input_path, quantiles, read, write
from step_simulator import CostModel, Job, KVPool, Request, features

BLOCK_BYTES = 128 * 57344
DOMAIN = "Qwen2.5-7B-Instruct:BF16"


def interpolate(rows, tokens, field):
    if tokens == 0:
        return 0.0
    points = [(0, 0.0), *sorted((row["tokens"], row[field]) for row in rows)]
    lower = max((point for point in points if point[0] <= tokens), default=points[-2])
    upper = min((point for point in points if point[0] >= tokens), default=points[-1])
    if lower == upper:
        if tokens == lower[0]:
            return lower[1]
        lower, upper = points[-2:]
    fraction = (tokens - lower[0]) / (upper[0] - lower[0])
    return lower[1] + fraction * (upper[1] - lower[1])


def memory_budget(
    hbm_gib=64, utilization=0.9, measured_peak_mib=35370, measured_kv_gib=16
):
    non_kv = measured_peak_mib / 1024 - measured_kv_gib
    kv = math.floor(hbm_gib * utilization - non_kv - 1.0)
    return {
        "HBM_GiB": hbm_gib,
        "utilization": utilization,
        "measured_non_KV_GiB": non_kv,
        "extra_margin_GiB": 1.0,
        "NPU_KV_GiB": kv,
        "host_memory_GiB_per_node": 64,
        "CPU_pool_GiB_per_instance": 24,
        "host_headroom_GiB_per_node": 16,
    }


class HostCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.keys = OrderedDict()
        self.pins = Counter()
        self.peak, self.evictions = 0, 0

    def prefix(self, hashes):
        for position, key in enumerate(hashes):
            if key not in self.keys:
                return position
        return len(hashes)

    def add(self, keys):
        for key in keys:
            if key in self.keys:
                self.keys.move_to_end(key)
                continue
            while (len(self.keys) + 1) * BLOCK_BYTES > self.capacity:
                victim = next((key for key in self.keys if not self.pins[key]), None)
                if victim is None:
                    raise ValueError("Pinned CPU cache cannot fit handoff")
                del self.keys[victim]
                self.evictions += 1
            self.keys[key] = None
        self.peak = max(self.peak, len(self.keys) * BLOCK_BYTES)


def seed_gpu(pool, hashes):
    """Publish a completed host-to-device restore into unreferenced GPU pages."""
    for logical in hashes:
        key = (DOMAIN, logical)
        if pool.cached.get(key):
            continue
        if not pool.free:
            raise ValueError("GPU restore requires an uncalibrated preemption")
        index, _ = pool.free.popitem(last=False)
        block = pool.blocks[index]
        if block.key is not None:
            previous = pool.cached[block.key]
            del previous[index]
            if not previous:
                del pool.cached[block.key]
            pool.evictions += 1
        block.key = key
        pool.cached.setdefault(key, OrderedDict())[index] = None
        pool.free[index] = None


class ServingSimulation:
    def __init__(
        self,
        bundle,
        protocol,
        step_model,
        components,
        *,
        p_tokens=8192,
        d_tokens=2048,
        max_sequences=128,
        overlap=True,
        npu_gib=None,
        p_count=2,
        d_count=2,
    ):
        self.rows = {row["id"]: row for row in bundle["rows"]}
        if any(
            set(row) != {"id", "task", "tokens", "hashes"} for row in bundle["rows"]
        ):
            raise ValueError("Unexpected workload fields")
        if min(p_tokens, d_tokens, max_sequences, p_count, d_count) < 1:
            raise ValueError("Positive serving budgets required")
        self.p_count, self.d_count = p_count, d_count
        self.node_count = p_count + d_count
        self.order, self.protocol = protocol["test_order"], protocol
        if list(self.rows) != self.order:
            raise ValueError("Workload order differs")
        self.tasks = sorted({row["task"] for row in self.rows.values()})
        self.offered = dict(
            zip(
                self.order,
                protocol["arrival_condition"]["offered_arrivals_seconds"],
                strict=True,
            )
        )
        self.budget = memory_budget()
        if npu_gib is not None:
            self.budget["NPU_KV_GiB"] = npu_gib
        blocks = int(self.budget["NPU_KV_GiB"] * 2**30 // BLOCK_BYTES)
        self.pools = [KVPool(blocks, 128) for _ in range(self.node_count)]
        self.host = [
            HostCache(self.budget["CPU_pool_GiB_per_instance"] * 2**30)
            for _ in range(self.node_count)
        ]
        for node, payload in enumerate(bundle["initial_inventory"]):
            self.host[node].add(
                bytes(block["key"]["hash"]).hex()
                for block in payload["cache"]["blocks"]
            )
        self.model = CostModel(step_model)
        self.native = components["native"]
        self.p_tokens, self.d_tokens = p_tokens, d_tokens
        self.max_sequences, self.overlap = max_sequences, overlap
        self.network_s_per_byte = bundle["costs"]["seconds_per_pd_byte"]
        self.events, self.sequence, self.now = [], itertools.count(), 0.0
        self.waiting = [deque() for _ in range(self.node_count)]
        self.running, self.preparing = [[] for _ in range(self.node_count)], set()
        self.engine_busy, self.copy_busy = set(), Counter()
        self.engine_until, self.device_copy_free = {}, {}
        self.resource_free, self.jobs = {}, {}
        self.arrived, self.live_tasks, self.sessions = [], set(), {}
        self.finished, self.steps, self.transfers = {}, [], []
        self.peak_live = 0
        self.peak_running, self.peak_resident = (
            [0] * self.node_count,
            [0] * self.node_count,
        )
        self.extrapolated_steps = 0
        for key in self.order:
            self.event(self.offered[key], "arrival", key)

    def event(self, at, kind, key, data=None):
        if at < self.now - 1e-8:
            raise ValueError("Noncausal event")
        heapq.heappush(self.events, (at, next(self.sequence), kind, key, data))

    def submit_ready(self):
        remaining = []
        for key in self.arrived:
            row = self.rows[key]
            if row["task"] in self.live_tasks:
                remaining.append(key)
                continue
            lane = self.tasks.index(row["task"]) % self.p_count
            self.live_tasks.add(row["task"])
            self.jobs[key] = {
                "id": key,
                "P": lane,
                "D": self.p_count + lane % self.d_count,
                "offered": self.offered[key],
                "ready": max(self.offered[key], self.sessions.get(row["task"], 0.0)),
                "stages": {},
                "engine_jobs": {},
            }
            self.prepare(key, lane)
        self.arrived = remaining
        self.peak_live = max(self.peak_live, len(self.jobs))

    def resource(self, resource, seconds, kind, key, data, *, payload=0, node=None):
        start = max(self.now, self.resource_free.get(resource, 0.0))
        if node is not None and not self.overlap:
            start = max(
                start,
                self.engine_until.get(node, 0.0),
                self.device_copy_free.get(node, 0.0),
            )
        end = start + seconds
        self.resource_free[resource] = end
        self.transfers.append(
            {
                "id": key,
                "resource": resource,
                "queued": self.now,
                "start": start,
                "end": end,
                "bytes": payload,
            }
        )
        if node is not None:
            self.copy_busy[node] += 1
            self.device_copy_free[node] = end
        self.event(end, kind, key, data)

    def prepare(self, key, node):
        row, live = self.rows[key], self.jobs[key]
        request = Request(
            key,
            self.now,
            row["tokens"],
            1 if node < self.p_count else 128,
            tuple(row["hashes"]),
            DOMAIN,
            task=row["task"],
        )
        job = Job(request)
        live["engine_jobs"][node] = job
        self.preparing.add((node, key))
        gpu = len(self.pools[node].prefix(job))
        target = (
            max(gpu, self.host[node].prefix(row["hashes"]))
            if node < self.p_count
            else len(row["hashes"])
        )
        live["stages"]["P" if node < self.p_count else "D"] = {
            "prepare": self.now,
            "node": node,
            "NPU_initial_prefix_blocks": gpu,
            "target_prefix_blocks": target,
            "restore_bytes": (target - gpu) * BLOCK_BYTES,
        }
        if node >= self.p_count:
            missing = [
                h for h in row["hashes"][gpu:target] if h not in self.host[node].keys
            ]
            source = live["P"]
            if not all(h in self.host[source].keys for h in missing):
                raise ValueError("P completed without publishing the required handoff")
            self.host[source].pins.update(missing)
            if missing:
                payload = len(missing) * BLOCK_BYTES
                query = interpolate(
                    self.native, min(row["tokens"], 24576), "query_seconds"
                )
                self.resource(
                    "RDMA:P-to-D",
                    query + payload * self.network_s_per_byte,
                    "remote_done",
                    key,
                    (node, gpu, target, source, missing),
                    payload=payload,
                )
                return
        self.restore(key, node, gpu, target)

    def restore(self, key, node, gpu, target):
        if target == gpu:
            self.admit(key, node)
            return
        tokens = (target - gpu) * 128
        seconds = interpolate(self.native, tokens, "load_complete_seconds")
        self.resource(
            f"H2D:{node}",
            seconds,
            "restore_done",
            key,
            (node, gpu, target),
            payload=tokens * 57344,
            node=node,
        )

    def admit(self, key, node):
        self.preparing.remove((node, key))
        self.waiting[node].append(self.jobs[key]["engine_jobs"][node])
        self.jobs[key]["stages"]["P" if node < self.p_count else "D"][
            "engine_queued"
        ] = self.now

    def schedule(self, node):
        if node in self.engine_busy or (not self.overlap and self.copy_busy[node]):
            return
        pool, running = self.pools[node], self.running[node]
        budget = self.p_tokens if node < self.p_count else self.d_tokens
        scheduled = []
        for job in running:
            take = min(budget, job.request.prompt_tokens + job.generated - job.computed)
            if take < 1:
                continue
            if not pool.allocate(job, take):
                raise ValueError("Active NPU growth requires uncalibrated preemption")
            scheduled.append((job, take, job.computed))
            budget -= take
        while self.waiting[node] and budget and len(running) < self.max_sequences:
            job = self.waiting[node][0]
            prefix = pool.prefix(job)
            computed = len(prefix) * 128
            take = min(budget, job.request.prompt_tokens + job.generated - computed)
            if not pool.allocate(job, take, prefix):
                break
            self.waiting[node].popleft()
            running.append(job)
            job.admitted_s = self.now
            live = self.jobs[job.request.id]
            live["stages"]["P" if node < self.p_count else "D"].update(
                engine_start=self.now, cached_tokens=job.cached_tokens
            )
            scheduled.append((job, take, computed))
            budget -= take
        if not scheduled:
            if self.waiting[node] and not running:
                raise ValueError("No feasible GPU admission")
            return
        row = features(
            [
                (before, take, job.request.prompt_tokens)
                for job, take, before in scheduled
            ]
        )
        duration = self.model.duration(row)
        self.extrapolated_steps += (
            row["prefill_tokens"] + row["decode_tokens"] > 2048 or row["sequences"] > 8
        )
        self.steps.append(
            {
                "node": node,
                "start": self.now,
                "end": self.now + duration,
                **row,
                "scheduled": {j.request.id: n for j, n, _ in scheduled},
                "running": len(running),
                "waiting": len(self.waiting[node]),
            }
        )
        self.peak_running[node] = max(self.peak_running[node], len(running))
        self.engine_busy.add(node)
        self.engine_until[node] = self.now + duration
        self.event(self.now + duration, "step_done", str(node), (node, scheduled))

    def step_done(self, node, scheduled):
        for job, take, before in scheduled:
            target = job.request.prompt_tokens + job.generated
            job.computed = before + take
            if job.computed < target:
                continue
            job.generated += 1
            key = job.request.id
            if job.first_s is None:
                job.first_s = self.now + self.model.data["response_delay_s"]
                if node >= self.p_count:
                    self.jobs[key]["first"] = job.first_s
            if job.generated == job.request.output_tokens:
                self.running[node].remove(job)
                job.finish_s = self.now + self.model.data["response_delay_s"]
                if node >= self.p_count:
                    self.pools[node].release(job)
                self.event(job.finish_s, "request_engine_done", key, node)
        self.event(
            self.now + self.model.data["dispatch_gap_s"],
            "engine_ready",
            str(node),
            node,
        )

    def engine_done(self, key, node):
        live, row = self.jobs[key], self.rows[key]
        live["stages"]["P" if node < self.p_count else "D"]["engine_end"] = self.now
        if node < self.p_count:
            missing = [h for h in row["hashes"] if h not in self.host[node].keys]
            seconds = interpolate(self.native, len(missing) * 128, "save_RPC_seconds")
            self.resource(
                f"D2H:{node}",
                seconds,
                "save_done",
                key,
                (node, missing),
                payload=len(missing) * BLOCK_BYTES,
                node=node,
            )
        else:
            live.update(
                end=self.now,
                TTFT=live["first"] - live["ready"],
                E2E=self.now - live["ready"],
            )
            live["TPOT_ms"] = (live["end"] - live["first"]) / 127 * 1000
            self.finished[key] = {k: v for k, v in live.items() if k != "engine_jobs"}
            self.sessions[row["task"]] = self.now
            self.live_tasks.remove(row["task"])
            del self.jobs[key]

    def extra_event(self, kind, key, data):
        return False

    def run(self):
        while self.events:
            self.now, _, kind, key, data = heapq.heappop(self.events)
            if kind == "arrival":
                self.arrived.append(key)
            elif kind == "remote_done":
                node, gpu, target, source, missing = data
                self.host[node].add(missing)
                self.host[source].pins.subtract(missing)
                self.restore(key, node, gpu, target)
            elif kind == "restore_done":
                node, gpu, target = data
                self.copy_busy[node] -= 1
                seed_gpu(self.pools[node], self.rows[key]["hashes"][gpu:target])
                self.admit(key, node)
            elif kind == "save_done":
                node, missing = data
                self.copy_busy[node] -= 1
                self.host[node].add(missing)
                self.pools[node].release(self.jobs[key]["engine_jobs"][node])
                self.prepare(key, self.jobs[key]["D"])
            elif kind == "step_done":
                self.step_done(*data)
            elif kind == "engine_ready":
                self.engine_busy.remove(data)
            elif kind == "request_engine_done":
                self.engine_done(key, data)
            elif self.extra_event(kind, key, data):
                pass
            else:
                raise ValueError(f"Unknown event {kind}")
            self.submit_ready()
            for node in range(self.node_count):
                self.peak_resident[node] = max(
                    self.peak_resident[node], len(self.pools[node].cached) * BLOCK_BYTES
                )
                self.schedule(node)
        if (
            self.jobs
            or self.arrived
            or self.preparing
            or len(self.finished) != len(self.rows)
        ):
            raise ValueError("Serving simulation did not drain")
        results = list(self.finished.values())
        return {
            "schema": "pegaflow-pd-serving-simulation/v1",
            "scope": "Production-oriented architecture scenario; component-calibrated, full P/D unvalidated",
            "settings": {
                "P": self.p_count,
                "D": self.d_count,
                "TP": 1,
                "model": DOMAIN,
                "graph_mode": True,
                "P_token_budget": self.p_tokens,
                "D_token_budget": self.d_tokens,
                "max_sequences_each": self.max_sequences,
                "GPU_APC": True,
                "copy_compute_overlap": self.overlap,
                "memory_budget": self.budget,
                "routing": "session affinity across configured producer and consumer engines",
                "arrivals": "historical nominal ledger with same-session completion dependency",
                "handoff": "full completed prefix blocks from P; D computes logits suffix",
                "transfer_model": "CPU save, P-to-D FIFO service, H2D restore; no background traffic",
            },
            "summary": {
                "requests": len(results),
                "TTFT": quantiles([r["TTFT"] for r in results]),
                "E2E": quantiles([r["E2E"] for r in results]),
                "TPOT_ms": quantiles([r["TPOT_ms"] for r in results]),
                "finish_span": max(r["end"] for r in results),
                "peak_live_requests": self.peak_live,
                "peak_running_each": self.peak_running,
                "peak_GPU_resident_GiB": [v / 2**30 for v in self.peak_resident],
                "GPU_evictions_each": [p.evictions for p in self.pools],
                "peak_CPU_GiB": [h.peak / 2**30 for h in self.host],
                "CPU_evictions_each": [h.evictions for h in self.host],
                "RDMA_GiB": sum(
                    t["bytes"]
                    for t in self.transfers
                    if t["resource"].startswith("RDMA")
                )
                / 2**30,
                "D_uncached_tokens": sum(
                    self.rows[r["id"]]["tokens"] - r["stages"]["D"]["cached_tokens"]
                    for r in results
                ),
                "P_uncached_tokens": sum(
                    self.rows[r["id"]]["tokens"] - r["stages"]["P"]["cached_tokens"]
                    for r in results
                ),
                "extrapolated_steps": self.extrapolated_steps,
            },
            "per_request": results,
            "steps": self.steps,
            "transfers": self.transfers,
            "limitations": [
                "Graph compute is calibrated on one engine, budgets <=2048 and sequences <=8",
                "P token budgets above2048 use explicitly counted cost extrapolation",
                "CPU restore/save measured outside the full model connector; production overlap requires validation",
                "Network service uses historical P-to-D fit and FIFO capacity; packet-level RDMA congestion unmodeled",
                "Host-memory headroom and NPU non-KV extrapolation require startup memory profiling",
                "Serving uses asynchronous handoff; historical whole-request application locks require implementation changes",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--step-model", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p-tokens", type=int, default=8192)
    parser.add_argument("--d-tokens", type=int, default=2048)
    parser.add_argument("--max-sequences", type=int, default=128)
    parser.add_argument("--serialize-copy-compute", action="store_true")
    parser.add_argument("--npu-gib", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = ServingSimulation(
        read(input_path(args.inputs, "simulation-inputs.json")),
        read(input_path(args.inputs, "protocol.json")),
        read(args.step_model),
        read(args.components),
        p_tokens=args.p_tokens,
        d_tokens=args.d_tokens,
        max_sequences=args.max_sequences,
        overlap=not args.serialize_copy_compute,
        npu_gib=args.npu_gib,
    ).run()
    write(args.output, result)
    print(result["summary"])


if __name__ == "__main__":
    main()
