"""P/D engine iterations with asynchronous queries and batched KV restoration."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

from pd_multinode_simulator import CalibratedPD
from pd_serving_simulator import BLOCK_BYTES, seed_gpu
from step_simulator import Job, vector


def graph_vector(row):
    if row["prefill_tokens"] == 0:
        return [
            1,
            row["padded_sequences"] / 8,
            row["decode_context"] / (8 * 8192),
            row["max_context"] / 8192,
        ]
    return [
        1,
        (row["prefill_tokens"] + row["decode_tokens"]) / 2048,
        row["prefill_attention"] / (2048 * 8192),
        row["decode_context"] / (8 * 8192),
        row["sequences"] / 8,
    ]


class GraphCost:
    def __init__(self, data):
        self.data = data

    def duration(self, row):
        kind = "decode" if row["prefill_tokens"] == 0 else "prefill"
        model = self.data["models"][kind]
        inputs = (
            graph_vector(row) if model.get("kind") == "graph_shape" else vector(row)
        )
        return max(
            1e-6, sum(a * b for a, b in zip(model["coefficients"], inputs, strict=True))
        )


class PipelineSimulation(CalibratedPD):
    def __init__(self, workload, profile, config=None):
        super().__init__(workload, profile, config)
        if profile.get("graph_models"):
            self.role_models = {
                role: GraphCost(model)
                for role, model in profile["graph_models"].items()
            }
        self.incoming = defaultdict(deque)
        self.fetch_ready = defaultdict(deque)
        self.copy_queue = defaultdict(deque)
        self.loaded = defaultdict(deque)
        self.flow = {}
        self.compute_active = defaultdict(bool)
        self.cycle_event = {}
        self.last_cycle = {}
        self.pipeline_ready = set()
        self.copy_sequence = 0
        self.copy_batches = []
        self.idle_poll = profile.get("idle_poll_s", 0.001)
        self.restore_model = profile["restore_model"]

    def api_delay(self, role, tokens):
        a, b = self.profile.get("api_inbound", {}).get(role, [0, 0])
        return max(0, a + b * tokens / 8192)

    def prepare(self, key, node):
        role = "P" if node < self.p_count else "D"
        marker = key, node
        if marker not in self.pipeline_ready:
            self.pipeline_ready.add(marker)
            delay = self.api_delay(role, self.rows[key]["tokens"])
            if role == "D":
                delay += self.profile.get("client_handoff_gap_s", 0)
            self.event(self.now + delay, "api_delivered", key, node)
            return
        if role == "P":
            return super().prepare(key, node)
        live = self.jobs[key]
        live["engine_jobs"][node] = Job(self.request(key, node))
        live["stages"]["D"] = {"prepare": self.now, "node": node}
        self.preparing.add((node, key))
        self.incoming[node].append(key)
        self.ensure_cycle(
            node,
            immediate=not (
                self.flow.get(node) or self.copy_queue[node] or self.fetch_ready[node]
            ),
        )

    def ensure_cycle(self, node, *, immediate=False):
        if node in self.engine_busy:
            return
        when = self.now if immediate else self.now + self.idle_poll
        if self.cycle_event.get(node, float("inf")) <= when:
            return
        self.cycle_event[node] = when
        self.event(when, "D_cycle", str(node), (node, when))

    def restore(self, key, node, gpu, target):
        # RPC prefetch completion becomes visible at a scheduler iteration.
        self.fetch_ready[node].append((key, gpu, target))
        if target > gpu:
            self.host[node].pins.update(self.rows[key]["hashes"][gpu:target])
        self.ensure_cycle(node)

    def cycle(self, node):
        if self.last_cycle.get(node) == self.now:
            return
        self.last_cycle[node] = self.now
        while self.loaded[node]:
            batch = self.loaded[node].popleft()
            for key, gpu, target in batch["items"]:
                seed_gpu(self.pools[node], self.rows[key]["hashes"][gpu:target])
                self.jobs[key]["stages"]["D"]["restore_end"] = batch["end"]
                self.admit(key, node)
        limit = self.config.get(
            "D_max_sequences", self.config.get("max_sequences", 128)
        )
        # vLLM skips asynchronous waits while a free running slot exists.
        if len(self.running[node]) + len(self.waiting[node]) < limit:
            remaining = deque()
            while self.incoming[node]:
                key = self.incoming[node].popleft()
                if len(self.running[node]) + len(self.waiting[node]) >= limit:
                    remaining.append(key)
                    continue
                live = self.jobs[key]
                row = self.rows[key]
                job = live["engine_jobs"][node]
                gpu = len(self.pools[node].prefix(job))
                target = len(row["hashes"])
                source = live["P"]
                if self.config.get("handoff_policy", "full") == "best_effort":
                    target = gpu
                    while target < len(row["hashes"]) and (
                        row["hashes"][target] in self.host[node].keys
                        or row["hashes"][target] in self.host[source].keys
                    ):
                        target += 1
                job.external_prefix = target
                live["stages"]["D"].update(
                    NPU_initial_prefix_blocks=gpu,
                    target_prefix_blocks=target,
                    restore_bytes=(target - gpu) * BLOCK_BYTES,
                    query_start=self.now,
                )
                if target == gpu:
                    self.admit(key, node)
                    continue
                missing = [
                    h
                    for h in row["hashes"][gpu:target]
                    if h not in self.host[node].keys
                ]
                if any(h not in self.host[source].keys for h in missing):
                    raise ValueError("P cache cannot supply the required prefix")
                if missing:
                    self.host[source].pins.update(missing)
                    a, b = self.profile.get("discovery_model", {}).get(
                        "coefficients", [0, 0]
                    )
                    query = (
                        self.profile.get("query_fixed_s", 0) + a + b * len(missing) / 64
                    )
                    self.event(
                        self.now + query,
                        "pipeline_fetch",
                        key,
                        (node, gpu, target, source, missing),
                    )
                else:
                    self.restore(key, node, gpu, target)
            self.incoming[node] = remaining
            if self.fetch_ready[node]:
                items = list(self.fetch_ready[node])
                self.fetch_ready[node].clear()
                self.copy_sequence += 1
                batch = {
                    "batch": self.copy_sequence,
                    "node": node,
                    "items": items,
                    "queued": self.now,
                    "blocks": sum(target - gpu for _, gpu, target in items),
                    "request_count": len(items),
                }
                for key, _, _ in items:
                    self.jobs[key]["stages"]["D"].update(
                        restore_batch=batch["batch"], restore_queued=self.now
                    )
                self.copy_queue[node].append(batch)
        if node not in self.flow and self.copy_queue[node]:
            self.start_copy(node)

    def start_copy(self, node):
        if node in self.flow or not self.copy_queue[node]:
            return
        batch = self.copy_queue[node].popleft()
        coefficients = self.restore_model["coefficients"]
        if self.restore_model.get("kind") == "capacity":
            rate = self.config.get("host_copy_GBps", {}).get(
                self.hosts[node], self.restore_model["default_copy_GBps"]
            )
            size = batch["blocks"] * BLOCK_BYTES / (rate * 1e9)
        else:
            size = batch["blocks"] / 64
        vector = [1, size, max(0, batch["request_count"] - 1) / 4]
        work = max(1e-6, sum(a * b for a, b in zip(coefficients, vector, strict=True)))
        batch.update(
            start=self.now, work_s=work, remaining=work, last=self.now, generation=0
        )
        self.flow[node] = batch
        self.copy_busy[node] += 1
        for key, _, _ in batch["items"]:
            self.jobs[key]["stages"]["D"]["restore_start"] = self.now
        self.advance_copy(node)

    def advance_copy(self, node):
        flow = self.flow.get(node)
        if flow is None:
            return
        old_rate = flow.get("rate", 1)
        flow["remaining"] = max(
            0, flow["remaining"] - (self.now - flow["last"]) * old_rate
        )
        flow["last"] = self.now
        slowdown = self.restore_model.get("busy_slowdown", 1)
        flow["rate"] = 1 / slowdown if self.compute_active[node] else 1
        flow["generation"] += 1
        self.event(
            self.now + flow["remaining"] / flow["rate"],
            "copy_complete",
            str(node),
            (node, flow["batch"], flow["generation"]),
        )

    def set_compute(self, node, active):
        if self.compute_active[node] == active:
            return
        self.compute_active[node] = active
        self.advance_copy(node)

    def schedule(self, node):
        if node < self.p_count:
            return super().schedule(node)
        if node in self.engine_busy:
            return
        if self.running[node] or self.waiting[node]:
            self.cycle(node)
        before = len(self.steps)
        super().schedule(node)
        if len(self.steps) > before:
            self.set_compute(node, True)
        elif self.incoming[node] or self.fetch_ready[node] or self.loaded[node]:
            self.ensure_cycle(node)

    def step_done(self, node, scheduled):
        super().step_done(node, scheduled)
        if node >= self.p_count and not self.running[node]:
            self.set_compute(node, False)

    def extra_event(self, kind, key, data):
        if kind == "pipeline_fetch":
            payload = len(data[4]) * BLOCK_BYTES
            self.resource(
                "RDMA:P-to-D",
                payload * self.network_s_per_byte,
                "remote_done",
                key,
                data,
                payload=payload,
            )
            return True
        if kind == "api_delivered":
            self.prepare(key, data)
            return True
        if kind == "D_cycle":
            node, when = data
            if self.cycle_event.get(node) != when:
                return True
            del self.cycle_event[node]
            if node in self.engine_busy:
                return True
            self.cycle(node)
            return True
        if kind == "copy_complete":
            node, batch_id, generation = data
            flow = self.flow.get(node)
            if (
                flow is None
                or flow["batch"] != batch_id
                or flow["generation"] != generation
            ):
                return True
            self.flow.pop(node)
            flow["end"] = self.now
            self.copy_busy[node] -= 1
            for request, gpu, target in flow["items"]:
                self.host[node].pins.subtract(self.rows[request]["hashes"][gpu:target])
            self.copy_batches.append(
                {
                    k: v
                    for k, v in flow.items()
                    if k not in ["remaining", "last", "generation", "rate"]
                }
            )
            self.transfers.append(
                {
                    "id": f"copy-batch-{batch_id}",
                    "request_ids": [r[0] for r in flow["items"]],
                    "resource": f"H2D:{node}",
                    "queued": flow["queued"],
                    "start": flow["start"],
                    "end": self.now,
                    "bytes": flow["blocks"] * BLOCK_BYTES,
                }
            )
            self.loaded[node].append(flow)
            self.ensure_cycle(node)
            if self.copy_queue[node]:
                self.start_copy(node)
            return True
        return False

    def run(self):
        result = super().run()
        result["schema"] = "pegaflow-pd-pipeline-simulation/v2"
        result["copy_batches"] = self.copy_batches
        result["settings"]["transfer_model"] = (
            "engine-iteration discovery, asynchronous prefetch, native copy cohorts and completion feedback"
        )
        result["settings"]["idle_poll_s"] = self.idle_poll
        result["settings"]["restore_busy_slowdown"] = self.restore_model.get(
            "busy_slowdown", 1
        )
        outside = sum(
            any(
                batch[key] < low or batch[key] > high
                for key, (low, high) in self.profile.get("restore_coverage", {}).items()
            )
            for batch in self.copy_batches
        )
        result["summary"]["extrapolated_restore_batches"] = outside
        if outside:
            result["coverage"]["extrapolations"].append(
                "Copy cohort exceeds calibrated size or request-count range"
            )
            result["coverage"]["classification"] = "extrapolated"
        if self.restore_model.get("kind") == "capacity" and any(
            self.hosts[n] not in self.config.get("host_copy_GBps", {})
            for n in range(self.p_count, self.node_count)
        ):
            result["coverage"]["extrapolations"].append(
                "Destination copy capability uses the profile median; supply host_copy_GBps for calibrated hardware"
            )
            result["coverage"]["classification"] = "extrapolated"
        result["scope"] = (
            "Iteration-aware, condition-calibrated P/D simulation with asynchronous cache handoff"
        )
        result["limitations"][2] = (
            "Copy/compute coupling is selected using calibration role swaps; host-memory and PCIe interference remain approximate"
        )
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = PipelineSimulation(
        json.loads(args.workload.read_text()),
        json.loads(args.profile.read_text()),
        json.loads(args.config.read_text()) if args.config else None,
    ).run()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
