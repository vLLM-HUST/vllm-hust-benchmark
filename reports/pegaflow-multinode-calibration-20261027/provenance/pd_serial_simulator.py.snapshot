"""Re-decide the historical serial P/D baseline from calibration and inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import itertools
import json
import math
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path


_RUNTIME_CACHE = None


def load_runtime():
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None:
        return _RUNTIME_CACHE
    root = Path(__file__).resolve().parents[2]
    runtime = root / "experiments/issue23_strategies/runtime"
    paths = list(sys.path)
    names = {p.stem for p in (runtime / "code").glob("*.py")}
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(root / "python"))
        import pegaflow

        overlay = str(runtime / "overlay/pegaflow")
        if overlay not in pegaflow.__path__:
            pegaflow.__path__.insert(0, overlay)
        sys.path.insert(0, str(runtime / "code"))
        import run as core
        from confirmation_state import state_digest
        from pegaflow.joint_policy import (
            Calendar,
            Costs,
            Demand,
            Layout,
            Unit,
            View,
            copy_package,
        )
        from pegaflow.joint_queue import QueueLedger

        _RUNTIME_CACHE = {
            "core": core,
            "state_digest": state_digest,
            "Calendar": Calendar,
            "Costs": Costs,
            "Demand": Demand,
            "Layout": Layout,
            "Unit": Unit,
            "View": View,
            "copy_package": copy_package,
            "QueueLedger": QueueLedger,
        }
    finally:
        sys.path[:] = paths
        for name, module in previous.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module
    return _RUNTIME_CACHE


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith(".gz") else raw)


def input_path(root, name):
    path = root / name
    return path if path.exists() else root / (name + ".gz")


def write(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(
        gzip.compress(raw, mtime=0) if str(path).endswith(".gz") else raw
    )


def quantiles(values):
    values = sorted(values)
    return {
        "mean": statistics.mean(values),
        **{f"p{p}": values[math.ceil(p * len(values) / 100) - 1] for p in (50, 95, 99)},
        "max": values[-1],
    }


class OriginalWorld:
    """The pre-test frozen calibration model, including cold warmups."""

    def __init__(self, costs):
        self.costs = costs

    def compute(self, node, tokens, prefix):
        return self.costs.compute(node, tokens, prefix)

    def restore(self, node, prefix):
        return (
            self.costs.service[node].seconds_per_rdma_byte
            * prefix
            * self.costs.bytes_per_token
        )

    def decode_tail(self, node, tokens):
        return self.costs.decode_seconds[node - 2]

    def copy_seconds(self, source, destination, payload):
        if payload == 0 or source == destination:
            return 0.0
        slope = (
            self.costs.seconds_per_pd_byte
            if source < 2 <= destination
            else (
                self.costs.seconds_per_dp_byte
                if destination < 2 <= source
                else self.costs.seconds_per_copy_byte
            )
        )
        return self.costs.transfer_fixed + slope * payload


class Simulation:
    def __init__(self, runtime, bundle, protocol, seed, physics="original"):
        self.rt, self.protocol = runtime, protocol
        self.rows = {row["id"]: row for row in bundle["rows"]}
        for row in bundle["rows"]:
            if set(row) != {"id", "task", "tokens", "hashes"}:
                raise ValueError("Prediction input contains unexpected fields")
        self.order = protocol["test_order"]
        if list(self.rows) != self.order:
            raise ValueError("Frozen request order differs")
        values = dict(bundle["costs"])
        values["service"] = tuple(
            runtime["core"].ObservedLatencyModel(**r) for r in values["service"]
        )
        values["decode_seconds"] = tuple(values["decode_seconds"])
        self.costs = runtime["Costs"](**values)
        if physics not in ("original", "seed-refit"):
            raise ValueError("Unknown physics model")
        self.physics = physics
        self.world = (
            OriginalWorld(self.costs)
            if physics == "original"
            else (
                runtime["core"].seed_operations(self.costs, seed, "parent").freeze(0.0)
            )
        )
        self.layout = runtime["Layout"].current(
            tuple(
                runtime["View"].parse(n, v)
                for n, v in enumerate(bundle["initial_inventory"])
            )
        )
        if self.layout.capacities != (protocol["pool_bytes_per_instance"],) * 4:
            raise ValueError("Host cache capacity differs")
        self.ledger = runtime["QueueLedger"]()
        self.tasks = sorted({row["task"] for row in self.rows.values()})
        self.times = dict(
            zip(
                self.order,
                protocol["arrival_condition"]["offered_arrivals_seconds"],
                strict=True,
            )
        )
        controls = bundle["control_cost_rows"]
        self.route_cpu = statistics.mean(r["route_cpu_seconds"] for r in controls)
        self.boundary_cpu = statistics.mean(
            r["D_boundary_CPU_seconds"] for r in controls
        )
        self.route_rpc = statistics.mean(
            max(0.0, r["route_seconds"] - r["route_cpu_seconds"]) for r in controls
        )
        self.boundary_rpc = statistics.mean(
            max(
                0.0,
                r["D_boundary_ready_seconds"]
                - r["D_boundary_start_seconds"]
                - r["D_boundary_CPU_seconds"],
            )
            for r in controls
        )
        self.now, self.serial = 0.0, itertools.count()
        self.events, self.pending = [], []
        self.queues = [deque() for _ in range(4)]
        self.active, self.gates, self.links = {}, {}, {}
        self.jobs, self.finished, self.sessions = {}, {}, {}
        self.busy_sessions, self.router_busy = set(), False
        self.wait_ops, self.actions, self.decisions, self.apis = [], [], [], []
        self.peak = list(self.layout.usage_values)
        for key in self.order:
            self.event(self.times[key], "arrival", key)

    def event(self, at, kind, key, data=None):
        if at < self.now - 1e-9:
            raise ValueError("Noncausal event")
        heapq.heappush(self.events, (at, next(self.serial), kind, key, data))

    def demand(self, key):
        row = self.rows[key]
        return self.rt["Demand"](
            key,
            row["task"],
            row["tokens"],
            tuple(bytes.fromhex(h) for h in row["hashes"]),
        )

    def route(self):
        if self.router_busy:
            return
        key = next(
            (k for k in self.pending if self.rows[k]["task"] not in self.busy_sessions),
            None,
        )
        if key is None:
            return
        self.pending.remove(key)
        item = replace(self.demand(key), arrival=self.now)
        lane = self.tasks.index(item.task) % 2
        began = time.perf_counter()
        route, tried, _ = self.rt["core"].best_route(
            item,
            self.layout,
            self.costs,
            self.rt["Calendar"](),
            fixed_pair=(lane, lane + 2),
            queues=self.ledger,
            max_candidates=64,
        )
        cpu = time.perf_counter() - began
        ready = max(self.times[key], self.sessions.get(item.task, 0.0))
        self.jobs[key] = {
            "key": key,
            "route": route,
            "ready": ready,
            "offered": self.times[key],
            "first": None,
            "phase": "routing",
            "stages": {},
        }
        self.ledger.reserve(key, route, self.now)
        self.busy_sessions.add(item.task)
        self.router_busy = True
        self.decisions.append(
            {
                "id": key,
                "at": self.now,
                "P": route.p,
                "D": route.d,
                "P_blocks": route.p_blocks,
                "initial_D_blocks": route.d_blocks,
                "candidates": tried,
                "CPU_wall_s": cpu,
            }
        )
        self.event(self.now + self.route_cpu + self.route_rpc, "routed", key)

    def stage(self, key, node):
        job, item = self.jobs[key], self.demand(key)
        self.active[node] = key
        self.ledger.observe(key, node, "started", self.now)
        route = job["route"]
        stage = {"start": self.now, "node": node}
        job["stages"]["P" if node < 2 else "D"] = stage
        if node < 2:
            source, count, delay = route.source, route.p_blocks, 0.0
        else:
            began = time.perf_counter()
            count, _ = self.rt["core"].decode_boundary(
                item, self.layout, self.costs, route.p, route.d
            )
            measured_boundary_cpu = time.perf_counter() - began
            delay = self.boundary_rpc + self.boundary_cpu
            source = route.p
            job["boundary"] = {
                "at": self.now,
                "P_available": self.layout.prefix(route.p, item.hashes),
                "D_available": self.layout.prefix(node, item.hashes),
                "selected": count,
                "D_free_bytes": self.layout.capacities[node] - self.layout.usage(node),
                "simulator_CPU_wall_s": measured_boundary_cpu,
            }
            job["boundary"]["candidates"] = self.boundary_candidates(
                item, route.p, route.d
            )
        job["node"], job["phase"] = node, "check_wait"
        self.event(self.now + delay, "check_ready", key, (source, node, count))

    def boundary_candidates(self, item, source, node):
        local = self.layout.prefix(node, item.hashes)
        limit = max(local, self.layout.prefix(source, item.hashes))
        result = []
        for length in range(limit + 1):
            payload = 0
            if length > local:
                try:
                    payload = (
                        self.rt["copy_package"](
                            self.layout, source, node, item.hashes[:length]
                        )
                        .operations[-1]
                        .payload_bytes
                    )
                except ValueError as error:
                    result.append(
                        {"blocks": length, "feasible": False, "reason": str(error)}
                    )
                    continue
            seconds = self.costs.compute(node, item.tokens, length * 128)
            if payload:
                seconds += (
                    self.costs.transfer_fixed + payload * self.costs.seconds_per_pd_byte
                )
            result.append(
                {
                    "blocks": length,
                    "feasible": True,
                    "predicted_first_service_s": seconds,
                    "payload_bytes": payload,
                }
            )
        return result

    def check(self, key, source, node, count):
        job, item = self.jobs[key], self.demand(key)
        local = self.layout.prefix(node, item.hashes)
        payload, copy = 0, None
        before = count
        if count > local and source != node:
            try:
                copy = self.rt["copy_package"](
                    self.layout, source, node, item.hashes[:count]
                ).operations[-1]
                payload = copy.payload_bytes
            except ValueError as error:
                count = min(local, count)
                self.actions.append(
                    {
                        "kind": "handoff_fallback",
                        "id": key,
                        "at": self.now,
                        "requested": before,
                        "accepted": count,
                        "reason": str(error),
                    }
                )
        if source == node:
            count = min(count, local)
        job["count"] = count
        direction = (
            "pd"
            if source < 2 <= node
            else "dp"
            if node < 2 <= source
            else "P"
            if node < 2
            else "D"
        )
        if payload and direction in self.links:
            job["phase"] = "link_wait"
            self.wait_ops.append(
                ("link", key, (source, node, count, payload, direction))
            )
            return
        if payload:
            self.links[direction] = key
        # The inventory query executes under both software gates, including resident handoffs.
        duration = self.route_rpc if source != node and before > 0 else 0.0
        if payload:
            duration += self.world.copy_seconds(source, node, payload)
        job["phase"] = "copy"
        self.event(
            self.now + duration,
            "copied",
            key,
            (source, node, count, payload, direction),
        )

    def copied(self, key, data):
        source, node, count, payload, direction = data
        job, item = self.jobs[key], self.demand(key)
        if payload:
            self.layout = self.rt["copy_package"](
                self.layout, source, node, item.hashes[:count]
            ).layout
            nodes = list(self.layout.units)
            nodes[node] = tuple(replace(u, virtual=False) for u in nodes[node])
            self.layout = replace(self.layout, units=tuple(nodes))
            del self.links[direction]
        self.actions.append(
            {
                "kind": "copy" if payload else "resident",
                "id": key,
                "at": self.now,
                "source": source,
                "destination": node,
                "blocks": count,
                "payload": payload,
            }
        )
        for owner in job.pop("held_gates", ()):
            del self.gates[owner]
        job["phase"] = "api_wait"
        self.wait_ops.append(("api", key, node))

    def publish(self, node, key, position, logical):
        if logical in self.layout.keys(node):
            return
        amount = 128 * self.costs.bytes_per_token
        if self.layout.usage(node) + amount > self.layout.capacities[node]:
            prefixes = {
                k: self.layout.prefix(node, self.demand(k).hashes) for k in self.jobs
            }
            victims, blocks = [], 0
            for unit in self.layout.units[node]:
                if unit.protected:
                    continue
                victims.append(unit.identity)
                blocks += len(unit.keys)
                if blocks >= self.protocol["source_derived_reclaim_batch"]:
                    break
            self.layout = self.layout.remove(node, victims)
            self.actions.append(
                {
                    "kind": "native_reclaim",
                    "at": self.now,
                    "id": key,
                    "node": node,
                    "blocks": blocks,
                    "victims": len(victims),
                    "pending_prefix_drops": {
                        k: [before, self.layout.prefix(node, self.demand(k).hashes)]
                        for k, before in prefixes.items()
                        if self.layout.prefix(node, self.demand(k).hashes) < before
                    },
                }
            )
        if self.layout.usage(node) + amount > self.layout.capacities[node]:
            raise ValueError("Native publication cannot fit")
        unit = self.rt["Unit"](
            f"sim:{node}:{key}:{position}",
            frozenset((logical,)),
            amount,
            ((logical, amount),),
        )
        nodes = list(self.layout.units)
        nodes[node] = (*nodes[node], unit)
        self.layout = replace(self.layout, units=tuple(nodes))

    def api(self, key, node):
        job, item = self.jobs[key], self.demand(key)
        count = min(job["count"], self.layout.prefix(node, item.hashes))
        prefix = count * 128
        first = self.world.compute(node, item.tokens, prefix)
        tail = self.world.decode_tail(node, item.tokens) if node >= 2 else 0.0
        restore = min(first, self.world.restore(node, prefix)) if prefix else 0.0
        if (
            item.tokens + (128 if node >= 2 else 1)
        ) * self.costs.bytes_per_token > self.protocol["npu_kv_bytes"]:
            raise ValueError("Serial active NPU KV exceeds original budget")
        self.gates[node] = key
        job["phase"] = "api"
        job["stages"]["P" if node < 2 else "D"].update(
            api_start=self.now,
            cached_tokens=prefix,
            first_service=first,
            tail_service=tail,
        )
        if node >= 2:
            self.event(self.now + first, "first", key)
        # Approximate incremental native publication at completed chunk boundaries.
        full = (item.tokens + (127 if node >= 2 else 0)) // 128
        for pos in range(count, full):
            computed = min(item.tokens, (pos + 1) * 128)
            if (pos + 1) * 128 <= item.tokens:
                at = (
                    self.now
                    + restore
                    + (first - restore) * (computed - prefix) / (item.tokens - prefix)
                )
            else:
                at = self.now + first + tail * ((pos + 1) * 128 - item.tokens) / 127
            logical = (
                item.hashes[pos]
                if pos < len(item.hashes)
                else hashlib.sha256(
                    f"private-generated:{node}:{key}:{pos}".encode()
                ).digest()
            )
            self.event(at, "publish", key, (node, pos, logical))
        self.apis.append(
            {
                "id": key,
                "node": node,
                "at": self.now,
                "tokens": item.tokens,
                "cached_tokens": prefix,
                "first_service": first,
                "full_service": first + tail,
            }
        )
        self.event(self.now + first + tail, "api_done", key, node)

    def pump(self):
        for node in range(4):
            if node not in self.active and self.queues[node]:
                self.stage(self.queues[node].popleft(), node)
        remaining = []
        for kind, key, data in self.wait_ops:
            job = self.jobs[key]
            if kind == "check":
                source, node, count = data
                gates = sorted({source, node}) if count > 0 and source != node else []
                if any(n in self.gates for n in gates):
                    remaining.append((kind, key, data))
                    continue
                for n in gates:
                    self.gates[n] = key
                job["held_gates"] = gates
                self.check(key, *data)
            elif kind == "link":
                if data[-1] in self.links:
                    remaining.append((kind, key, data))
                    continue
                self.links[data[-1]] = key
                job["phase"] = "copy"
                seconds = self.route_rpc + self.world.copy_seconds(
                    data[0], data[1], data[3]
                )
                self.event(self.now + seconds, "copied", key, data)
            else:
                if data in self.gates:
                    remaining.append((kind, key, data))
                    continue
                self.api(key, data)
        self.wait_ops = remaining

    def complete(self, key, node):
        job = self.jobs[key]
        del self.gates[node]
        del self.active[node]
        self.ledger.observe(key, node, "finished", self.now)
        stage = job["stages"]["P" if node < 2 else "D"]
        stage["end"] = self.now
        stage["published_prefix_at_end"] = self.layout.prefix(
            node, self.demand(key).hashes
        )
        if node < 2:
            self.queues[job["route"].d].append(key)
            self.ledger.observe(key, job["route"].d, "enqueued", self.now)
        else:
            self.finished[key] = {
                "id": key,
                "P": job["route"].p,
                "D": job["route"].d,
                "ready": job["ready"],
                "offered": job["offered"],
                "first": job["first"],
                "end": self.now,
                "TTFT": job["first"] - job["ready"],
                "E2E": self.now - job["ready"],
                "stages": job["stages"],
                "boundary": job["boundary"],
            }
            task = self.rows[key]["task"]
            self.sessions[task] = self.now
            self.busy_sessions.remove(task)
            del self.jobs[key]

    def extra_event(self, kind, key, data):
        return False

    def run(self):
        while self.events:
            at, _, kind, key, data = heapq.heappop(self.events)
            self.now = max(at, self.now)
            if self.extra_event(kind, key, data):
                pass
            elif kind == "arrival":
                self.pending.append(key)
            elif kind == "routed":
                self.router_busy = False
                self.queues[self.jobs[key]["route"].p].append(key)
            elif kind == "check_ready":
                self.wait_ops.append(("check", key, data))
            elif kind == "copied":
                self.copied(key, data)
            elif kind == "publish":
                self.publish(data[0], key, data[1], data[2])
            elif kind == "first":
                self.jobs[key]["first"] = self.now
                self.ledger.observe(key, self.jobs[key]["route"].d, "first", self.now)
            elif kind == "api_done":
                self.complete(key, data)
            else:
                raise ValueError(f"Unknown event: {kind}")
            self.route()
            self.pump()
            self.peak = [
                max(a, b)
                for a, b in zip(self.peak, self.layout.usage_values, strict=True)
            ]
        if (
            self.jobs
            or self.pending
            or self.wait_ops
            or self.gates
            or len(self.finished) != len(self.rows)
        ):
            raise ValueError("Serial simulation did not drain")
        return {
            "scope": "Original task-affinity serial baseline; endogenous cache and restore decisions",
            "physics": self.physics,
            "TTFT": quantiles([r["TTFT"] for r in self.finished.values()]),
            "E2E": quantiles([r["E2E"] for r in self.finished.values()]),
            "finish_span": self.now,
            "cache_hits": {
                str(n): sum(r["cached_tokens"] for r in self.apis if r["node"] == n)
                for n in range(4)
            },
            "route_counts": dict(Counter(str(r["D"]) for r in self.finished.values())),
            "peak_host_bytes": self.peak,
            "per_request": list(self.finished.values()),
            "decisions": self.decisions,
            "apis": self.apis,
            "cache_events": self.actions,
            "control_cost": {
                "route_RPC_mean": self.route_rpc,
                "boundary_RPC_mean": self.boundary_rpc,
                "route_CPU_mean": self.route_cpu,
                "boundary_CPU_mean": self.boundary_cpu,
                "clock": "Frozen aggregate of prior 380 control records; local simulator CPU is telemetry only",
            },
            "native_publication": "Uniform completed-prefill-chunk timing; physical owner grouping approximated",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--physics", choices=("original", "seed-refit"), default="original"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rt = load_runtime()
    bundle, protocol = (
        read(input_path(args.inputs, "simulation-inputs.json")),
        read(input_path(args.inputs, "protocol.json")),
    )
    expected = read(input_path(args.inputs, "B-before-initial-state-contract.json"))[
        "digest"
    ]
    if rt["state_digest"](bundle["initial_inventory"]) != expected:
        raise ValueError("Initial native cache state differs from real baseline")
    result = Simulation(
        rt, bundle, protocol, args.inputs / "calibration-seed", args.physics
    ).run()
    result["initial_cache_state_exact"] = True
    result["service_model_source"] = (
        "Original pre-test frozen calibration model including cold warmups; no target service labels"
        if args.physics == "original"
        else "Independent32 refit without cold warmup; no target service labels"
    )
    write(args.output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["TTFT", "finish_span", "cache_hits", "route_counts"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
