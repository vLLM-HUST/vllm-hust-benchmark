"""Run fixed-commit P/D execution models under one diagnostic contract.

Each engine runs in a separate process because their frozen overlays share
Python module names. Adapters change experiment inputs and routing/cost oracles;
cache publication, leases, eviction, copy gates and batching remain engine code.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith(".gz") else raw)


def write(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def bundle_input(root):
    path = Path(root) / "simulation-inputs.json"
    return read(path if path.exists() else path.with_suffix(".json.gz"))


def quantiles(values):
    values = sorted(values)
    return {
        "mean": statistics.mean(values),
        **{f"p{p}": values[math.ceil(p * len(values) / 100) - 1] for p in (50, 95, 99)},
        "max": values[-1],
    }


class CommonWorld:
    """The same frozen singleton totals and copy slopes for both engines.

    Restore is a component of first-token service, never added to it. No
    separately identified synchronous-save term is available in this oracle.
    """

    fits = ((0, 0, 0, 0, 0),) * 4

    def __init__(self, costs):
        self.costs = costs

    def __getattr__(self, name):
        return getattr(self.costs, name)

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
            else self.costs.seconds_per_dp_byte
            if destination < 2 <= source
            else self.costs.seconds_per_copy_byte
        )
        return self.costs.transfer_fixed + slope * payload


def worker(args):
    contract = read(args.contract)
    case = next(c for c in contract["cases"] if c["name"] == args.case)
    bundle = copy.deepcopy(bundle_input(args.inputs))
    protocol = read(Path(args.inputs) / "protocol.json")
    if case.get("cold"):
        for row in bundle["rows"]:
            row["hashes"] = []
    tasks = sorted({r["task"] for r in bundle["rows"]})
    seed = Path(args.inputs) / "calibration-seed"
    started = time.perf_counter()

    if args.worker == "ours":
        sys.path.insert(0, str(args.ours / "scripts/issue23"))
        from pd_capacity_batch_simulator import BatchedSimulation
        from pd_serial_simulator import load_runtime

        rt = load_runtime()
        from pegaflow.joint_policy import Route

        def fixed_route(item, layout, costs, calendar, *, fixed_pair, **kwargs):
            p, d = fixed_pair
            source = min(
                range(4),
                key=lambda n: (-layout.prefix(n, item.hashes), n != p, n),
            )
            blocks = layout.prefix(source, item.hashes)
            first = costs.compute(p, item.tokens, blocks * 128)
            df = costs.compute(d, item.tokens, 0)
            return (
                Route(
                    p,
                    d,
                    source,
                    blocks,
                    0,
                    first + df,
                    first,
                    first + df,
                    0,
                    calendar,
                    first,
                    df,
                    df + costs.decode_seconds[d - 2],
                ),
                1,
                None,
            )

        rt["core"].best_route = fixed_route
        rt["core"].decode_boundary = lambda item, layout, costs, p, d: (
            max(layout.prefix(p, item.hashes), layout.prefix(d, item.hashes)),
            1,
        )

        class ComparedOurs(BatchedSimulation):
            def pump(self):
                if case.get("resident_local"):
                    updated = []
                    for kind, key, data in self.wait_ops:
                        if kind == "check":
                            source, node, count = data
                            if (
                                self.layout.prefix(node, self.demand(key).hashes)
                                >= count
                            ):
                                data = (node, node, count)
                        updated.append((kind, key, data))
                    self.wait_ops = updated
                super().pump()

            def stage(self, key, node):
                visible = [
                    self.layout.prefix(n, self.demand(key).hashes) for n in range(4)
                ]
                super().stage(key, node)
                selected = (
                    self.jobs[key]["route"].p_blocks
                    if node < 2
                    else self.jobs[key]["boundary"]["selected"]
                )
                self.comparison_stages.append(
                    {
                        "id": key,
                        "node": node,
                        "at": self.now,
                        "visible_blocks": visible,
                        "selected_blocks": selected,
                    }
                )

        identity_ratio = [{"context_tokens": 1, "normalized": {"1": 1, "4": 1}}]
        sim = ComparedOurs(
            rt,
            bundle,
            protocol,
            seed,
            identity_ratio,
            cap=case["D_cap"],
            host_gib=case["host_gib"],
            npu_gib=16,
            unbatchable=1 - case.get("decode_sharing", 0),
        )
        sim.comparison_stages = []
        sim.route_cpu = sim.route_rpc = sim.boundary_cpu = sim.boundary_rpc = 0.0
        if case.get("cold"):
            sim.layout = replace(sim.layout, units=((),) * 4)
        if case.get("graph"):
            from pd_graph_comparison import GraphWorld, configure_ours

            configure_ours(
                sim,
                GraphWorld(
                    sim.costs, read(args.graph_model), read(args.graph_components)
                ),
            )
        result = sim.run()
    else:
        peer = Path(args.peer)
        sys.path.insert(0, str(peer / "python"))
        sys.path.insert(0, str(peer / "experiments/issue23_strategies"))
        import simulate_continuous_batch as module
        from pegaflow.strategies.resource_schedule import Work

        module.parent.offsets = lambda *a: protocol["arrival_condition"][
            "offered_arrivals_seconds"
        ]

        class FixedAffinity(module.BatchSimulation):
            def stage(self, key, node):
                item = self.item(self.jobs[key]["row"])
                visible = [self.layout.prefix(n, item.hashes) for n in range(4)]
                super().stage(key, node)
                self.comparison_stages.append(
                    {
                        "id": key,
                        "node": node,
                        "at": self.now,
                        "visible_blocks": visible,
                        "selected_blocks": self.jobs[key]["count"],
                    }
                )

            def commit_copy(self, transition, key):
                super().commit_copy(transition, key)
                if case.get("commit_physical"):
                    self.layout = replace(
                        self.layout,
                        units=tuple(
                            tuple(replace(u, virtual=False) for u in units)
                            for units in self.layout.units
                        ),
                    )

            def choose(self, row, ready):
                lane = tasks.index(row["task"]) % 2
                p, d = lane, lane + 2
                pf = self.world.compute(p, row["tokens"], 0)
                df = self.world.compute(d, row["tokens"], 0)
                plan = module.parent.Candidate(
                    p,
                    d,
                    p,
                    0,
                    0,
                    pf,
                    pf,
                    df,
                    df + self.world.decode_tail(d, row["tokens"]),
                )
                self.planning.append({"id": row["id"], "CPU_wall": 0.0})
                return (
                    plan,
                    Work(row["id"], p, d, self.sequence, ready, (), ()),
                    0.0,
                    None,
                )

            def api_phase_split(self, key, node, prefix, first, tail):
                if case.get("amortized_restore"):
                    return 0.0, 0.0
                return super().api_phase_split(key, node, prefix, first, tail)

        values = dict(bundle["costs"])
        values["service"] = tuple(
            module.parent.core.ObservedLatencyModel(**r) for r in values["service"]
        )
        values["decode_seconds"] = tuple(values["decode_seconds"])
        costs = module.Costs(**values)
        sim = FixedAffinity(
            bundle["rows"],
            bundle["initial_inventory"],
            costs,
            seed,
            bundle["control_cost_rows"],
            0.49032341538142143,
            "native",
            False,
            batch_config=module.BatchConfig(
                max_sequences=case["D_cap"],
                kv_bytes=16 * 2**30,
                bytes_per_token=costs.bytes_per_token,
                shared_fraction=case.get("decode_sharing", 0),
            ),
        )
        sim.comparison_stages = []
        if not case.get("peer_phase_world"):
            sim.world = CommonWorld(costs)
        sim.layout = replace(sim.layout, capacities=(case["host_gib"] * 2**30,) * 4)
        if case.get("cold"):
            sim.layout = replace(sim.layout, units=((),) * 4)
        if case.get("graph"):
            sys.path.insert(0, str(args.ours / "scripts/issue23"))
            from pd_graph_comparison import GraphWorld, configure_peer

            configure_peer(
                sim,
                GraphWorld(costs, read(args.graph_model), read(args.graph_components)),
            )
        result = sim.run()
    if case.get("graph"):
        result["graph_cost_model"] = read(args.graph_model)["metadata"]
        result["graph_scope"] = (
            "Measured step-cost adapter; P serial, CPU restore each request, GPU APC off"
        )
        if args.worker == "ours":
            result["settings"].update(
                mode="graph-cost2048",
                max_num_batched_tokens=2048,
                decode_unbatchable_fraction=None,
            )
    result["comparison_stages"] = sim.comparison_stages
    result["final_cache"] = [
        {
            "node": n,
            "bytes": sim.layout.usage(n),
            "units": len(units),
            "virtual_bytes": sum(u.charged for u in units if u.virtual),
            "protected_bytes": sum(u.charged for u in units if u.protected),
        }
        for n, units in enumerate(sim.layout.units)
    ]
    result["comparison"] = {
        "case": case,
        "engine": args.worker,
        "wall_seconds": time.perf_counter() - started,
        "adapters": contract["adapters"],
    }
    write(args.output, result)
    print(
        json.dumps(
            {
                "case": args.case,
                "engine": args.worker,
                "TTFT": result["TTFT"],
                "wall": result["comparison"]["wall_seconds"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ours", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--peer", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("ours", "peer"))
    parser.add_argument("--case")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--graph-model", type=Path)
    parser.add_argument("--graph-components", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    bundle = bundle_input(args.inputs)
    protocol = read(args.inputs / "protocol.json")
    if [r["id"] for r in bundle["rows"]] != protocol["test_order"]:
        raise ValueError("Request order mismatch")
    cases = [
        {
            "name": f"{kind}-host{host}-D{cap}",
            "cold": kind == "cold",
            "host_gib": host,
            "D_cap": cap,
        }
        for kind, host in (("cold", 16), ("cache", 2), ("cache", 16))
        for cap in (1, 4)
    ]
    cases += [
        {
            "name": "cache-host16-D4-sharing75",
            "host_gib": 16,
            "D_cap": 4,
            "decode_sharing": 0.75,
        },
        {
            "name": "cache-host16-D4-amortized",
            "host_gib": 16,
            "D_cap": 4,
            "amortized_restore": True,
        },
        {
            "name": "cache-host16-D4-phase-world",
            "host_gib": 16,
            "D_cap": 4,
            "peer_phase_world": True,
        },
        {
            "name": "cache-host2-D4-physical-copy",
            "host_gib": 2,
            "D_cap": 4,
            "commit_physical": True,
        },
        {
            "name": "cache-host16-D4-phase-and-resident",
            "host_gib": 16,
            "D_cap": 4,
            "amortized_restore": True,
            "resident_local": True,
        },
        {
            "name": "cache-host16-D4-phase-and-resident-sharing75",
            "host_gib": 16,
            "D_cap": 4,
            "amortized_restore": True,
            "resident_local": True,
            "decode_sharing": 0.75,
        },
    ]
    if args.graph_model:
        if not args.graph_components:
            raise ValueError("Graph comparison needs native save/restore components")
        cases = [
            {
                "name": f"graph-host16-D{cap}",
                "host_gib": 16,
                "D_cap": cap,
                "graph": True,
            }
            for cap in (1, 4)
        ]
    source = args.ours
    contract = {
        "kind": "synthetic-simulator-crosscheck",
        "source_commits": {
            name: subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            for name, root in (("ours", source), ("peer", args.peer))
        },
        "requests": len(bundle["rows"]),
        "shared": {
            "model": "Qwen2.5-7B-Instruct",
            "dtype": "bfloat16",
            "P": 2,
            "D": 2,
            "tp": 1,
            "P_cap": 1,
            "NPU_KV_GiB": 16,
            "block_tokens": 128,
            "token_budget": 128,
            "P_output": 1,
            "D_output": 128,
            "GPU_APC": False,
            "mode": "eager",
            "task_affinity": "sorted(task).index(task) % 2",
            "arrivals_seconds": protocol["arrival_condition"][
                "offered_arrivals_seconds"
            ],
            "session_causality": "ready=max(offered, previous session end)",
        },
        "adapters": {
            "routing": "fixed task-affinity pair; longest visible CPU prefix; local tie preference",
            "costs": "same frozen original singleton/copy totals; phase-world sensitivity separate",
            "batching": "prefill additive; decode=(1-s)*sum(single)+s*max(single)",
            "control": "zero modeled RPC/planner CPU in both; model-mechanism diagnostic",
            "peer_kernel": "BatchSimulation native execution; policy search disabled",
            "preserved_differences": [
                "restore phase placement",
                "cache publication timing",
                "allocation leases",
                "reclaim/fallback",
                "copy resource gates",
                "D source search: ours paired P/local, peer all four instances",
            ],
        },
        "cases": cases,
    }
    if args.graph_model:
        contract["shared"].update(mode="graph-cost", token_budget=2048)
        contract["graph_model_metadata"] = read(args.graph_model)["metadata"]
        contract["adapters"].update(
            costs="Same measured graph step function at each actual batch; native load/save measurements and historical RDMA coefficients",
            batching="Measured sequence/token/context-sensitive graph step cost; no shared-fraction assumption",
            graph_scope="P serial; no GPU APC; original kernel copy/publication/restore ordering retained",
        )
    write(args.output / "contract.json", contract)
    for case in cases:
        for engine in ("ours", "peer"):
            target = args.output / "runs" / f"{case['name']}-{engine}.json.gz"
            if target.exists():
                existing = read(target)["comparison"]
                if existing["case"] != case or existing["engine"] != engine:
                    raise ValueError(
                        "Existing result has a different experiment contract"
                    )
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--peer",
                str(args.peer),
                "--inputs",
                str(args.inputs),
                "--output",
                str(target),
                "--ours",
                str(args.ours),
                "--worker",
                engine,
                "--case",
                case["name"],
                "--contract",
                str(args.output / "contract.json"),
            ]
            if args.graph_model:
                command += [
                    "--graph-model",
                    str(args.graph_model),
                    "--graph-components",
                    str(args.graph_components),
                ]
            result = subprocess.run(
                command, text=True, capture_output=True, timeout=180
            )
            if result.returncode:
                write(
                    args.output / "failures" / f"{case['name']}-{engine}.json",
                    {
                        "returncode": result.returncode,
                        "stderr": result.stderr,
                        "stdout": result.stdout,
                    },
                )
                raise RuntimeError(f"{case['name']} {engine}: {result.stderr[-3000:]}")
            print(result.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
