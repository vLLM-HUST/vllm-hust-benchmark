"""Measured graph-step costs for the two preserved P/D execution kernels."""

from dataclasses import replace
from types import MethodType

from pd_serving_simulator import interpolate
from step_simulator import CostModel, features


class GraphWorld:
    def __init__(self, costs, model_data, components, token_budget=2048):
        self.costs = costs
        self.model = CostModel(model_data)
        self.native = components["native"]
        self.token_budget = token_budget
        self.gap = model_data["dispatch_gap_s"]
        self.response = model_data["response_delay_s"]
        self.save_tail = interpolate(self.native, 128, "save_RPC_seconds")
        self.fits = tuple(
            (0, 0, 0, 0, self.save_tail if n >= 2 else 0) for n in range(4)
        )

    def __getattr__(self, name):
        return getattr(self.costs, name)

    def step(self, assignments):
        return self.model.duration(features(assignments)) + self.gap

    def body(self, tokens, prefix):
        seconds = 0.0
        while prefix < tokens:
            amount = min(self.token_budget, tokens - prefix)
            seconds += self.step([(prefix, amount, tokens)])
            prefix += amount
        return seconds

    def restore(self, node, prefix):
        return interpolate(self.native, prefix, "load_complete_seconds")

    def compute(self, node, tokens, prefix):
        seconds = self.body(tokens, prefix) + self.restore(node, prefix) + self.response
        if node < 2:
            new_tokens = (tokens // 128 - prefix // 128) * 128
            seconds += interpolate(self.native, new_tokens, "save_RPC_seconds")
        return seconds

    def decode_tail(self, node, tokens):
        return (
            sum(self.step([(tokens + i, 1, tokens)]) for i in range(127))
            + self.save_tail
        )

    def copy_seconds(self, source, destination, payload):
        if not payload or source == destination:
            return 0.0
        slope = (
            self.costs.seconds_per_pd_byte
            if source < 2 <= destination
            else self.costs.seconds_per_dp_byte
            if destination < 2 <= source
            else self.costs.seconds_per_copy_byte
        )
        return self.costs.transfer_fixed + slope * payload


def ours_step(self, node):
    if node in self.gates or not self.engine[node]:
        return
    engine = self.engine[node]
    decoding = [key for key, state in engine.items() if state["remaining"] == 0]
    budget = self.protocol["prefill"] - len(decoding)
    heads, assignments, extra = [], [], 0.0
    for key, state in engine.items():
        if state["remaining"] and budget:
            amount = min(budget, state["remaining"])
            heads.append((key, amount))
            assignments.append((state["prompt"], amount, self.rows[key]["tokens"]))
            extra += state["graph_extra_per_token"] * amount
            budget -= amount
    for key in decoding:
        state, tokens = engine[key], self.rows[key]["tokens"]
        assignments.append((tokens + 127 - state["decode_left"], 1, tokens))
        if state["decode_left"] == 1:
            extra += self.world.save_tail
    duration = self.world.step(assignments) + extra
    self.steps.append(
        {
            "node": node,
            "start": self.now,
            "end": self.now + duration,
            "decoding": decoding,
            "heads": heads,
            "active": list(engine),
            "graph_features": features(assignments),
        }
    )
    self.gates[node] = "engine"
    self.event(self.now + duration, "engine_step", str(node), (node, decoding, heads))


def configure_ours(sim, world):
    sim.world = world
    sim.protocol = dict(sim.protocol, prefill=world.token_budget)
    original_api = sim.api

    def api(self, key, node):
        original_api(key, node)
        if node >= 2:
            state = self.engine[node][key]
            tokens = self.rows[key]["tokens"]
            prefix = state["prompt"]
            body = self.world.body(tokens, prefix)
            first = self.jobs[key]["stages"]["D"]["first_service"]
            state["graph_extra_per_token"] = (first - body) / state["remaining"]

    sim.api = MethodType(api, sim)
    sim.schedule_step = MethodType(ours_step, sim)


def configure_peer(sim, world):
    sim.world = world
    sim.batch_config = replace(sim.batch_config, token_budget=world.token_budget)
    for batch in sim.d_batches.values():
        batch.config = sim.batch_config
    original_begin = sim.begin_iteration

    def begin(self, node, batch):
        step = original_begin(node, batch)
        if step is None or step.kind != "compute":
            return step
        assignments, extra = [], 0.0
        for key, amount, phase in step.pieces:
            seq = batch.active[key]
            if phase == "prefill":
                computed = seq.prefix_tokens + seq.prefilled
                # Retain the peer restore barrier and amortize response overhead.
                extra += (
                    self.world.response
                    * amount
                    / (seq.prompt_tokens - seq.prefix_tokens)
                )
            else:
                computed = seq.prompt_tokens + seq.decoded - 1
            assignments.append((computed, amount, seq.prompt_tokens))
        step = replace(step, seconds=self.world.step(assignments) + extra)
        batch.busy = step
        return step

    sim.begin_iteration = MethodType(begin, sim)
