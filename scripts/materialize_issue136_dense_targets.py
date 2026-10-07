#!/usr/bin/env python3
"""Materialize the versioned Dense targets required by benchmark issue #136."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path


WORKLOADS = (
    "random-online",
    "sharegpt-online",
    "prefix-repetition-online",
    "agent-research-online",
)
TPS = (1, 2, 4)
STACK_ID = "vllm-0.23.0-vllm-ascend-0.25.1rc1"
CORE_VERSION = "0.23.0"
PLUGIN_VERSION = "0.25.1rc1"
FULL_SHA = re.compile(r"[0-9a-f]{40}")
GRAPH_CAPTURE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
RateMap = dict[int, float]
RateMatrix = dict[str, RateMap]


def _parse_rates(value: str) -> RateMap:
    rates: RateMap = {}
    for item in value.split(","):
        key, separator, raw_rate = item.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError("rates must use TP=RPS entries")
        try:
            tp = int(key)
            rate = float(raw_rate)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("rates must be numeric") from exc
        if tp not in TPS or rate <= 0:
            raise argparse.ArgumentTypeError(f"invalid TP/rate entry: {item}")
        rates[tp] = rate
    if set(rates) != set(TPS):
        raise argparse.ArgumentTypeError("rates must define TP1, TP2, and TP4")
    return rates


def _rate_matrix(load_profile: str, rates: RateMap | RateMatrix) -> RateMatrix:
    if set(rates) == set(TPS):
        flat = rates
        if not all(isinstance(key, int) for key in flat):
            raise ValueError("rate keys must be tensor-parallel sizes")
        if load_profile != "fixed-1-rps":
            raise ValueError(
                "scaled-load requires workload-specific rates; a TP-only rate "
                "cannot be applied to every workload"
            )
        if set(flat.values()) != {1.0}:
            raise ValueError("fixed-1-rps requires TP1/2/4 request_rate=1")
        return {workload: dict(flat) for workload in WORKLOADS}

    if set(rates) != set(WORKLOADS):
        raise ValueError("workload rate matrix must define every issue #136 workload")
    matrix: RateMatrix = {}
    for workload in WORKLOADS:
        workload_rates = rates[workload]
        if not isinstance(workload_rates, dict) or set(workload_rates) != set(TPS):
            raise ValueError(f"{workload} rates must define TP1, TP2, and TP4")
        if any(
            not isinstance(tp, int)
            or not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or rate <= 0
            for tp, rate in workload_rates.items()
        ):
            raise ValueError(f"{workload} contains an invalid TP/rate entry")
        matrix[workload] = {tp: float(workload_rates[tp]) for tp in TPS}
    if load_profile == "fixed-1-rps" and {
        rate for workload_rates in matrix.values() for rate in workload_rates.values()
    } != {1.0}:
        raise ValueError("fixed-1-rps requires every workload/TP request_rate=1")
    return matrix


def _load_rate_matrix(path: Path) -> RateMatrix:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "issue-136-workload-rate-matrix/v1"
    ):
        raise ValueError("rate matrix has an unsupported schema")
    raw = payload.get("rates")
    if not isinstance(raw, dict) or set(raw) != set(WORKLOADS):
        raise ValueError("rate matrix must define every issue #136 workload")
    matrix: RateMatrix = {}
    expected_keys = {str(tp) for tp in TPS}
    for workload in WORKLOADS:
        workload_rates = raw[workload]
        if not isinstance(workload_rates, dict) or set(workload_rates) != expected_keys:
            raise ValueError(f"{workload} rates must define TP1, TP2, and TP4")
        matrix[workload] = {tp: float(workload_rates[str(tp)]) for tp in TPS}
    return matrix


def _rate_token(rate: float) -> str:
    return f"{rate:g}".replace(".", "p") + "rps"


def _template_path(repo: Path, workload: str) -> Path:
    return (
        repo
        / "docs"
        / "official-baselines"
        / f"official-ascend-jan-2026-v0180-{workload}-qwen25-14b-910b2.json"
    )


def _target_path(repo: Path, workload: str, tp: int, load_token: str) -> Path:
    return (
        repo
        / "docs"
        / "official-baselines"
        / (
            f"specialty-ascend-{STACK_ID}-{workload}-qwen25-14b-fp16-"
            f"tp{tp}-{load_token}-910b2.json"
        )
    )


def _build_target(
    template: dict[str, object],
    *,
    workload: str,
    tp: int,
    request_rate: float,
    load_profile: str,
    core_commit: str,
    plugin_commit: str,
) -> dict[str, object]:
    target = copy.deepcopy(template)
    load_token = (
        "fixed-1rps"
        if load_profile == "fixed-1-rps"
        else f"scaled-{_rate_token(request_rate)}"
    )
    target_id = (
        f"specialty-ascend-{STACK_ID}-{workload}-qwen25-14b-fp16-"
        f"tp{tp}-{load_token}-910b2"
    )
    target["id"] = target_id
    target["title"] = (
        f"Ascend 910B2 vLLM {CORE_VERSION} + vLLM Ascend {PLUGIN_VERSION} "
        f"{workload} Qwen2.5-14B FP16 TP{tp} {load_token}"
    )
    target["description"] = (
        "Issue #136 Dense scaling target. The ID states the runtime versions, "
        "workload, precision, tensor parallel size, and load profile directly."
    )
    target["chip_count"] = tp
    target["node_count"] = 1
    target["baseline_target"] = {
        "id": f"{STACK_ID}-{plugin_commit[:12]}",
        "label": f"vLLM {CORE_VERSION} + vLLM Ascend {PLUGIN_VERSION}",
        "engine": "vllm",
        "engine_version": CORE_VERSION,
        "github_repository": "vLLM-HUST/vllm-ascend-hust",
        "vllm_ref": core_commit,
        "vllm_ascend_ref": plugin_commit,
    }
    server = target["server_parameters"]
    assert isinstance(server, dict)
    server["tensor_parallel_size"] = tp
    server["compilation_config"] = {
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": list(GRAPH_CAPTURE_SIZES),
    }
    client = target["client_parameters"]
    assert isinstance(client, dict)
    client["request_rate"] = request_rate
    export = target["export"]
    assert isinstance(export, dict)
    export.update(
        {
            "engine": "vllm",
            "engine_version": CORE_VERSION,
            "submitter": "vllm-hust-issue-136",
            "baseline_engine": "vllm",
            "github_repository": "vLLM-HUST/vllm-ascend-hust",
            "github_ref": plugin_commit,
            "git_commit": plugin_commit,
            "data_source": "issue-136-current-main-dense",
        }
    )
    target["issue_136_contract"] = {
        "core_commit": core_commit,
        "plugin_commit": plugin_commit,
        "load_profile": load_profile,
        "request_rate": request_rate,
        "tensor_parallel_size": tp,
        "graph_capture_sizes": list(GRAPH_CAPTURE_SIZES),
    }
    return target


def materialize(
    repo: Path,
    *,
    core_commit: str,
    plugin_commit: str,
    load_profile: str,
    rates: RateMap | RateMatrix,
) -> list[Path]:
    if not FULL_SHA.fullmatch(core_commit) or not FULL_SHA.fullmatch(plugin_commit):
        raise ValueError(
            "core and plugin commits must be full lowercase 40-character SHAs"
        )
    matrix = _rate_matrix(load_profile, rates)

    written: list[Path] = []
    target_ids: set[str] = set()
    for workload in WORKLOADS:
        template = json.loads(
            _template_path(repo, workload).read_text(encoding="utf-8")
        )
        for tp in TPS:
            rate = matrix[workload][tp]
            load_token = (
                "fixed-1rps"
                if load_profile == "fixed-1-rps"
                else f"scaled-{_rate_token(rate)}"
            )
            target = _build_target(
                template,
                workload=workload,
                tp=tp,
                request_rate=rate,
                load_profile=load_profile,
                core_commit=core_commit,
                plugin_commit=plugin_commit,
            )
            target_id = str(target["id"])
            if target_id in target_ids:
                raise ValueError(f"duplicate target ID: {target_id}")
            target_ids.add(target_id)
            path = _target_path(repo, workload, tp, load_token)
            path.write_text(
                json.dumps(target, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--core-commit", required=True)
    parser.add_argument("--plugin-commit", required=True)
    parser.add_argument(
        "--load-profile", choices=("fixed-1-rps", "scaled-load"), required=True
    )
    rate_source = parser.add_mutually_exclusive_group(required=True)
    rate_source.add_argument(
        "--rates",
        type=_parse_rates,
        help="comma-separated TP=RPS entries, for example 1=1,2=1,4=1",
    )
    rate_source.add_argument(
        "--rate-matrix",
        type=Path,
        help=(
            "JSON issue-136-workload-rate-matrix/v1 file with an independently "
            "selected TP1/2/4 rate for every workload"
        ),
    )
    args = parser.parse_args()
    rates = (
        _load_rate_matrix(args.rate_matrix.resolve())
        if args.rate_matrix is not None
        else args.rates
    )
    written = materialize(
        args.repo.resolve(),
        core_commit=args.core_commit,
        plugin_commit=args.plugin_commit,
        load_profile=args.load_profile,
        rates=rates,
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
