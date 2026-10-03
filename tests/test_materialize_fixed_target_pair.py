from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/materialize_fixed_target_pair.py"
SPEC = importlib.util.spec_from_file_location("materialize_fixed_target_pair", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_materialize_pair_calculates_metrics_and_preserves_unknowns(
    tmp_path: Path,
) -> None:
    common = {
        "same_spec": {"resolved_spec_hash": "a" * 64},
        "constraints": {
            "accountable_scope": {},
            "metrics": {
                "long_context_ttft_p95_ms": 30.0,
                "long_context_tpot_p95_ms": 4.0,
            },
        },
    }
    baseline = {**common, "engine": "vllm", "metrics": {"throughput_tps": 10, "ttft_ms": 20, "tbt_ms": 5}}
    current = {**common, "engine": "vllm-hust", "metrics": {"throughput_tps": 12, "ttft_ms": 15, "tbt_ms": 4}}
    raw = {key: [1, 2] for key in ("input_lens", "output_lens", "ttfts", "itls", "errors")}

    MODULE.materialize(
        _write(tmp_path / "baseline.json", baseline),
        _write(tmp_path / "current.json", current),
        _write(tmp_path / "baseline-raw.json", raw),
        _write(tmp_path / "current-raw.json", raw),
        tmp_path / "out",
    )

    summary = json.loads((tmp_path / "out/pair_summary.json").read_text())
    assert summary["derived"]["throughput_ratio"]["value"] == 1.2
    assert summary["derived"]["ttft_reduction_pct"]["value"] == 25.0
    assert summary["hard_constraint_inputs"]["unknown"]["single_chip_effective_utilization_pct"]["status"] == "not-measured"


def test_materialize_pair_rejects_spec_hash_mismatch(tmp_path: Path) -> None:
    baseline = {"same_spec": {"resolved_spec_hash": "a" * 64}}
    current = {"same_spec": {"resolved_spec_hash": "b" * 64}}
    raw = {}

    try:
        MODULE.materialize(
            _write(tmp_path / "baseline.json", baseline),
            _write(tmp_path / "current.json", current),
            _write(tmp_path / "baseline-raw.json", raw),
            _write(tmp_path / "current-raw.json", raw),
            tmp_path / "out",
        )
    except ValueError as error:
        assert "identical" in str(error)
    else:
        raise AssertionError("expected hash mismatch to fail")
