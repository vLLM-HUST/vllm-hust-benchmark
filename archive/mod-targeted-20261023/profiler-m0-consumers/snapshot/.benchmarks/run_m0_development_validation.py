"""M0 development-level validation of the evaluation pipeline.

Cases (NOT blind-scored; ground truth is known from prior analysis):
  A: G4/G6 restore tail (tiering_enabled vs tiering_disabled) -> restore/wakeup
  B: concurrency-2 prefill tail (vs concurrency-1) -> prefill
  C: abstain negative case (no single root cause)
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from vllm_request_lifecycle_profiler.m0_evaluation import (
    RequestPhases,
    evaluate_case,
    load_phase_rows_csv,
    phases_from_kv_recovery_and_client,
    report_to_dict,
)

RUNS = Path("/root/kv-recovery-service-g4/runs")
RES = Path(__file__).resolve().parents[1] / ".benchmarks/results"


def _client_totals(run_prefix: str) -> dict[str, float]:
    """Map request labels (req0..req3) to end-to-end ms from pressure_client.out."""
    dirs = glob.glob(str(RUNS / run_prefix))
    totals: dict[str, float] = {}
    for d in dirs:
        out = Path(d) / "pressure_client.out"
        if not out.exists():
            continue
        for line in out.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = rec.get("label")
            total_s = rec.get("total_s")
            if label and isinstance(total_s, (int, float)):
                totals[label] = float(total_s) * 1000.0
    return totals


def _kv_recovery_span_ms(run_prefix: str) -> dict[str, float]:
    """Return recovered-request -> preempt->first_compute span ms."""
    spans: dict[str, float] = {}
    for d in glob.glob(str(RUNS / run_prefix)):
        for shard in glob.glob(str(Path(d) / "trace.rlp-kv-recovery.*.jsonl")):
            start: dict[str, int] = {}
            for line in Path(shard).read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("record_type") != "recovery_event":
                    continue
                req, stage, ts = ev.get("runtime_request_id"), ev.get("stage"), ev.get("timestamp_ns")
                if not req or not stage or not isinstance(ts, int):
                    continue
                if stage == "preempt":
                    start[req] = ts
                elif stage == "first_prefill_or_decode" and req in start:
                    spans[req] = max(0.0, (ts - start[req]) / 1e6)
    return spans


def case_a_rows(mode_prefix: str) -> list[RequestPhases]:
    """Build per-request phases for enabled(anomaly)/disabled(baseline)."""
    totals = _client_totals(mode_prefix)
    spans = _kv_recovery_span_ms(mode_prefix)
    rows = []
    # if spans exist, the recovered request is the max-total one; assign recovery span
    recovered = None
    if spans:
        # identify recovered request's label: the max total belongs to the recovered one
        pass
    # Build per-request: recovered gets {kv_recovery: span, decode: total-span}
    # others get {decode: total}. Since client labels are req0..req3 and totals list is
    # in label order, use totals directly.
    # If we have a recovery span, attribute it to the max-total request.
    for label, total in totals.items():
        if spans and label == max(totals, key=totals.get):
            span = next(iter(spans.values()), 0.0)
            rows.append(RequestPhases(label, {"kv_recovery": span, "decode": max(0.0, total - span)}))
        else:
            rows.append(RequestPhases(label, {"decode": total}))
    return rows


def run() -> None:
    out = {}

    # ---- Case A: restore tail (G6 matched pair + G4 corroboration) ----
    base = case_a_rows("f4b1b083*") + case_a_rows("0fe82cc1*") + case_a_rows("d20f2126*")
    base += case_a_rows("eaa94f8b*") + case_a_rows("73d47651*") + case_a_rows("ababc337*")
    anomaly = case_a_rows("7d36ef9f*") + case_a_rows("c04e4681*") + case_a_rows("b2e2a7cf*")
    anomaly += case_a_rows("24b0508b*") + case_a_rows("adf18208*") + case_a_rows("fb81d458*")
    rep_a = evaluate_case(base, anomaly, case_id="g4-g6_restore_tail", ground_truth="restore/wakeup")
    out["case_A_restore_tail"] = report_to_dict(rep_a)

    # ---- Case B: concurrency-2 prefill tail ----
    collection = Path(os.environ["BENCHMARK_COLLECTION_ROOT"])
    if not collection.is_absolute() or collection.resolve().is_relative_to(
        Path(__file__).resolve().parents[1]
    ):
        raise ValueError("BENCHMARK_COLLECTION_ROOT must be outside the checkout")
    csv_path = collection / ".benchmarks/results/npu6_runtime_hooks_concurrency_sweep/runtime_internal_spans.csv"
    rows_all = load_phase_rows_csv(csv_path)
    rows_measured = [r for r in rows_all if "warmup" not in str(r.request_id) or True]
    # filter by assigned_concurrency is not in the RequestPhases; reload with concurrency
    base_b, anomaly_b = [], []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        import csv as _csv
        for rec in _csv.DictReader(handle):
            if rec.get("phase") != "measured":
                continue
            phases = {}
            for name in ("tokenization", "queueing", "prefill", "first_token_gap", "decode", "streaming", "cleanup"):
                if rec.get(name):
                    try:
                        phases[name if name != "first_token_gap" else "decode"] = float(rec[name])
                    except ValueError:
                        pass
            rp = RequestPhases(rec.get("chain_id", "?"), phases)
            if rec.get("assigned_concurrency") == "1":
                base_b.append(rp)
            elif rec.get("assigned_concurrency") == "2":
                anomaly_b.append(rp)
    rep_b = evaluate_case(base_b, anomaly_b, case_id="concurrency2_prefill_tail", ground_truth="prefill")
    out["case_B_concurrency2_prefill_tail"] = report_to_dict(rep_b)

    # ---- Case C: abstain (synthetic no-single-root-cause: deltas spread across phases) ----
    synth_base = [
        RequestPhases("s1", {"queueing": 100.0, "prefill": 200.0, "decode": 300.0}),
        RequestPhases("s2", {"queueing": 110.0, "prefill": 210.0, "decode": 310.0}),
        RequestPhases("s3", {"queueing": 105.0, "prefill": 205.0, "decode": 305.0}),
    ]
    synth_anom = [
        RequestPhases("s1", {"queueing": 125.0, "prefill": 220.0, "decode": 320.0}),
        RequestPhases("s2", {"queueing": 130.0, "prefill": 230.0, "decode": 330.0}),
        RequestPhases("s3", {"queueing": 127.0, "prefill": 225.0, "decode": 325.0}),
    ]
    rep_c = evaluate_case(synth_base, synth_anom, case_id="synthetic_no_single_root_cause",
                          ground_truth=None, abstain_threshold_ms=1.0)
    out["case_C_abstain"] = report_to_dict(rep_c)

    # ---- Case D: abstain (concurrency-1 vs concurrency-3, low evidence) ----

    base_c, anomaly_c = [], []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        import csv as _csv
        for rec in _csv.DictReader(handle):
            if rec.get("phase") != "measured":
                continue
            phases = {}
            for name in ("tokenization", "queueing", "prefill", "first_token_gap", "decode", "streaming", "cleanup"):
                if rec.get(name):
                    try:
                        phases[name if name != "first_token_gap" else "decode"] = float(rec[name])
                    except ValueError:
                        pass
            rp = RequestPhases(rec.get("chain_id", "?"), phases)
            if rec.get("assigned_concurrency") == "1":
                base_c.append(rp)
            elif rec.get("assigned_concurrency") == "3":
                anomaly_c.append(rp)
    rep_d = evaluate_case(base_c, anomaly_c, case_id="concurrency1_vs_3_low_evidence", ground_truth=None,
                          abstain_threshold_ms=50.0)
    out["case_D_concurrency1_vs_3_low_evidence"] = report_to_dict(rep_d)

    dest = RES / "m0_development_validation_20260811"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    run()
