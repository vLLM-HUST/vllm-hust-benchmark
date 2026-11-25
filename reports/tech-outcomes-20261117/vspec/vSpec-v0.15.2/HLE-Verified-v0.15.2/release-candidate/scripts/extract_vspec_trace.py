"""Extract every adaptive decision and switch from a completed trace run.

The complete server.log remains the primary evidence. The generated JSONL and
JSON index are navigational copies with source line numbers, not a new run.
"""

import argparse
import ast
import json
import re
from pathlib import Path


DECISION = re.compile(
    r"vSpec Adaptive decision: batch=(?P<batch>\d+) "
    r"context_tokens=(?P<context_tokens>\d+) "
    r"gamma=(?P<gamma>\d+) previous_gamma=(?P<previous_gamma>\d+) "
    r"acceptance=(?P<acceptance>[\d.]+).*?reason=(?P<reason>\S+)"
)


def extract(run_dir: Path) -> None:
    log_path = run_dir / "server.log"
    receipt_path = run_dir / "solver-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["arm"] != "dynamic" or receipt["record_count"] != receipt["case_count"]:
        raise RuntimeError(f"Trace solver run is incomplete: {run_dir}")
    decisions = []
    summaries = []
    bridge_commits = []
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            match = DECISION.search(line)
            if match:
                observed_previous_gamma = decisions[-1]["gamma"] if decisions else None
                row = {
                    "decision_index": len(decisions) + 1,
                    "line": line_number,
                    "log_prefix": line.split("vSpec Adaptive decision:", 1)[0].strip(),
                    "batch": int(match["batch"]),
                    "context_tokens": int(match["context_tokens"]),
                    "gamma": int(match["gamma"]),
                    "observed_previous_gamma": observed_previous_gamma,
                    "previous_gamma": int(match["previous_gamma"]),
                    "acceptance": float(match["acceptance"]),
                    "reason": match["reason"],
                }
                # The controller can update previous_gamma inside a prefix-bridge
                # transition before this log line. Compare consecutive logged
                # decisions to recover the actual observed gamma sequence.
                row["changed"] = (
                    observed_previous_gamma is not None
                    and row["gamma"] != observed_previous_gamma
                )
                decisions.append(row)
            if "[vspec-prefix-bridge-committed]" in line:
                payload = line.split("[vspec-prefix-bridge-committed]", 1)[1].strip()
                try:
                    bridge = json.loads(payload)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Cannot parse bridge commit at line {line_number}") from error
                bridge_commits.append({"line": line_number, **bridge})
            if "vSpec Online Adaptive summary:" in line:
                payload = line.split("vSpec Online Adaptive summary:", 1)[1].strip()
                try:
                    summary = ast.literal_eval(payload)
                except (SyntaxError, ValueError) as error:
                    raise RuntimeError(f"Cannot parse adaptive summary at line {line_number}") from error
                summaries.append((line_number, summary))
    if not decisions:
        raise RuntimeError(f"No per-decision trace in {log_path}")
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one final adaptive summary, found {len(summaries)}")
    summary_line, summary = summaries[0]
    switches = [row for row in decisions if row["changed"]]
    out_decisions = run_dir / "trace-decisions.jsonl"
    with out_decisions.open("x", encoding="utf-8") as stream:
        for row in decisions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "schema_version": 1,
        "dataset": receipt["dataset"],
        "run_dir": str(run_dir),
        "server_log": str(log_path),
        "solver_receipt": str(receipt_path),
        "decision_count": len(decisions),
        "trace_switch_count": len(switches),
        "summary_line": summary_line,
        "summary_initial_gamma": summary.get("initial_gamma"),
        "summary_gamma_switches": summary.get("gamma_switches"),
        "switch_count_matches_summary": len(switches) == summary.get("gamma_switches"),
        "summary_selection_counts": summary.get("selection_counts"),
        "summary_reason_counts": summary.get("reason_counts"),
        "bridge_commit_count": len(bridge_commits),
        "bridge_commits": bridge_commits,
        "switches": switches,
        "note": "Switches compare consecutive logged gamma values. The raw previous_gamma/reason fields may already reflect a committed bridge and need not identify its cause; bridge_commits retains those explicit events. Trace mode has logging/runtime overhead; do not mix its throughput with trace-off formal pairing.",
    }
    with (run_dir / "trace-switches.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"switches", "bridge_commits"}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    extract(parser.parse_args().run_dir)
