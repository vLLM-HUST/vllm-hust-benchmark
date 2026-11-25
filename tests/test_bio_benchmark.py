"""CPU fixtures for biological measurement accounting and report reconstruction."""

import json
from pathlib import Path
import tempfile
import unittest

from vllm_hust_benchmark.bio_report import (
    latency_statistics,
    render_markdown,
    summarize,
)


def save(path, value):
    path.write_text(json.dumps(value))


class BioBenchmarkTests(unittest.TestCase):
    def build_fixture(self, root):
        # Temporary accounting fixtures carry no hardware measurement claim.
        protocol = {
            "independent_starts": 3,
            "warmup_repetitions": 1,
            "steady_repetitions": 2,
            "quantiles": [0.5, 0.9],
            "memory_growth_limit_bytes": 16 * 1024 * 1024,
        }
        case = {
            "id": "case",
            "profile_id": "esm2-150m",
            "task_id": "esm2.embedding",
            "scale": "short",
        }
        identity = {"pid": 1073741824, "pid_start_ticks": 1}
        memory = identity | {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
        runs = [
            {
                "profile_id": "esm2-150m",
                "start_index": -1,
                "fault": True,
                "directory": "fault",
            }
        ]
        fault = root / "fault"
        fault.mkdir()
        save(
            fault / "native.json",
            {
                "closed": True,
                "worker_initial": identity,
                "passed": False,
                "error": {
                    "type": "RuntimeError",
                    "message": "benchmark_injected_host_failure_after_native_output",
                },
                "requests": 1,
            },
        )
        for start in range(3):
            name = "start" + str(start)
            runs.append(
                {
                    "profile_id": "esm2-150m",
                    "start_index": start,
                    "fault": False,
                    "directory": name,
                }
            )
            directory = root / name
            directory.mkdir()
            rows = []
            phases = [
                ("first", 0, 1.0),
                ("warmup", 0, 2.0),
                ("steady", 0, 4.0),
                ("steady", 1, 6.0),
            ]
            phases.extend(("recovery", index, 1.0) for index in range(8))
            clock = 1000000
            for ordinal, (phase, repetition, elapsed) in enumerate(phases):
                finish = clock + int(elapsed * 1_000_000)
                rows.append(
                    {
                        "request_id": f"{start}/{ordinal}",
                        "case_id": "case",
                        "profile_id": "esm2-150m",
                        "start_index": start,
                        "phase": phase,
                        "repetition": repetition,
                        "status": "passed",
                        "prior_task_calls": ordinal,
                        "started_ns": clock,
                        "finished_ns": finish,
                        "elapsed_ms": elapsed,
                        "timing_scope": "prepared_native_call_to_cpu_output",
                        "memory_before": memory,
                        "memory_after": memory,
                        "work": {
                            "jobs": 1,
                            "input_sequences": 2,
                            "input_residues": 10,
                            "input_tokens": 14,
                            "generated_tokens": 0,
                        },
                    }
                )
                clock = finish + 1000000
            (directory / "requests.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )
            save(
                directory / "native.json",
                {
                    "profile_id": "esm2-150m",
                    "start_index": start,
                    "protocol": protocol,
                    "closed": True,
                    "passed": True,
                    "error": None,
                    "worker_initial": identity,
                    "requests": len(rows),
                    "engine_construct_ms": 10.0,
                    "rejections": [
                        {
                            "name": "invalid" + str(i),
                            "error": {"type": "ValueError"},
                            "memory_before": memory,
                            "memory_after": memory,
                        }
                        for i in range(8)
                    ],
                },
            )
        save(root / "suite.json", {"id": "temporary_accounting_fixture"})
        save(
            root / "manifest.json",
            {
                "pilot": True,
                "effective_protocol": protocol,
                "resolved_cases": [case],
                "runs": runs,
            },
        )

    def test_linear_quantiles_use_all_samples(self):
        values = latency_statistics([4.0, 6.0], [0.5, 0.9])
        self.assertEqual(values["mean_ms"], 5.0)
        self.assertEqual(values["quantiles_ms"]["0.5"], 5.0)
        self.assertAlmostEqual(values["quantiles_ms"]["0.9"], 5.8)

    def test_reconstruction_excludes_warmup_and_preserves_window_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_fixture(root)
            summary = summarize(root, numeric=False)
            case = summary["cases"][0]
            self.assertEqual(case["pooled_steady"]["count"], 6)
            self.assertEqual(case["pooled_steady"]["mean_ms"], 5.0)
            self.assertAlmostEqual(case["rates_per_second"]["jobs"], 6 / 0.033)
            self.assertEqual(summary["rejected_inputs"], 24)
            self.assertEqual(summary, summarize(root, numeric=False))
            self.assertIn("p90 ms", render_markdown(summary))

    def mutate_requests(self, mutate, expected):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_fixture(root)
            path = root / "start0/requests.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            mutate(rows)
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, expected):
                summarize(root, numeric=False)

    def test_clock_corruption_is_rejected(self):
        self.mutate_requests(lambda rows: rows[2].update(elapsed_ms=0.1), "clock")

    def test_duplicate_request_is_rejected(self):
        self.mutate_requests(
            lambda rows: rows[2].update(request_id=rows[1]["request_id"]), "Duplicate"
        )

    def test_steady_repetition_cannot_be_relabelled_as_warmup(self):
        self.mutate_requests(lambda rows: rows[2].update(phase="warmup"), "missing")

    def test_incomplete_recovery_is_rejected(self):
        self.mutate_requests(lambda rows: rows[-1].update(phase="steady"), "recovery")

    def test_missing_independent_start_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_fixture(root)
            path = root / "manifest.json"
            value = json.loads(path.read_text())
            value["runs"].pop()
            save(path, value)
            with self.assertRaisesRegex(ValueError, "Independent"):
                summarize(root, numeric=False)

    def test_fault_cleanup_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_fixture(root)
            path = root / "fault/native.json"
            value = json.loads(path.read_text())
            value["closed"] = False
            save(path, value)
            with self.assertRaisesRegex(ValueError, "active"):
                summarize(root, numeric=False)


if __name__ == "__main__":
    unittest.main()
