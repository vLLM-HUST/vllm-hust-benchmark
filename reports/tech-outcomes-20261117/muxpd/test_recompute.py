"""Run with: python -m unittest discover -s <this directory> -p test_recompute.py."""

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("muxpd_recompute", ROOT / "recompute.py")
if SPEC and SPEC.loader and (ROOT / "recompute.py").is_file():
    recompute = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(recompute)
else:
    recompute = None


def synthetic():
    # One success; three requests each fail exactly one SLO; two transport/empty failures.
    events = [[0.5, 0.5625], [1.125, 1.1875], [0.5, 0.75], [0.5, 1.125], [], []]
    rows, requests = [], []
    for i, ev in enumerate(events):
        out = 11 if i == 3 else 2
        kind = "long" if i == 3 else "short"
        requests.append({"id": str(i), "t": 0, "kind": kind, "in": 3, "out": out})
        rows.append(
            {
                "id": str(i),
                "kind": kind,
                "in": 3,
                "out": out,
                "sched": 0,
                "sent": 0,
                "events": ev,
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": out,
                    "total_tokens": 3 + out,
                }
                if ev
                else None,
                "error": "connection closed" if i == 4 else None,
            }
        )
    return {"t0": 0, "rows": rows}, {"requests": requests}


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(recompute, "portable recompute.py is missing")

    def test_percentile_interpolates_instead_of_rounding_rank(self):
        self.assertEqual(recompute.percentile([0, 10, 20, 30], 95), 28.5)
        self.assertIsNone(recompute.percentile([], 95))

    def test_conjunctive_slo_errors_stay_in_denominator(self):
        result, manifest = synthetic()
        s = recompute.summarize(result, manifest)
        self.assertEqual(
            (s["requests"], s["completed"], s["errors"], s["good"]), (6, 4, 2, 1)
        )
        self.assertEqual(s["goodput_frac"], 1 / 6)
        self.assertEqual(s["violations"], {"error": 2, "ttft": 1, "tpot": 1, "gap": 1})
        self.assertEqual(s["sse_event_count_equals_completion_tokens"], 3)
        self.assertAlmostEqual(s["maxgap_s"]["p95"], 0.56875)

    def test_slo_boundaries_are_inclusive_for_each_request_kind(self):
        result, manifest = synthetic()
        result["rows"] = result["rows"][:4]
        manifest["requests"] = manifest["requests"][:4]
        for row, req, events, out, kind in zip(
            result["rows"],
            manifest["requests"],
            [[1], [3], [0, 0.1], [0, 0.5]],
            [1, 1, 2, 6],
            ["short", "long", "short", "short"],
        ):
            row.update(
                events=events,
                out=out,
                kind=kind,
                usage={
                    "prompt_tokens": 3,
                    "completion_tokens": out,
                    "total_tokens": 3 + out,
                },
            )
            req.update(out=out, kind=kind)
        self.assertEqual(recompute.summarize(result, manifest)["good"], 4)

    def test_missing_duplicate_or_mismatched_request_rejected(self):
        for change in ("missing", "duplicate", "kind", "in", "out", "sched", "usage"):
            with self.subTest(change=change):
                result, manifest = synthetic()
                if change == "missing":
                    result["rows"].pop()
                elif change == "duplicate":
                    result["rows"][-1]["id"] = "0"
                elif change == "usage":
                    result["rows"][0]["usage"]["completion_tokens"] += 1
                elif change in ("in", "out", "sched"):
                    result["rows"][0][change] += 1
                else:
                    result["rows"][0]["kind"] = "long"
                with self.assertRaises(ValueError):
                    recompute.summarize(result, manifest)

    def test_backwards_or_nonfinite_events_rejected(self):
        for events in ([0.5, 0.25], [0.5, float("nan")]):
            result, manifest = synthetic()
            result["rows"][0]["events"] = events
            with self.assertRaises(ValueError):
                recompute.summarize(result, manifest)

    def test_real_sweep_has_exact_counts_and_unrounded_deltas(self):
        d = recompute.recompute(ROOT)
        self.assertEqual(len(d["workloads"]), 6)
        classes = [x["classification"] for x in d["workloads"].values()]
        self.assertEqual(classes.count("very good"), 5)
        self.assertEqual(classes.count("cost"), 1)
        c = d["workloads"]["wl-2wiki-r1.6"]["comparisons"]["tp2"]
        self.assertEqual(
            (c["native"]["good"], c["mux43"]["good"], c["native"]["requests"]),
            (10, 128, 229),
        )
        self.assertEqual(c["delta"]["goodput_pp"], 118 / 229 * 100)
        self.assertAlmostEqual(
            c["native"]["ttft_s"]["p95"], 28.262617470044642, places=5
        )
        self.assertEqual(d["totals"]["requests_across_all_arms"], 3716)
        self.assertEqual(d["totals"]["errors_across_all_arms"], 0)

    def test_absent_input_rejected_even_if_other_results_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            (root / "data/results/wl38-tp4/native/pass1/wl-chat-r1.5.json.gz").unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                recompute.recompute(root)

    def test_cli_check_detects_stale_metrics_without_modifying_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").symlink_to(ROOT / "data", target_is_directory=True)
            command = [sys.executable, str(ROOT / "recompute.py"), "--root", str(root)]
            written = subprocess.run(
                command + ["--write"], check=False, capture_output=True, text=True
            )
            self.assertEqual(written.returncode, 0, written.stderr)
            checked = subprocess.run(
                command + ["--check"], check=False, capture_output=True, text=True
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            target = root / "metrics-v1.json"
            target.write_text("{}\n")
            failed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(target.read_text(), "{}\n")


if __name__ == "__main__":
    unittest.main()
