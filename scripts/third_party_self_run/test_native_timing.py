"""Clock-origin regressions, unrelated to the real measurement dataset."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))
from measurements import request_metrics


class Timing(unittest.TestCase):
    def test_epoch_arrival_is_not_subtracted_from_monotonic_first_token(self):
        result = request_metrics(
            {
                "arrival_time": 1794400000.0,
                "first_token_ts": 101.0,
                "first_token_latency": 0.75,
            },
            100.0,
            103.0,
            5,
        )
        self.assertEqual(result["ttft_ms"], 750)
        self.assertEqual(result["tpot_ms"], 500)

    def test_incompatible_monotonic_clock_is_rejected(self):
        with self.assertRaises(ValueError):
            request_metrics(
                {"first_token_ts": 1794400000.0, "first_token_latency": 0.75},
                100,
                103,
                5,
            )

    def test_one_token_has_no_tpot(self):
        result = request_metrics(
            {"first_token_time": 100.5, "arrival_time": 100, "finished_time": 101},
            10,
            11,
            1,
        )
        self.assertEqual(result["ttft_ms"], 500)
        self.assertIsNone(result["tpot_ms"])


if __name__ == "__main__":
    unittest.main()
