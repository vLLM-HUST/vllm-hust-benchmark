"""Boundary tests use mutations of retained numeric rows, never new measurements."""

import copy
import unittest

import evidence


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = evidence.load_rows("B0")
        cls.identities = {
            row["case_id"]: row for row in evidence.read(evidence.ROOT / "inputs.json")
        }

    def test_all_numeric_results(self):
        evidence.check()

    def test_reject_generated_text_in_numeric_export(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["content"] = "Synthetic forbidden output field"
        with self.assertRaisesRegex(ValueError, "allowlist"):
            evidence.check_rows(rows, self.identities)

    def test_reject_output_length_change(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["output_tokens"] = 33
        with self.assertRaisesRegex(ValueError, "fixed output"):
            evidence.check_rows(rows, self.identities)

    def test_reject_duplicate_request(self):
        rows = copy.deepcopy(self.rows)
        rows[-1] = rows[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evidence.check_rows(rows, self.identities)

    def test_reject_request_identity_change(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["input_sha256"] = "synthetic-not-a-request-hash"
        with self.assertRaisesRegex(ValueError, "request identity"):
            evidence.check_rows(rows, self.identities)


if __name__ == "__main__":
    unittest.main()
