"""Boundary checks for evidence selection, numeric arithmetic and public field limits."""

import copy
import gzip
import json
import unittest

import public_evidence as evidence


class PublicEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = evidence.ROOT / "public"
        cls.rows = [
            json.loads(line)
            for line in gzip.decompress(
                (root / "pressure-default-requests.jsonl.gz").read_bytes()
            ).splitlines()
        ]
        cls.phases = json.loads(
            gzip.decompress((root / "pressure-default-phases.json.gz").read_bytes())
        )

    def evaluate(self, rows=None, phases=None):
        return evidence.evaluate_arm(
            rows if rows is not None else self.rows,
            phases if phases is not None else self.phases,
            evidence.GROUPS["full128"],
            plugin_enabled=True,
        )

    def test_complete_public_evidence_includes_all_negative_pairs(self):
        result = evidence.recompute()
        pairs = result["groups"]["full32"]["comparisons"]
        self.assertLess(pairs["r1-working-set"]["output_tok_s_change_pct"], 0)
        self.assertLess(pairs["r2-working-set"]["output_tok_s_change_pct"], 0)
        self.assertGreater(pairs["r3-working-set"]["output_tok_s_change_pct"], 0)

    def test_incomplete_or_half_run_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing/extra requests"):
            self.evaluate(rows=self.rows[: len(self.rows) // 2])

    def test_duplicate_request_rejected(self):
        rows = [self.rows[0], *self.rows[:-1]]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.evaluate(rows=rows)

    def test_cold_round_cannot_be_removed(self):
        phases = copy.deepcopy(self.phases)
        phases["phases"] = phases["phases"][1:]
        with self.assertRaisesRegex(ValueError, "Missing/duplicate phase"):
            self.evaluate(phases=phases)

    def test_tpot_formula_cannot_be_changed(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["tpot_s"] *= 2
        with self.assertRaisesRegex(ValueError, "TPOT"):
            self.evaluate(rows=rows)

    def test_counter_reset_rejected(self):
        phases = copy.deepcopy(self.phases)
        p = phases["phases"][0]
        key = next(iter(p["metrics_after"]))
        p["metrics_after"][key] = p["metrics_before"].get(key, 0) - 1
        with self.assertRaisesRegex(ValueError, "Counter reset"):
            self.evaluate(phases=phases)

    def test_candidate_missing_transfer_counters_rejected(self):
        phases = copy.deepcopy(self.phases)
        for when in ("metrics_before", "metrics_after"):
            phases["phases"][0][when] = {
                k: v
                for k, v in phases["phases"][0][when].items()
                if not k.startswith("vllm:kv_offload_load_bytes_total")
            }
        with self.assertRaisesRegex(ValueError, "Missing core metric"):
            self.evaluate(phases=phases)

    def test_wrong_phase_denominator_rejected(self):
        phases = copy.deepcopy(self.phases)
        phases["phases"][0]["output_tok_s"] *= 2
        with self.assertRaisesRegex(ValueError, "phase throughput"):
            self.evaluate(phases=phases)

    def test_public_projection_omits_hidden_contents(self):
        original = {k: self.rows[0][k] for k in evidence.REQUEST_FIELDS}
        original.update(
            chunk_times_s=[1.0, 2.0],
            output_text="not for publication",
            prompt=[1, 2, 3],
            prompt_file="private path",
        )
        projected = evidence.project_request(original)
        self.assertEqual(set(projected), set(evidence.PUBLIC_FIELDS))
        self.assertEqual(projected["first_chunk_s"], 1.0)
        self.assertEqual(projected["nonempty_chunk_count"], 2)
        rows = copy.deepcopy(self.rows)
        rows[0]["output_text"] = "not for publication"
        with self.assertRaisesRegex(ValueError, "Unexpected/missing public"):
            self.evaluate(rows=rows)


if __name__ == "__main__":
    unittest.main()
