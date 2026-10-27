import unittest
from compare_tp2_structures import lane_signature, shared_lanes, union_ns


class LaneComparisonTests(unittest.TestCase):
    def test_lane_order_matters(self):
        a = (("a", "compute"), ("b", "compute"))
        self.assertEqual(shared_lanes([a], [tuple(reversed(a))]), set())

    def test_kind_matters(self):
        self.assertEqual(
            shared_lanes([(("a", "compute"),)], [(("a", "communication"),)]), set()
        )

    def test_duplicate_lane_is_ambiguous(self):
        a = (("a", "compute"),)
        with self.assertRaises(ValueError):
            shared_lanes([a, a], [a])

    def test_local_stream_ids_do_not_define_correspondence(self):
        a = lane_signature([dict(identity="a", kind="compute", stream_id=17)])
        b = lane_signature([dict(identity="a", kind="compute", stream_id=71)])
        self.assertEqual(shared_lanes([a], [b]), {a})

    def test_both_sides_can_have_specific_lanes(self):
        a, b, c = [(("x" + str(i), "compute"),) for i in range(3)]
        self.assertEqual(shared_lanes([a, b], [a, c]), {a})

    def test_ambiguous_provider_names_cannot_become_shared(self):
        lane = ((("ambiguous_provider_labels", ("op_1", "op_2")), "communication"),)
        with self.assertRaises(ValueError):
            shared_lanes([lane], [lane])
        self.assertEqual(
            shared_lanes([lane], [(("AivKernel", "communication"),)]), set()
        )

    def test_nested_union_is_not_task_sum(self):
        rows = [
            dict(start_ns=a, end_ns=b)
            for a, b in [(0, 100), (10, 20), (80, 130), (150, 150)]
        ]
        self.assertEqual(union_ns(rows), 130)


if __name__ == "__main__":
    unittest.main()
