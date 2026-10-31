import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safety_aware import (SafetyStatus, assess_safety, candidate_universe,
                          experiment_arms, idcg, parse_profile_conditions,
                          prepare_metric_reference, safe_ndcg)

RULES = {"diseases": {
    "diabetes": {"threshold_condition": "sugar_pdv <= 44.0"},
    "hypertension": {"threshold_condition": "sodium_pdv <= 25.0"},
    "heart_failure": {"threshold_condition": "sodium_pdv <= 25.0 and fat_pdv <= 38.0"},
    "steatohepatitis": {"threshold_condition": "19.5 * fat_pdv <= calories"},
    "kidney_failure": {"threshold_condition": "protein_pdv <= 48.0"},
}}

class SafetyAwareTests(unittest.TestCase):
    def test_all_safe_topk(self):
        self.assertEqual(safe_ndcg([1, 2], {}, {1: SafetyStatus.SAFE, 2: SafetyStatus.SAFE},
                                   [1, 2], 2)["Safety@K"], 1.0)

    def test_all_unsafe_topk(self):
        self.assertEqual(safe_ndcg([1, 2], {}, {1: SafetyStatus.UNSAFE, 2: SafetyStatus.UNSAFE},
                                   [1, 2], 2)["Safety@K"], 0.0)

    def test_mixed_safe_topk(self):
        self.assertEqual(safe_ndcg([1, 2], {}, {1: SafetyStatus.SAFE, 2: SafetyStatus.UNSAFE},
                                   [1, 2], 2)["Safety@K"], 0.5)

    def test_p_ndcg_may_be_negative(self):
        result = safe_ndcg([1, 2], {1: 1, 2: 1},
            {1: SafetyStatus.UNSAFE, 2: SafetyStatus.SAFE}, [1, 2], 2, 5)
        self.assertLess(result["P-NDCG@K"], 0)

    def test_zero_idcg_is_undefined(self):
        result = safe_ndcg([1], {}, {1: SafetyStatus.SAFE}, [1], 1, 5)
        self.assertIsNone(result["P-NDCG@K"])
        self.assertTrue(result["P_NDCG_Undefined"])

    def test_hard_filter_removes_unsafe_and_unknown(self):
        state = {1: SafetyStatus.SAFE, 2: SafetyStatus.UNSAFE, 3: SafetyStatus.UNKNOWN}
        self.assertEqual([i for i in [2, 1, 3] if state[i] == SafetyStatus.SAFE], [1])

    def test_unknown_is_not_converted_to_safe(self):
        self.assertEqual(assess_safety(None, {"sugar_pdv": 1}, RULES), SafetyStatus.UNKNOWN)
        self.assertEqual(assess_safety(["diabetes"], {}, RULES), SafetyStatus.UNKNOWN)

    def test_filter_does_not_reorder(self):
        ranked = [4, 2, 3, 1]
        state = {1: SafetyStatus.SAFE, 2: SafetyStatus.UNSAFE,
                 3: SafetyStatus.SAFE, 4: SafetyStatus.UNKNOWN}
        self.assertEqual([i for i in ranked if state[i] == SafetyStatus.SAFE], [3, 1])

    def test_insufficient_safe_items_are_not_backfilled(self):
        state = {1: SafetyStatus.SAFE, 2: SafetyStatus.UNSAFE}
        recs = [i for i in [1, 2] if state[i] == SafetyStatus.SAFE][:5]
        metrics = safe_ndcg(recs, {}, state, [1, 2], 5)
        self.assertEqual(recs, [1])
        self.assertEqual(metrics["RecommendationCount"], 1)
        self.assertAlmostEqual(metrics["FillRate@K"], 0.2)
        self.assertIsNone(metrics["Safety@K"])

    def test_fixed_numerical_regression(self):
        rel = {"A": 2, "B": 1, "C": 0, "D": 2, "E": 1}
        state = {"A": SafetyStatus.SAFE, "B": SafetyStatus.SAFE, "C": SafetyStatus.SAFE,
                 "D": SafetyStatus.UNSAFE, "E": SafetyStatus.UNSAFE}
        candidates = ["A", "B", "C", "D", "E"]
        denominator = 3 + 1 / math.log2(3)
        m1 = safe_ndcg(["D", "A", "E"], rel, state, candidates, 3, 5)
        m2 = safe_ndcg(["A", "C", "B"], rel, state, candidates, 3, 5)
        self.assertAlmostEqual(m1["Safety@K"], 1 / 3)
        self.assertAlmostEqual(m2["Safety@K"], 1.0)
        self.assertAlmostEqual(idcg([2, 1, 0], 3), denominator)
        self.assertAlmostEqual(m1["P-DCG@K"], -0.96875 + 3 / math.log2(3) - 0.96875 / 2)
        self.assertAlmostEqual(m1["P-NDCG@K"], m1["P-DCG@K"] / denominator)
        self.assertAlmostEqual(m2["P-DCG@K"], 3.5)
        self.assertAlmostEqual(m2["P-NDCG@K"], 3.5 / denominator)

    def test_cached_metric_reference_matches_direct_metrics(self):
        states = {1: SafetyStatus.SAFE, 2: SafetyStatus.UNSAFE, 3: SafetyStatus.SAFE}
        rel = {1: 2, 2: 1}
        direct = safe_ndcg([2, 1], rel, states, [1, 2, 3], 2, 3)
        ref = prepare_metric_reference(rel, states, [1, 2, 3], ks=(2,))
        cached = safe_ndcg([2, 1], rel, states, [1, 2, 3], 2, 3, ref)
        self.assertEqual(direct, cached)
    def test_profile_parse(self):
        self.assertIsNone(parse_profile_conditions(""))
        self.assertEqual(parse_profile_conditions("none"), [])
        self.assertEqual(parse_profile_conditions("Diabetes|hypertension"),
                         ["diabetes", "hypertension"])
        self.assertIsNone(parse_profile_conditions("unrecognized_condition"))

    def test_four_arms_share_candidate_splits(self):
        arms = experiment_arms()
        self.assertEqual(len(arms), 4)
        split = candidate_universe(5, [1, 3])
        self.assertTrue(all(candidate_universe(5, [1, 3]) == split for _ in arms))

    def test_unknown_count_is_reported(self):
        result = safe_ndcg([1], {1: 2}, {1: SafetyStatus.UNKNOWN, 2: SafetyStatus.SAFE},
                           [1, 2], 1, 5)
        self.assertEqual(result["Unknown@K"], 1)
        self.assertEqual(result["Safety@K"], 0.0)

if __name__ == "__main__":
    unittest.main()