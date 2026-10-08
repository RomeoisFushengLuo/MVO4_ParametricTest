"""Regression tests for the discrepancies in the uploaded code and guide edges."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from mvo_algo import DataError, EVENTS, crash_curve, clean_prices, dbscores, optimize, preselect, summarize


class PipelineTests(unittest.TestCase):
    def test_inclusive_seeds_are_used(self):
        beta = pd.DataFrame([[0, 0], [0, .1], [3, 3], [3, 3.1]])
        with patch("mvo_algo.KMeans", wraps=KMeans) as fit:
            scores = dbscores(beta, 2, number_iteration=3)
            self.assertEqual([call.kwargs["random_state"] for call in fit.call_args_list], [1, 2, 3])
        self.assertEqual(sorted(s for s, _ in scores), [1, 2, 3])

    def test_vote_threshold_has_no_off_by_one(self):
        beta = pd.DataFrame([[i / 10, i / 10] for i in range(24)], index=[f"T{i:02}" for i in range(24)])
        selected, votes = preselect(beta, [(1, 0)], 2, 1, 2)
        self.assertEqual(len(selected), 0)
        self.assertTrue(votes.eq(1).all())

    def test_curve_front_rounding_and_length(self):
        curve = crash_curve(74, 216, .253876, 293)
        self.assertEqual(len(curve), 293)
        self.assertEqual(curve[0], 0)
        self.assertEqual(curve[73], .2539)
        np.testing.assert_array_equal(curve[-3:], [0, 0, 0])
        with self.assertRaises(DataError):
            crash_curve(74, 216, .253876, 289)

    def test_missing_prices_use_actual_row_and_neighbors(self):
        data = pd.DataFrame({"middle": [1, 4, np.nan, 16, 25], "first": [np.nan, 4, 8, 16, 32],
                             "last": [1, 2, 4, 8, np.nan]})
        clean = clean_prices(data)
        self.assertEqual(clean.loc[2, "middle"], 8)
        self.assertEqual(clean.loc[0, "first"], 2)
        self.assertEqual(clean.loc[4, "last"], 16)
        self.assertTrue(pd.isna(data.loc[2, "middle"]))  # no input mutation

    def test_ambiguous_missing_run_fails(self):
        with self.assertRaises(DataError):
            clean_prices(pd.DataFrame({"A": [1, np.nan, np.nan, 4]}))

    def test_infeasible_cap_has_no_fallback(self):
        expected = pd.Series([1.] * 5)
        weights, error = optimize(expected, pd.DataFrame(np.eye(5)), .17)
        self.assertIsNone(weights)
        self.assertIn("infeasible", error)

    def test_rf_zero_floor_and_renormalization(self):
        with patch("mvo_algo.EfficientFrontier") as frontier:
            frontier.return_value.clean_weights.return_value = {"A": .6, "B": .38, "C": .02}
            weights, error = optimize(pd.Series([1, 1, 1], index=["A", "B", "C"]),
                                      pd.DataFrame(np.eye(3), index=["A", "B", "C"], columns=["A", "B", "C"]), .7)
            frontier.return_value.max_sharpe.assert_called_once_with(risk_free_rate=0.0)
            self.assertIsNone(error)
            self.assertEqual(list(weights.index), ["A", "B"])
            self.assertAlmostEqual(weights.sum(), 1)
            self.assertAlmostEqual(weights.A, .6 / .98)

    def test_training_never_contains_predicted_swan(self):
        for name, event in EVENTS.items():
            self.assertNotIn(name.replace(" ", ""), event.train_swans)
            self.assertEqual(len(event.train_swans), 5)

    def test_summaries_exclude_incomplete_combinations(self):
        rows = []
        for K in [7, 8]:
            for event in ["Swan 8", "Swan 9"]:
                rows.append(dict(event=event, K=K, Seeds=3, MinAppear=3, w1=.17,
                                 status="opt_failed" if K == 8 and event == "Swan 9" else "ok", PR=.4))
        summary = summarize(pd.DataFrame(rows), ["Swan 8", "Swan 9"])
        self.assertEqual(summary["all_combos"].K.tolist(), [7])


if __name__ == "__main__":
    unittest.main()
