"""Integration checks against the uploaded frozen SP500 inputs."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from run_sp500 import SP500Experiment, EVENTS


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/tmp", prefix="sp500-tests-")
        self.model = SP500Experiment(Path(__file__).resolve().parent / "inputs", Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_current_returns_are_validated_and_not_historical(self):
        self.assertEqual(len(self.model.analysts), 514)
        self.assertEqual(len(self.model.forward), 508)
        a = self.model.analysts.loc[self.model.forward.index]
        np.testing.assert_allclose(self.model.forward, a.P_fwd / a.MA30_Close - 1, rtol=0, atol=1e-10)
        self.assertFalse(a.historical_point_in_time.any())
        self.assertNotIn("AVB", self.model.forward.index)

    def test_price_eligibility_matches_uploaded_reports(self):
        expected = {"Swan 8": 367, "Swan 9": 405, "Swan 10D": 412, "Swan 11A": 419}
        for event in EVENTS:
            beta = self.model.training_beta(event)
            self.assertEqual(len(beta), expected[event])
            self.assertNotIn(event.replace(" ", ""), beta.columns)
            px, bm = self.model.prices(event)
            self.assertEqual(bm.name, "^GSPC")
            self.assertTrue(px.index.equals(bm.index))
            self.assertTrue(self.model.coverage[event].loc[beta.index, "complete_training_and_endpoint_prices"].all())

    def test_crash_depth_is_from_sp500_and_timing_stays_fixed(self):
        self.assertAlmostEqual(self.model.market.loc["Swan9", "Actual DD"], .3392496000265327)
        self.assertEqual(self.model.curve_params("Swan 8")[:2], (74, 216))
        self.assertEqual(str(self.model.training_window("Swan 8")[0].date()), "2017-08-07")


if __name__ == "__main__":
    unittest.main()
