"""Offline checks for cache coverage and current-input calculations, not live downloads."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from download_sp500 import WINDOWS, assemble


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sp500-assembly-", dir="/tmp")
        self.root = Path(self.temporary.name)
        for relative in ["raw/history", "raw/analysts", "cache/prices", "reports", "data"]:
            (self.root / relative).mkdir(parents=True)
        self.universe = pd.DataFrame({"Ticker": ["TEST"], "YahooSymbol": ["TEST"]})
        labels = sorted(set(s for w in WINDOWS.values() for s in w["betas"]))
        self.beta = pd.DataFrame(1., index=["TEST"], columns=labels)
        self.dates = pd.bdate_range("2017-05-16", datetime.now(timezone.utc).date())
        self.prices = pd.DataFrame({"Close": 100.}, index=self.dates)

    def tearDown(self):
        self.temporary.cleanup()

    def save_prices(self):
        self.prices.to_parquet(self.root / "raw/history/TEST.parquet")
        pd.DataFrame({"Close": 200.}, index=self.dates).to_parquet(self.root / "raw/history/SPY.parquet")

    def test_missing_downloads_do_not_produce_ready_cache(self):
        result = assemble(self.root, self.universe, self.beta)
        self.assertEqual(result["price_history_files"], 0)
        self.assertEqual(result["analyst_targets"], 0)
        self.assertTrue(all(r["complete_training_and_endpoints"] == 0 for r in result["event_coverage"]))
        self.assertFalse(list((self.root / "cache/prices").glob("*.parquet")))

    def test_coverage_and_current_returns(self):
        self.save_prices()
        (self.root / "raw/analysts/TEST.json").write_text(json.dumps(dict(
            targetMeanPrice=120., currency="USD", retrieved_at_utc="test fixture")))
        result = assemble(self.root, self.universe, self.beta)
        self.assertEqual(result["price_history_files"], 1)
        self.assertTrue(all(r["complete_training_and_endpoints"] == 1 for r in result["event_coverage"]))
        data = pd.read_csv(self.root / "data/current_analyst_inputs.csv")
        self.assertAlmostEqual(data.R_fwd_30_MA.iloc[0], .2)
        self.assertFalse(data.historical_point_in_time.iloc[0])

    def test_missing_bottom_and_stale_prices_are_not_usable(self):
        self.prices.loc["2020-03-23", "Close"] = np.nan
        self.prices = self.prices.loc[:"2025-04-18"]
        self.save_prices()
        (self.root / "raw/analysts/TEST.json").write_text(json.dumps(dict(
            targetMeanPrice=120., currency="USD", retrieved_at_utc="test fixture")))
        result = assemble(self.root, self.universe, self.beta)
        row = next(r for r in result["event_coverage"] if r["event"] == "Swan_9")
        self.assertEqual(row["complete_training_and_endpoints"], 0)
        self.assertEqual(result["usable_analyst_inputs"], 0)


if __name__ == "__main__":
    unittest.main()
