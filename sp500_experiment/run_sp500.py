"""Same MVO4 grid on uploaded SP500 prices, using a CURRENT analyst snapshot."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import sys

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "replication"))
import numpy as np
import pandas as pd
import sklearn
from mvo_algo import Replication, EVENTS, GRID, SWAN_ORDER, PARAMETERS, versions, write_outputs


class SP500Experiment(Replication):
    def __init__(self, inputs: Path, cache: Path):
        if sklearn.__version__ != "1.7.2":
            raise RuntimeError("Use scikit-learn 1.7.2 for reproducible clustering")
        self.source = inputs.resolve()
        self.cache = cache.resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.beta = pd.read_csv(self.source / "data/historical_beta_sp500.csv", index_col=0)
        self.beta = self.beta.rename(columns={"S9": "Swan9", "S10D": "Swan10D", "S11A": "Swan11A"})[SWAN_ORDER]
        if self.beta.index.has_duplicates:
            raise ValueError("Duplicate beta tickers")
        self.market = pd.read_csv(self.source / "market_sp500.csv", index_col=0)
        self.analysts = pd.read_csv(self.source / "data/current_analyst_inputs.csv", index_col="Symbol")
        if self.analysts.index.has_duplicates:
            raise ValueError("Duplicate analyst tickers")
        a = self.analysts
        numeric = ["P_fwd", "MA30_Close", "R_fwd_30_MA"]
        a[numeric] = a[numeric].apply(pd.to_numeric, errors="raise")
        implied = a.P_fwd / a.MA30_Close - 1
        complete = a[numeric].notna().all(axis=1)
        if not np.allclose(implied[complete], a.loc[complete, "R_fwd_30_MA"], rtol=0, atol=1e-10):
            raise ValueError("Supplied analyst return does not equal target/current 30-day MA - 1")
        if not pd.api.types.is_bool_dtype(a.usable) or not pd.api.types.is_bool_dtype(a.historical_point_in_time):
            raise ValueError("Analyst flags must be booleans")
        if a.historical_point_in_time.any():
            raise ValueError("This runner expects an explicitly current, non-historical snapshot")
        snapshot = pd.to_datetime(a.snapshot_retrieved_at_utc, utc=True)
        last_ma = pd.to_datetime(a.MA30_end, utc=True)
        age = (snapshot.dt.normalize() - last_ma).dt.days
        valid = (a.usable & complete & np.isfinite(a[numeric]).all(axis=1) & a.currency.eq("USD")
                 & a.P_fwd.gt(0) & a.MA30_Close.gt(0) & age.between(0, 14))
        self.forward = a.loc[valid, "R_fwd_30_MA"]
        self._prices, self._rankings, self._selection, self._risk = {}, {}, {}, {}
        self.coverage = {}

    def prices(self, event):
        if event not in self._prices:
            slug = event.replace(" ", "_")
            stocks = pd.read_parquet(self.source / f"prices/{slug}.parquet")
            benchmark = pd.read_parquet(self.source / f"prices/{slug}_GSPC.parquet").iloc[:, 0]
            for frame in [stocks, benchmark]:
                if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
                    raise ValueError("Price dates must be unique and increasing")
            if stocks.columns.has_duplicates or not stocks.index.equals(benchmark.index):
                raise ValueError("Price columns must be unique and benchmark calendar must align")
            metadata = json.loads((self.source / f"prices/{slug}.json").read_text())
            if metadata.get("auto_adjust") is not False:
                raise ValueError("Price cache must explicitly declare auto_adjust=False")
            self._prices[event] = stocks, benchmark
        return self._prices[event]

    def training_beta(self, event):
        tb = super().training_beta(event)
        stocks, bm = self.prices(event)
        start, end, extended = self.training_window(event)
        ev = EVENTS[event]
        endpoints = [pd.Timestamp(ev.peak), pd.Timestamp(ev.bottom)]
        if not np.isfinite(bm.reindex(endpoints)).all() or (bm.reindex(endpoints) <= 0).any():
            raise ValueError("Missing or invalid benchmark endpoints")
        train = stocks.reindex(columns=tb.index).loc[start:end]
        prices = stocks.reindex(index=endpoints, columns=tb.index)
        good = (np.isfinite(train).all() & train.gt(0).all()
                & np.isfinite(prices).all() & prices.gt(0).all())
        if train.empty:
            raise ValueError("No training rows")
        flags = pd.DataFrame(index=tb.index)
        flags.index.name = "Ticker"
        flags["complete_training_and_endpoint_prices"] = good
        flags["usable_current_analyst"] = flags.index.isin(self.forward.index)
        self.coverage[event] = flags
        # Historical-price eligibility is fixed before clustering. Missing analyst
        # returns are dropped after preselection, as in the guide, without top-up.
        return tb.loc[good]

    def forward_returns(self, event):
        return self.forward


def validate_rows(raw, model):
    expected = len(EVENTS) * np.prod([len(GRID[p]) for p in PARAMETERS])
    if len(raw) != expected or raw.duplicated(["event"] + PARAMETERS).any():
        raise AssertionError("Full sweep has duplicate or missing rows")
    for event in EVENTS:
        actual = set(map(tuple, raw.loc[raw.event == event, PARAMETERS].to_numpy()))
        desired = set(product(*(GRID[p] for p in PARAMETERS)))
        if actual != desired:
            raise AssertionError("Parameter grid mismatch")
    checks = []
    for _, row in raw.loc[raw.status.eq("ok")].iterrows():
        weights = pd.Series(json.loads(row.holdings), dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0).any() or abs(weights.sum() - 1) > 1e-10:
            raise AssertionError("Invalid solved portfolio weights")
        if len(weights) != row.n_final:
            raise AssertionError("Holding count mismatch")
        if not model.coverage[row.event].loc[weights.index, "complete_training_and_endpoint_prices"].all():
            raise AssertionError("Portfolio uses stock without required historical prices")
        if not weights.index.isin(model.forward.index).all():
            raise AssertionError("Portfolio uses stock without current analyst input")
        # Independent endpoint-return algebra, rather than calling evaluate()
        # again: fixed-share portfolio ratio = sum(w * P_bottom / P_peak).
        stocks, benchmark = model.prices(row.event)
        event = EVENTS[row.event]
        relative = stocks.loc[event.bottom, weights.index] / stocks.loc[event.peak, weights.index]
        portfolio_drop = abs(float(np.dot(weights.to_numpy(), relative.to_numpy())) - 1)
        market_drop = abs(float(benchmark.loc[event.bottom] / benchmark.loc[event.peak] - 1))
        independent_PR = 1 - portfolio_drop / market_drop
        difference = abs(independent_PR - row.PR)
        if difference > 1e-12:
            raise AssertionError("Independent portfolio reevaluation differs")
        checks.append(difference)
    return dict(expected_runs=int(expected), actual_runs=len(raw), unique_complete_grid=True,
                solved_portfolios_re_evaluated=len(checks),
                method="Independent weighted endpoint-return identity, not evaluate()",
                max_re_evaluation_PR_difference=max(checks, default=None),
                weights_and_data_eligibility_passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parent / "inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent / ".cache/dbi")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = SP500Experiment(args.inputs, args.cache)
    coverage_rows = []
    for event in EVENTS:
        model.training_beta(event)
        flags = model.coverage[event]
        flags.to_csv(args.output / f"{event.replace(' ', '_')}_coverage.csv")
        start, end, extended = model.training_window(event)
        coverage_rows.append(dict(event=event, five_beta_stocks=len(flags),
                price_eligible=int(flags.complete_training_and_endpoint_prices.sum()),
                price_and_analyst_eligible=int((flags.complete_training_and_endpoint_prices & flags.usable_current_analyst).sum()),
                train_start=str(start.date()), train_end=str(end.date()), window_extended=extended,
                t1=model.curve_params(event)[0], t2=model.curve_params(event)[1], dd=model.curve_params(event)[2]))
    pd.DataFrame(coverage_rows).to_csv(args.output / "coverage_summary.csv", index=False)
    grid = dict(K=[7], Seeds=[3], MinAppear=[3], w1=[.17]) if args.reference_only else GRID
    rows = []
    for event in EVENTS:
        for K in grid["K"]:
            for Seeds, MinAppear, w1 in product(grid["Seeds"], grid["MinAppear"], grid["w1"]):
                rows.append(model.run_one(event, K, Seeds, MinAppear, w1))
            pd.DataFrame(rows).to_csv(args.output / "raw_runs_partial.csv", index=False)
            print(f"{event}, K={K}: {len(rows)} completed", flush=True)
    raw = pd.DataFrame(rows)
    tables = write_outputs(raw, args.output)
    metadata = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(), experiment="SP500_current_analyst_snapshot",
                    historical_point_in_time=False, benchmark="^GSPC", grid=grid, environment=versions(),
                    analyst_snapshot_first=model.analysts.snapshot_retrieved_at_utc.min(),
                    analyst_snapshot_last=model.analysts.snapshot_retrieved_at_utc.max(),
                    analyst_targets=len(model.analysts), usable_current_analysts=len(model.forward),
                    beta_tickers=len(model.beta), runs=len(raw), solved=int(raw.status.eq("ok").sum()),
                    complete_combos=len(tables["all_combos"]),
                    crash_duration_source="SP100 guide MarketDD T1/T2 retained as shared timing convention",
                    crash_depth_source="Original kappa_beta_per_swan.csv, Universe=SP500 index_dd",
                    event_dates="Same fixed peak/bottom dates as REPLICATION_GUIDE.md",
                    missing_price_policy="Exclude incomplete historical training/endpoint prices before clustering",
                    missing_analyst_policy="Drop missing/invalid current forward returns after preselection, without top-up",
                    input_hashes={str(p.relative_to(args.inputs)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in args.inputs.rglob('*') if p.is_file()})
    (args.output / "run_info.json").write_text(json.dumps(metadata, indent=2))
    if not args.reference_only:
        checks = validate_rows(raw, model)
        (args.output / "validation.json").write_text(json.dumps(checks, indent=2))
        model.export_checkpoints(args.output / "reference_checkpoints.xlsx", K=7, Seeds=3, MinAppear=3, w1=.17)
    (args.output / "raw_runs_partial.csv").unlink()
    print(json.dumps({k:metadata[k] for k in ["runs", "solved", "complete_combos"]}))
    print(tables["top_mean"].head(3).to_string(index=False))
    print(tables["top_worst_case"].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
