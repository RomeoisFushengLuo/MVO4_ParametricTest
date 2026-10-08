"""Validate intermediates against the engine and results against the saved sweep.

The engine is an oracle ONLY in this validator; the production module is independent.
"""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from mvo_algo import EVENTS, PARAMETERS, Replication, summarize


def check_intermediates(model: Replication) -> pd.DataFrame:
    sys.path.insert(0, str(model.source))
    import mvo4_engine as oracle
    import config
    config.T1_METHOD = "average"  # guide defaults; original parity script uses weighted t1
    oracle.clear_memory()
    rows = []

    def check(event, stage, matches, maximum_difference=None):
        rows.append(dict(event=event, check=stage, match=bool(matches), max_abs_diff=maximum_difference))

    expected = {
        "Swan 8": (78, 15, "2017-08-07", "2018-09-28", 290, "2017-09-28", 74, 216, .531486, .199593),
        "Swan 9": (86, 20, "2018-12-30", "2020-02-14", 284, "2019-02-14", 72, 200, .491195, .315312),
        "Swan 10D": (92, 14, "2020-11-15", "2021-12-31", 284, "2020-12-31", 66, 188, .446094, .272298),
        "Swan 11A": (94, 13, "2023-12-30", "2025-02-14", 282, "2024-02-15", 79, 198, .994976, .198868),
    }
    for event, ev in EVENTS.items():
        result = model.run_one(event, detail=True)
        reference = oracle.run_one(event, 7, 3, 3, .17, detail=True)
        a, b = result["risk"], reference["risk"]
        values = expected[event]
        check(event, "guide universe and selected counts",
              len(model.training_beta(event)) == values[0] and len(result["preselected"]) == values[1])
        check(event, "guide dates, cutoff, days, t1 and t2",
              (str(a["start"].date()), str(a["end"].date()), a["total_days"],
               str(a["cutoff"].date()), a["t1"], a["t2"]) == values[2:8])
        check(event, "guide PR and benchmark drawdown",
              abs(result["PR"] - values[8]) < 1e-6 and abs(result["market_dd"] - values[9]) < 1e-6)
        ranking = model.dbi_ranking(event, 7)
        oracle_ranking = oracle.dbi_ranking(list(ev.train_swans), 7)
        difference = float(np.max(np.abs(np.array(ranking)[:, 1] - np.array(oracle_ranking)[:, 1])))
        check(event, "all 1000 ranked seeds and DBI scores",
              [s for s, _ in ranking] == [s for s, _ in oracle_ranking] and difference < 1e-10, difference)
        check(event, "all stock votes", result["votes"].equals(reference["votes"]))
        check(event, "selected stocks", result["preselected"].index.equals(reference["preselected"].index))
        for stage, left, right in [
            ("injection betas", result["preselected"].avg, reference["preselected"].avg),
            ("curve", a["curve"], b["curve"]),
            ("raw prices", a["raw_prices"], b["raw_prices"]),
            ("injected prices", a["injected_prices"], b["injected_prices"]),
            ("fall", a["fall"], b["fall"]), ("R_fwd", a["R_fwd"], b["R_fwd"]),
            ("E_2", a["E_2"], b["E_2"]), ("covariance", a["cov"], b["cov"]),
            ("final weights", result["weights"], reference["weights"]),
        ]:
            same_labels = True
            if isinstance(left, (pd.Series, pd.DataFrame)):
                same_labels = left.index.equals(right.index)
                if isinstance(left, pd.DataFrame):
                    same_labels = same_labels and left.columns.equals(right.columns)
            difference = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
            check(event, stage, same_labels and difference < 1e-10, difference)
        check(event, "curve shape t1/t2/dd", a["t1"] == b["t1"] and a["t2"] == b["t2"] and a["dd"] == b["dd"])
        check(event, "evaluation PR", abs(result["PR"] - reference["PR"]) < 1e-10,
              abs(result["PR"] - reference["PR"]))
    return pd.DataFrame(rows)


def compare_full(model: Replication, path: Path) -> dict:
    actual = pd.read_csv(path)
    reference = pd.read_csv(model.source / "results/signature_avg_13p5mo/raw_runs.csv")
    keys = ["event"] + PARAMETERS
    if actual.duplicated(keys).any() or reference.duplicated(keys).any():
        raise AssertionError("Duplicate parameter/event rows")
    joined = actual.merge(reference, on=keys, suffixes=("_new", "_ref"), how="outer", indicator=True,
                          validate="one_to_one")
    solved = joined.status_new.eq("ok") & joined.status_ref.eq("ok")
    difference = (joined.loc[solved, "PR_new"] - joined.loc[solved, "PR_ref"]).abs()
    portfolio_difference = (joined.loc[solved, "port_dd_new"] - joined.loc[solved, "port_dd_ref"]).abs()
    benchmark_difference = (joined.loc[solved, "market_dd"] - joined.loc[solved, "mkt_dd"]).abs()
    names_equal, weight_difference = True, 0.0
    for _, row in joined.loc[solved].iterrows():
        new, ref = json.loads(row.holdings_new), json.loads(row.holdings_ref)
        names_equal = names_equal and new.keys() == ref.keys()
        for ticker in new.keys() | ref.keys():
            weight_difference = max(weight_difference, abs(new.get(ticker, 0) - ref.get(ticker, 0)))
    same_counts = all(joined.loc[solved, f"{name}_new"].eq(joined.loc[solved, f"{name}_ref"]).all()
                      for name in ["n_preselect", "n_final"])
    tables = summarize(actual)
    best_worst = tables["top_worst_case"].iloc[0].to_dict()
    best_mean = tables["top_mean"].iloc[0].to_dict()
    report = dict(runs=len(actual), solved=int(actual.status.eq("ok").sum()),
                  complete_combos=len(tables["all_combos"]),
                  same_parameter_rows=bool(joined._merge.eq("both").all()),
                  same_statuses=bool(joined.status_new.eq(joined.status_ref).all()),
                  max_PR_difference=float(difference.max()), same_stock_counts=bool(same_counts),
                  max_portfolio_drawdown_difference=float(portfolio_difference.max()),
                  max_benchmark_drawdown_difference=float(benchmark_difference.max()),
                  same_training_starts=bool(joined.loc[solved, "train_start_new"].eq(joined.loc[solved, "train_start_ref"]).all()),
                  same_holdings=bool(names_equal), max_weight_difference_vs_rounded_reference=weight_difference,
                  best_worst_case=best_worst, best_mean=best_mean)
    report["passed"] = (len(actual) == 5376 and report["solved"] == 4130
                        and report["complete_combos"] == 871 and report["same_parameter_rows"]
                        and report["same_statuses"] and report["max_PR_difference"] < 1e-10
                        and report["max_portfolio_drawdown_difference"] < 1e-10
                        and report["max_benchmark_drawdown_difference"] < 1e-10
                        and report["same_training_starts"]
                        and report["same_stock_counts"] and report["same_holdings"]
                        and weight_difference <= 5.01e-7)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent / ".cache/dbi")
    parser.add_argument("--full-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = Replication(args.source, args.cache)
    checks = check_intermediates(model)
    checks.to_csv(args.output / "intermediate_checks.csv", index=False)
    print(checks.to_string(index=False))
    passed = bool(checks.match.all())
    if args.full_results:
        full = compare_full(model, args.full_results)
        (args.output / "full_comparison.json").write_text(json.dumps(full, indent=2))
        print(json.dumps(full, indent=2))
        passed = passed and full["passed"]
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
