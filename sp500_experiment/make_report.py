"""Build an interpretation report and exported chart from the completed SP500 sweep."""
from pathlib import Path
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mvo4-sp500-matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "results/full"
    info = json.loads((output / "run_info.json").read_text())
    validation = json.loads((output / "validation.json").read_text())
    raw = pd.read_csv(output / "raw_runs.csv")
    summary = pd.read_excel(output / "summary.xlsx", sheet_name=None)
    mean = summary["top_mean"].iloc[0]
    worst = summary["top_worst_case"].iloc[0]
    coverage = pd.read_csv(output / "coverage_summary.csv")
    ref = pd.read_csv(HERE / "results/reference/raw_runs.csv")
    old = pd.read_csv(HERE.parent / "replication/validation/reference/raw_runs.csv")
    events = coverage.event.tolist()
    analyst_dates = [pd.Timestamp(info[k]).tz_convert("Asia/Shanghai").isoformat()
                     for k in ["analyst_snapshot_first", "analyst_snapshot_last"]]

    def combo(row):
        return f"K={int(row.K)}, Seeds={int(row.Seeds)}, MinAppear={int(row.MinAppear)}, w1={row.w1:.2f}"

    lines = ["# SP500 parametric test results", "",
        "This is a historical experiment using the uploaded Yahoo price caches and CURRENT analyst return estimates. It is not a point-in-time historical backtest.", "",
        f"Completed the same guide grid: **{info['runs']:,} runs**, **{info['solved']:,} solved**, and **{info['complete_combos']:,} parameter combinations solved on all four events**.", "",
        f"Best mean PR: **{combo(mean)}**, mean **{mean.mean_PR:.2%}**, minimum **{mean.min_PR:.2%}**.", "",
        f"Best worst-case PR: **{combo(worst)}**, minimum **{worst.min_PR:.2%}**, mean **{worst.mean_PR:.2%}**.", "",
        "| Event | Best-mean PR | Best-worst-case PR |", "|---|---:|---:|"]
    lines += [f"| {event} | {mean[event]:.2%} | {worst[event]:.2%} |" for event in events]
    lines += ["", "## Data coverage", "",
        f"Uploaded prices have 518 ticker columns per event. The historical beta source has {info['beta_tickers']} tickers; stocks without all five training betas are ineligible. Of {info['analyst_targets']} current analyst records, {info['usable_current_analysts']} have usable forward returns.", "",
        "| Event | Five historical betas | Complete historical prices | Also usable current analyst input |", "|---|---:|---:|---:|"]
    lines += [f"| {r.event} | {r.five_beta_stocks} | {r.price_eligible} | {r.price_and_analyst_eligible} |" for r in coverage.itertuples()]
    lines += ["", "Stocks with incomplete training or peak/bottom prices are excluded before clustering. Missing analyst returns are removed after voting, with no replacement stocks. No missing histories were invented or filled from another ticker.", "",
        "## Method", "",
        "- Same K=5–12; Seeds=3,5,7,9,11,13; MinAppear=2,3,4,5; w1=0.05–0.17 by 0.02; 1,344 combinations per event.",
        "- Same 1,000-seed K-means ranking, n_init=10, top-22 vote rule, crash-at-front injection, last-year risk statistics, weekly sample covariance ×51, max Sharpe at rf=0, 3% weight floor, and peak-to-bottom evaluation.",
        "- Uses SP500 beta rows throughout; preserves the existing beta universe rather than filling new constituents' missing betas.",
        "- Evaluation benchmark is ^GSPC. The beta source's SP500 index drawdowns match the uploaded ^GSPC event returns for Swan9, Swan10D, and Swan11A. SPY is retained as an input but is not the primary benchmark.",
        "- Crash depths are plain averages of SP500 index_dd values over all prior swans. T1/T2 durations are retained from the guide workbook as a shared timing convention; SP500-specific recovery-duration data was not supplied.",
        "- Event peak/bottom and nominal training-window rules remain those in REPLICATION_GUIDE.md. Swan8 is extended to 290 trading days so its complete curve fits.",
        "- Uses the same current R_fwd_30_MA estimates for all four events, as authorized. These were supplied as current target / current 30-day MA - 1, not target / historical event price - 1.", "",
        f"Analyst retrieval range (Asia/Shanghai): {analyst_dates[0]} through {analyst_dates[1]}.", "",
        "## Reference combination and earlier SP100 results", "",
        "K=7, Seeds=3, MinAppear=3, w1=0.17:", "",
        "| Event | New SP500 PR | Earlier SP100 PR |", "|---|---:|---:|"]
    for event in events:
        a = ref.loc[ref.event.eq(event), "PR"].iloc[0]
        b = old.loc[old.event.eq(event), "PR"].iloc[0]
        lines.append(f"| {event} | {a:.2%} | {b:.2%} |")
    lines += ["", "This is contextual, not a controlled universe comparison: the new experiment also changes analyst inputs, beta definitions, benchmark, crash depths, price snapshot, and eligibility. Differences cannot be attributed solely to moving from SP100 to SP500.", "",
        "## Validation", "",
        "- All nine existing shared-pipeline regression tests passed after the analyst-input hook refactor.",
        "- All three SP500 integration tests passed: current-return arithmetic/flags, price eligibility/benchmark alignment, and crash depth/timing.",
        "- The original SP100 Swan9 reference PR remained 0.49119493879261333 after the shared analyst-input hook refactor; see shared_hook_check.json.",
        f"- All {validation['actual_runs']:,} event/parameter rows are unique and the grid is complete.",
        f"- Re-evaluated all {validation['solved_portfolios_re_evaluated']:,} solved portfolios against the uploaded historical prices. Maximum PR difference: {validation['max_re_evaluation_PR_difference']:.3g}.",
        "- Solved portfolios have finite positive weights summing to one, consistent holding counts, complete required prices, and usable analyst inputs.",
        "- Seed rankings are freshly computed on each event's SP500 eligible universe with scikit-learn 1.7.2; 32 event/K rankings of 1,000 seeds each.",
        "- Checks do not prove that the upstream Yahoo data or analyst targets are economically correct. Full original per-symbol histories were not supplied; the 30-day averages are validated arithmetically, not independently reconstructed.", "",
        "## Limitations", "",
        "1. Current analyst targets and the current-source constituent universe introduce look-ahead and survivorship effects. This experiment does not establish historical predictability or out-of-sample performance.",
        "2. Complete-case screening uses availability at the historical evaluation endpoint. This is an explicit data-availability rule and can also create survivorship bias.",
        "3. Swan8 uses the guide's 2018-10-03 peak. The beta CSV's SP500 drawdown is 0.197782, versus 0.196345 for ^GSPC over the guide's fixed dates. The uploaded ^GSPC series has its nearby maximum on 2018-09-20. No event dates or supplied betas were silently rewritten to force agreement.",
        "4. T1/T2 are the shared guide durations, not newly measured SP500 durations. A stricter SP500 signature study needs independently verified SP500 event and recovery dates.",
        "5. No external parity oracle or fixed expected SP500 result exists; verification checks the implementation, input consistency, and resulting portfolios. It is not a claim of equivalence with a previously published SP500 strategy.", "",
        "## Files", "",
        "- results/full/raw_runs.csv: every status and solved portfolio.",
        "- results/full/summary.xlsx: complete combinations, top rankings, marginal results, statuses, and all raw rows.",
        "- results/full/reference_checkpoints.xlsx: reference-combination votes, rankings, curves, risk inputs, and weights.",
        "- results/full/coverage_summary.csv and per-event coverage CSVs: all availability exclusions.",
        "- results/full/run_info.json: package versions, choices, analyst timestamps, and input hashes.",
        "- results/full/validation.json: full-grid and portfolio checks.",
        "- input_audit.csv: cached-date coverage and benchmark/depth differences.",
        "- inputs/: frozen copies of all uploaded data, prices, and reports, plus the explicitly prepared SP500 crash-shape table."]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    # Publication/export chart; it is descriptive of this in-sample experiment.
    combos = summary["all_combos"]
    heat = combos.groupby(["K", "w1"]).min_PR.max().unstack().reindex(index=range(5, 13), columns=[.05,.07,.09,.11,.13,.15,.17])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    x = np.arange(len(events));width = .36
    axes[0].bar(x-width/2, [mean[e]*100 for e in events], width, label="Best mean combination")
    axes[0].bar(x+width/2, [worst[e]*100 for e in events], width, label="Best worst-case combination")
    axes[0].set_xticks(x, events);axes[0].set_ylabel("PR (%)");axes[0].axhline(0,color="black",linewidth=.6)
    axes[0].legend(fontsize=8);axes[0].set_title("Event outcomes with current analyst inputs")
    image = axes[1].imshow(heat.to_numpy()*100, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_xticks(range(7),[f"{v:.2f}" for v in heat.columns]);axes[1].set_yticks(range(8),list(heat.index))
    axes[1].set_xlabel("Weight cap w1");axes[1].set_ylabel("K")
    axes[1].set_title("Best minimum PR across Seeds / MinAppear")
    fig.colorbar(image,ax=axes[1],label="Minimum event PR (%)")
    fig.savefig(HERE / "results/full/parameter_summary.png",dpi=180)
    print("Wrote REPORT.md and parameter_summary.png")


if __name__ == "__main__":
    main()
