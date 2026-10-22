# SP500 parametric test results

This is a historical experiment using the uploaded Yahoo price caches and CURRENT analyst return estimates. It is not a point-in-time historical backtest.

Completed the same guide grid: **5,376 runs**, **4,816 solved**, and **1,161 parameter combinations solved on all four events**.

Best mean PR: **K=12, Seeds=5, MinAppear=3, w1=0.09**, mean **70.07%**, minimum **11.04%**.

Best worst-case PR: **K=7, Seeds=9, MinAppear=2, w1=0.17**, minimum **17.69%**, mean **55.10%**.

| Event | Best-mean PR | Best-worst-case PR |
|---|---:|---:|
| Swan 8 | 99.74% | 69.49% |
| Swan 9 | 11.04% | 17.69% |
| Swan 10D | 76.08% | 45.98% |
| Swan 11A | 93.42% | 87.23% |

## Data coverage

Uploaded prices have 518 ticker columns per event. The historical beta source has 502 tickers; stocks without all five training betas are ineligible. Of 514 current analyst records, 508 have usable forward returns.

| Event | Five historical betas | Complete historical prices | Also usable current analyst input |
|---|---:|---:|---:|
| Swan 8 | 370 | 367 | 365 |
| Swan 9 | 410 | 405 | 403 |
| Swan 10D | 417 | 412 | 410 |
| Swan 11A | 424 | 419 | 417 |

Stocks with incomplete training or peak/bottom prices are excluded before clustering. Missing analyst returns are removed after voting, with no replacement stocks. No missing histories were invented or filled from another ticker.

## Method

- Same K=5–12; Seeds=3,5,7,9,11,13; MinAppear=2,3,4,5; w1=0.05–0.17 by 0.02; 1,344 combinations per event.
- Same 1,000-seed K-means ranking, n_init=10, top-22 vote rule, crash-at-front injection, last-year risk statistics, weekly sample covariance ×51, max Sharpe at rf=0, 3% weight floor, and peak-to-bottom evaluation.
- Uses SP500 beta rows throughout; preserves the existing beta universe rather than filling new constituents' missing betas.
- Evaluation benchmark is ^GSPC. The beta source's SP500 index drawdowns match the uploaded ^GSPC event returns for Swan9, Swan10D, and Swan11A. SPY is retained as an input but is not the primary benchmark.
- Crash depths are plain averages of SP500 index_dd values over all prior swans. T1/T2 durations are retained from the guide workbook as a shared timing convention; SP500-specific recovery-duration data was not supplied.
- Event peak/bottom and nominal training-window rules remain those in REPLICATION_GUIDE.md. Swan8 is extended to 290 trading days so its complete curve fits.
- Uses the same current R_fwd_30_MA estimates for all four events, as authorized. These were supplied as current target / current 30-day MA - 1, not target / historical event price - 1.

Analyst retrieval range (Asia/Shanghai): 2026-10-20T06:23:09.647790+08:00 through 2026-10-20T06:37:50.283047+08:00.

## Reference combination and earlier SP100 results

K=7, Seeds=3, MinAppear=3, w1=0.17:

| Event | New SP500 PR | Earlier SP100 PR |
|---|---:|---:|
| Swan 8 | 69.49% | 53.15% |
| Swan 9 | 8.40% | 49.12% |
| Swan 10D | 45.98% | 44.61% |
| Swan 11A | 87.23% | 99.50% |

This is contextual, not a controlled universe comparison: the new experiment also changes analyst inputs, beta definitions, benchmark, crash depths, price snapshot, and eligibility. Differences cannot be attributed solely to moving from SP100 to SP500.

## Validation

- All nine existing shared-pipeline regression tests passed after the analyst-input hook refactor.
- All three SP500 integration tests passed: current-return arithmetic/flags, price eligibility/benchmark alignment, and crash depth/timing.
- The original SP100 Swan9 reference PR remained 0.49119493879261333 after the shared analyst-input hook refactor; see shared_hook_check.json.
- All 5,376 event/parameter rows are unique and the grid is complete.
- Re-evaluated all 4,816 solved portfolios against the uploaded historical prices. Maximum PR difference: 3.55e-15.
- Solved portfolios have finite positive weights summing to one, consistent holding counts, complete required prices, and usable analyst inputs.
- Seed rankings are freshly computed on each event's SP500 eligible universe with scikit-learn 1.7.2; 32 event/K rankings of 1,000 seeds each.
- Checks do not prove that the upstream Yahoo data or analyst targets are economically correct. Full original per-symbol histories were not supplied; the 30-day averages are validated arithmetically, not independently reconstructed.

## Limitations

1. Current analyst targets and the current-source constituent universe introduce look-ahead and survivorship effects. This experiment does not establish historical predictability or out-of-sample performance.
2. Complete-case screening uses availability at the historical evaluation endpoint. This is an explicit data-availability rule and can also create survivorship bias.
3. Swan8 uses the guide's 2018-10-03 peak. The beta CSV's SP500 drawdown is 0.197782, versus 0.196345 for ^GSPC over the guide's fixed dates. The uploaded ^GSPC series has its nearby maximum on 2018-09-20. No event dates or supplied betas were silently rewritten to force agreement.
4. T1/T2 are the shared guide durations, not newly measured SP500 durations. A stricter SP500 signature study needs independently verified SP500 event and recovery dates.
5. No external parity oracle or fixed expected SP500 result exists; verification checks the implementation, input consistency, and resulting portfolios. It is not a claim of equivalence with a previously published SP500 strategy.

## Files

- results/full/raw_runs.csv: every status and solved portfolio.
- results/full/summary.xlsx: complete combinations, top rankings, marginal results, statuses, and all raw rows.
- results/full/reference_checkpoints.xlsx: reference-combination votes, rankings, curves, risk inputs, and weights.
- results/full/coverage_summary.csv and per-event coverage CSVs: all availability exclusions.
- results/full/run_info.json: package versions, choices, analyst timestamps, and input hashes.
- results/full/validation.json: full-grid and portfolio checks.
- input_audit.csv: cached-date coverage and benchmark/depth differences.
- inputs/: frozen copies of all uploaded data, prices, and reports, plus the explicitly prepared SP500 crash-shape table.
