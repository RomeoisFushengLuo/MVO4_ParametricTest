# Implementation against REPLICATION_GUIDE.md

The guide is the controlling specification. The uploaded `mvo_algo.py` and three
notebooks are evidence of the earlier implementation, not additional requirements.
The uploaded DOCX contains only the placeholder text “文档标题” and “这是正文内容。”
No substantive pipeline requirements could be recovered from it.

## Design decisions

1. Use one shared implementation for every event. Event-specific copies had drifted
   in dates, training swans, stock limits, and optimizer settings. Frozen event
   definitions now encode exactly the four events in the guide.
2. Implement the stages independently from the guide and data. The existing engine
   is imported only by the comparison script, after implementation, as a validation
   oracle. The pipeline does not delegate computation to it.
3. Start with the shipped unadjusted Close caches. This fixes historical revisions
   and delisted-ticker availability as variables, and avoids accidental changes in
   Yahoo's adjustment defaults.
4. Validate intermediate stages before trusting final PR. The guide's original
   checkpoint workbook is missing, so validation combines its explicit published
   values with direct intermediate comparisons against the existing engine.
5. Cache only reusable work. Seed rankings depend on the beta matrix and K;
   preselection depends on K/Seeds/MinAppear; risk inputs depend on the resulting
   event/basket. None depend on w1. This makes the full grid practical without
   changing its computations.
6. Keep expected model failures distinct from input defects. Infeasible optimization
   yields `opt_failed`, with no alternate risk-free rate or objective. Unexpected
   missing/broken inputs stop with `DataError`, rather than being mislabeled as a
   valid failed model.

## Differences from the uploads

| Area | Observed behavior in uploaded files | Revised behavior |
|---|---|---|
| Seed scoring | `dbscores` excludes the final seed; `kmeans_db` repeatedly fits `random_number` instead of its loop seed | Score seeds 1–1000 inclusive using the actual seed, `n_init=10` |
| Voting | Module fixes five seeds and retains the top 22 without applying MinAppear | Sweep Seeds and MinAppear; top 22 first, then votes ≥ MinAppear; ticker order resolves voting ties |
| Event configuration | The notebook named Swan9 is configured for Swan8; Swan11 trains through Swan11A and selects up to 30 stocks | Correct event names, five historical swans, top-22 limit, no target-swan beta in preselection |
| Crash parameters | Weighted t1 and selected-range averages; Swan10 notebook includes the predicted Swan10D in its t2/dd range | Plain averages over Swan1 through the last historical training swan, excluding the event |
| Window | Handwritten dates, download-exclusive end dates, and possible curves longer than the window | Friday before peak week; subtract 13 months and 15 days; inclusive endpoints; extend only enough to fit the curve |
| Injection | Module's ramp/recovery are unrounded; notebook-specific overrides and an EXC removal appear in Swan9 | Four-decimal curve at the front; mean historical injection beta; no hard-coded ticker removal |
| Expected returns | Module recomputes/adjusts forward returns from target prices | Read the supplied `R_fwd_30_MA` column unchanged, then divide by fall |
| Covariance | Module uses monthly returns ×12 | Literal `%W` weekly boundaries, omit unfinished final week, sample covariance ×51 |
| Optimization | Module tries multiple risk-free rates; notebooks contain rf=0.01 and alternative PR objectives | One `max_sharpe(risk_free_rate=0.0)` call; no fallback; discard weights <3%, renormalize |
| Evaluation | Event-specific purchase dates and evaluation ranges | Fixed shares bought at peak close, evaluated at bottom close against ^OEX |
| Portability | Absolute `/Users/...` paths and extra experimental dependencies | Repository-relative data paths or explicit `--source`; shared pinned dependencies |
| Sweep/reporting | Experimental single-event paths | Complete guide grid, all statuses, per-run results, mean/worst-case rankings, marginal tables, metadata, checkpoint export |

The replacement module intentionally has a new API. It does not preserve legacy
Holt/GARCH/quandl features or alternate PR optimization methods. The revised
notebooks use the new API; old notebooks must not be paired with it unchanged.
Original uploads and the repository's existing engine/configuration remain intact.

## Reference-combination results

K=7, Seeds=3, MinAppear=3, w1=0.17:

| Event | Eligible / selected / final stocks | Window, inclusive | Days | t1 / t2 | PR |
|---|---|---|---|---|---|
| Swan 8 | 78 / 15 / 7 | 2017-08-07–2018-09-28 | 290 | 74 / 216 | 0.531486 |
| Swan 9 | 86 / 20 / 7 | 2018-12-30–2020-02-14 | 284 | 72 / 200 | 0.491195 |
| Swan 10D | 92 / 14 / 7 | 2020-11-15–2021-12-31 | 284 | 66 / 188 | 0.446094 |
| Swan 11A | 94 / 13 / 7 | 2023-12-30–2025-02-14 | 282 | 79 / 198 | 0.994976 |

All four PRs and benchmark drawdowns match the guide within its six-decimal
precision. All 68 validation checks pass. They include all 1,000 ranked seeds per
event, voting counts, injection betas, raw/injected prices, curves, fall, forward
returns, E_2, covariance, final weights, window/cutoff dates, and PR. The maximum
absolute numerical difference from the repository engine is 2.775557561562892e-17.

Nine regression tests pass. All four revised notebooks were executed, with all
six code cells in each completing successfully. The standalone checkpoint command
was executed and its workbook verified: 26 sheets and a four-event overview.

## Full-grid verification

The full grid was executed independently, including fresh seed-ranking computation
for all 32 event/K combinations. Comparison to the shipped
`mvo4_sweep/results/signature_avg_13p5mo/raw_runs.csv` passed:

| Check | Result |
|---|---|
| Total runs | 5,376, as specified |
| Solved runs | 4,130, as specified |
| Combinations solved on all four events | 871, as specified |
| Event/parameter membership and statuses | All rows match; no duplicate or missing rows |
| Largest PR difference over solved rows | 2.7755575615628914e-15 |
| Largest portfolio drawdown difference | 6.002143226879753e-16 |
| Benchmark drawdowns | Identical |
| Preselected/final stock counts on solved rows | Identical |
| Training starts on solved rows | Identical |
| Held ticker sets on solved rows | Identical |
| Largest weight difference from six-decimal saved weights | 4.999850001519146e-7, within rounding precision |
| Best worst-case combination | K=7, Seeds=3, MinAppear=3, w1=0.17; min PR=0.446094, mean PR=0.615938 |
| Best mean combination | K=12, Seeds=3, MinAppear=3, w1=0.11; mean PR=0.651317, min PR=0.338697 |

Machine-readable evidence is in `validation/intermediate_checks.csv` and
`validation/full_comparison.json`. Full raw results, summary workbook, and package
version metadata are in `validation/full/`. Reference results and the new checkpoint
workbook are in `validation/reference/`. Executed notebook copies are in
`validation/notebooks/`. The full result is numerically equivalent, not a claim
that CSV files or all floating-point calculations are bit-for-bit identical.

Guide sections 1–9 are implemented and their stated reference values and full-grid
targets have been reproduced. This establishes replication with the supplied data,
subject to the limitations below.

## What remains limited or unverified

- `replication_checkpoints.xlsx` and `make_checkpoints.py` referenced by the original
  guide are absent from the checkout. A new exporter and independently calculated
  checkpoint workbook are supplied, but comparison to the missing original workbook
  cannot be claimed. Reference-combination intermediate comparison used the engine.
- Intermediate parity is checked for the reference combination on all events.
  Full-grid comparison checks output rows; it does not compare every intermediate
  matrix for all 5,376 combinations.
- Live Yahoo downloads, new events, and revised analyst inputs are not implemented
  or validated. The deliverable is for exact replication using the bundled data.
- The guide and repository README disagree about scikit-learn 1.8.0 equivalence.
  Use 1.7.2, the version actually used and validated here.
- The guide's first-row missing-price instruction uses “prev” where no previous
  row exists. The implementation uses next²/next-next there and prev²/prev-prev at
  the last row. Isolated gaps are tested; ambiguous consecutive gaps raise a clear
  error. The current reference training data does not exercise missing-price filling.
- Swan9's possible non-point-in-time analyst data remains a data caveat. This work
  reproduces the supplied inputs; it does not establish that historical forecasts
  were available before the event or that the strategy predicts future crashes.
- Dependencies were validated on Python 3.12.14; other Python/platform combinations
  have not been tested. The shipped price caches remain unchanged.
