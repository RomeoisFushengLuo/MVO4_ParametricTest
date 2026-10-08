# Replicating the MVO4 sweep from scratch

The goal: understand what the sweep does, write **your own** implementation from this description, and get the same results. Try to build it from this guide and the data files. Only open `mvo4_engine.py` once you're stuck or comparing results.

Every number you need to match is listed at the end of each stage, and `replication_checkpoints.xlsx` holds the full intermediate values for one reference combination (K=7, Seeds=3, MinAppear=3, w1=0.17). Check each stage before moving to the next. Errors compound, and matching only the final PR tells you nothing about where you went wrong.

---

## 0. The big picture

For each crash event (Swan 8, Swan 9, Swan 10D, Swan 11A) and each parameter combination:

1. **Preselect** a basket of low-beta stocks by voting across many K-means clusterings of past-crash betas.
2. **Inject** a synthetic crash into those stocks' pre-event prices, scaled by each stock's beta, so the risk statistics "see" a crash.
3. **Compute risk inputs** from the injected prices: max drawdown, weekly covariance, and an expected return built from analyst targets.
4. **Optimize** with max Sharpe (rf = 0) under a per-stock cap w1.
5. **Evaluate** the basket on the real crash: Protection Ratio, PR = 1 − (portfolio drop / ^OEX drop), from peak to bottom.

The grid is K ∈ {5…12}, Seeds ∈ {3,5,7,9,11,13}, MinAppear ∈ {2,3,4,5}, w1 ∈ {0.05,0.07,…,0.17}. That's 1,344 combinations × 4 events = 5,376 runs.

---

## 1. Inputs (all in `data/`)

| File | What it is |
|---|---|
| `All_SwanBeta100_MVO2.xlsx` | Beta of each S&P 100 stock in each past crash, columns `Swan1`…`Swan9`. |
| `kappa_beta_per_swan.csv` | Adds the `Swan10D` and `Swan11A` beta columns: rows with `Swan` = `S10D` / `S11A` and `Universe` = `SP100`. |
| `MarketDD.xlsx` | One row per swan: `T1` (crash length, trading days), `T2` (recovery length), `Actual DD` (market drawdown). |
| `optdata/*.xlsx` | Column `R_fwd_30_MA` (forward-return estimate from analyst targets) per ticker, one file per event. |

Prices are Yahoo daily **`Close` with `auto_adjust=False`** (not adjusted close) for the stocks and the benchmark `^OEX`. The cached copies are in `cache/prices/*.parquet` if you want to remove download differences as a variable. Do that at first.

### Events

| Event | Training swans (beta columns) | Peak | Bottom | optdata file |
|---|---|---|---|---|
| Swan 8 | Swan3–Swan7 | 2018-10-03 | 2018-12-24 | `Swan8_sp100_optdata4.xlsx` |
| Swan 9 | Swan4–Swan8 | 2020-02-19 | 2020-03-23 | `revised_swan9_sp100_optdata4.xlsx` |
| Swan 10D | Swan5–Swan9 | 2022-01-03 | 2022-10-12 | `revised_swan10d_sp100_optdata4.xlsx` |
| Swan 11A | Swan6, 7, 8, 9, 10D | 2025-02-19 | 2025-04-08 | `revised_swan11a_sp100_optdata4.xlsx` |

Chronological swan order is Swan1, Swan2, …, Swan9, Swan10D, Swan11A.

---

## 2. Crash-curve shape (t1, t2, dd)

Average over **every** swan from Swan1 up to and including the event's last training swan, using `MarketDD.xlsx`. For Swan 8 that's Swan1–Swan7.

- t1 = round(mean T1)
- t2 = round(mean T2)
- dd = mean Actual DD (not rounded)

| | t1 | t2 | dd |
|---|---|---|---|
| Swan 8 | 74 | 216 | 0.253876 |
| Swan 9 | 72 | 200 | 0.247090 |
| Swan 10D | 66 | 188 | 0.254670 |
| Swan 11A | 79 | 198 | 0.256433 |

---

## 3. Training window

- **End** = the Friday before the Monday of the peak's week, i.e. peak − (weekday + 3) days with Monday = 0.
- **Start** = end − 13 months − 15 days.
- If the window has fewer than t1 + t2 trading days, move the start back to the trading day that gives **exactly** t1 + t2 trading days. Only Swan 8 needs this.
- Both ends are inclusive.

✅ **Check:**

| | Start | End | Trading days |
|---|---|---|---|
| Swan 8 | 2017-08-07 | 2018-09-28 | 290 (extended) |
| Swan 9 | 2018-12-30 | 2020-02-14 | 284 |
| Swan 10D | 2020-11-15 | 2021-12-31 | 284 |
| Swan 11A | 2023-12-30 | 2025-02-14 | 282 |

---

## 4. Preselection (depends on K, Seeds, MinAppear)

**4a. Training beta matrix.** Take the 5 training-swan beta columns. Keep stocks that have a beta in all 5. This is a matrix of stocks × 5, used unscaled.

✅ Stock counts: Swan 8 78, Swan 9 86, Swan 10D 92, Swan 11A 94.

**4b. Rank seeds by Davies-Bouldin.** For random_state = 1…1000, fit `sklearn.cluster.KMeans(n_clusters=K, random_state=seed, n_init=10)` and compute `davies_bouldin_score` on the labels. Sort seeds by score, lowest first.

**4c. Vote.** For each of the top `Seeds` seeds:
1. Refit K-means with that seed.
2. Give each cluster a score: the mean of its centroid's 5 coordinates, with negative scores set to 0.
3. Take the cluster with the lowest score. If it has fewer than 10 stocks, also take the next-lowest cluster.
4. Each stock in the taken cluster(s) gets one vote.

**4d. Keep.** Sort stocks by vote count (descending). Take the top 22, then keep those with **at least MinAppear** votes. If fewer than 5 survive, the run fails as `no_preselect`. MinAppear > Seeds always fails this way.

**4e. Injection beta** for each kept stock = the mean of its 5 training-swan betas.

✅ Check against the "Top DBI seeds" and "Votes and stock stats" sheets. At the reference combo the preselected counts are Swan 8 15, Swan 9 20, Swan 10D 14, Swan 11A 13.

⚠️ **Use scikit-learn 1.7.2 (1.8.0 gives the same result).** Seed scores sit at near-ties. For Swan 9 at K=7, three seeds tie at DBI 1.062034. scikit-learn 1.2.2 ranks them differently and produces 15 stocks instead of 20. If your preselection differs, check the version before anything else. Don't loosen the MinAppear rule to compensate.

---

## 5. Crash injection

Let N = number of trading days in the window. Build a curve of length N:

- **Ramp, t1 days:** values 0, inc, 2·inc, … with inc = dd / (t1 − 1). Each value is rounded to 4 decimals.
- **Recovery, t2 days:** for i = 0…t2−1, dd − log_base(i + 2) / 100, with base = (t2 + 1)^(1 / (dd·100)). Each value is rounded to 4 decimals.
- **Zeros** for the remaining N − t1 − t2 days.

The curve starts on the **first day** of the window: the crash sits at the front, and the most recent days are untouched.

Injected price = real price × (1 − injection_beta × curve(day)). Fill any missing price with the geometric mean of its neighbours (√(prev × next)). At the first or last row, use prev² / prev-prev instead.

✅ Check against the "Crash curves" sheet. Swan 8 day 73 = 0.2539, the top of the ramp.

---

## 6. Risk inputs (computed on injected prices)

- **One-year cutoff** = the last trading day on or before train_end − 365 days. ✅ Swan 8 2017-09-28, Swan 9 2019-02-14, Swan 10D 2020-12-31, Swan 11A 2024-02-15.
- **fall** (per stock) = max drawdown of daily returns from the cutoff onward. Compute daily pct_change over the whole window first, then slice from the cutoff date, inclusive.
- **Weekly returns:**
  1. Label each day with its `%W` week number (Monday-start). Use `%W` literally: a week that spans New Year gets split in two (week 52 and week 00), and the original code does this too.
  2. Take the last trading day of each week. A week counts only when the next week starts inside the window, so the final week isn't included.
  3. Compute pct_change between those days and drop the first NaN.
  4. Keep the weeks dated on or after the cutoff.
- **Covariance** = sample covariance (n − 1) of those weekly simple returns × **51**.
- **E_2** = R_fwd_30_MA / fall. Drop stocks with missing or infinite values.

✅ Check fall, R_fwd_30_MA and E_2 per stock in "Votes and stock stats", and the covariance matrices in the "Cov …" sheets.

---

## 7. Optimization (depends on w1)

PyPortfolioOpt 1.6.0, default solver:

```
ef = EfficientFrontier(E_2 / 100, cov, weight_bounds=(0, w1))
ef.max_sharpe(risk_free_rate=0.0)
w = ef.clean_weights()
```

Drop weights below 0.03 and renormalize the rest to sum to 1. Final weights can therefore end slightly above w1, by up to about 1 point; that's expected.

**rf = 0.0 only:** no retry at another rf and no min-volatility fallback. If the solver fails, the run is `opt_failed`.

✅ Check final_weight in "Votes and stock stats". For example, Swan 9 = AMZN, COST, MDLZ, MRK, WMT at 17.31%, LLY 7.07%, MO 6.38%.

---

## 8. Evaluation (real prices, not injected)

1. Buy at the close on the **peak** date: shares = 10,000 × w / price(peak). Forward-fill any missing prices.
2. Portfolio drop = |PV(bottom) / PV(peak) − 1|.
3. ^OEX drop = the same, over the same dates.
4. **PR** = 1 − portfolio drop / ^OEX drop.

✅ At the reference combo:

| | Swan 8 | Swan 9 | Swan 10D | Swan 11A |
|---|---|---|---|---|
| PR | 0.531486 | 0.491195 | 0.446094 | 0.994976 |
| ^OEX drop | 0.199593 | 0.315312 | 0.272298 | 0.198868 |

---

## 9. The full sweep

Run every combination on all four events, then summarize the combos that solved on all four by mean PR and by min (worst-case) PR.

✅ **Totals:** 5,376 runs, 4,130 solved and 871 complete combos.
- **Best worst case:** K=7 / Seeds=3 / MinAppear=3 / w1=0.17, with min 44.6% (Swan 10D) and mean 61.6%. 13 combos tie at a 44.6% min; this one has the highest mean.
- **Best mean:** K=12 / Seeds=3 / MinAppear=3 / w1=0.11, with mean 65.1% and min 33.9% (Swan 9).

Compare your per-run results with `results/signature_avg_13p5mo/raw_runs.csv` (also the "Raw runs" sheet of `summary.xlsx`).

Speed tip: the DBI ranking (4b) depends only on the event and K, so compute it once and reuse it. Steps 5–6 don't depend on w1.

---

## Common mistakes (all seen in practice)

- **Wrong scikit-learn version.** It changes Swan 9's preselection (section 4).
- **Treating MinAppear as "MinAppear − 1".** The rule is ≥ MinAppear.
- **Adjusted Close instead of Close.**
- **Different dates.** Buying the day after the peak, ending the window on a different day, or measuring the one-year cutoff from another date. Swan 8 is especially sensitive: moving its window start by single days moves PR between 47% and 66%.
- **Crash at the wrong end.** Placing it at the end of the window, or cutting the curve short instead of extending the window.
- **Off-by-one in the window.** Many download APIs (including `yf.download`) exclude the end date.

## Caveat

Swan 9's `R_fwd_30_MA` may not be point-in-time: it predicts the crash returns better than data from 2020-02-14 should. Treat Swan 9 results as provisional. This doesn't affect replication, since you're matching the same inputs.

## When you're done

`parity_check.py` compares this folder's engine with the original `mvo4.ipynb` code step by step. You can point the same idea at your own implementation: run one combo and diff every intermediate against `replication_checkpoints.xlsx`. To make checkpoints for another combo, run `python make_checkpoints.py --K 12 --Seeds 3 --MinAppear 3 --w1 0.11`.
