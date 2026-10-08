# MVO4 parameter sweep

Everything needed to run the MVO4 black-swan preselection sweep: the code, the input data, a price and clustering cache, a notebook that walks through one run step by step, and a check that proves the sweep code matches the original `mvo4.ipynb`.

The default settings sweep the K × Seeds × MinAppear × w1 grid across Swan 8, Swan 9, Swan 10D and Swan 11A, with crash injection on and `max_sharpe` locked at rf = 0.0. The crash curve's shape follows the **Black Swan Signatures** workbook: plain-average T1/T2, Swan 10D corrected, Swan 11A added. Each training window ends the Friday before the swan's week and runs 13.5 months.

The earlier **Locked Sweep Report** settings are kept as a preset, `presets/locked_sweep_report.py`, and reproduce that report exactly (see "Presets").

---

## Quick start

```bash
cd mvo4_sweep
pip install -r requirements.txt       # use the same Python you'll run with (python -m pip ...)

jupyter notebook mvo4_sweep.ipynb     # one run step by step, parity check, small sweep
python run_sweep.py                   # full sweep -> results/<timestamp>/
python parity_check.py                # engine vs. original notebook code, all events
```

To change what gets swept, **edit `config.py`**. It's the only file you should need to touch.

---

## What's in the folder

| Path | What it is |
|---|---|
| `config.py` | **Edit this.** Parameter grid, event dates, model settings, data file names. |
| `mvo4_engine.py` | The sweep logic, one function per step of `mvo4.ipynb` (each docstring names the notebook cells it reproduces). |
| `run_sweep.py` | Command-line sweep runner. Writes results to `results/<name>/`. |
| `mvo4_sweep.ipynb` | Notebook: one run step by step in `mvo4.ipynb`'s order, the parity check, a small sweep, and results tables. |
| `parity_check.py` | Runs the engine and the original notebook's code side by side and compares every intermediate. |
| `reference/mvo4_notebook_reference.py` | The original notebook's cells copied verbatim into one function (used by the parity check). |
| `reference/mvo4_original_MVODB.ipynb` | Copy of `New Training Docs/MVODB/mvo4.ipynb` the reference was taken from. |
| `data/All_SwanBeta100_MVO2.xlsx` | Per-stock beta for each historical swan (95 S&P 100 tickers). |
| `data/kappa_beta_per_swan.csv` | Real drawdown betas; used to add the `Swan10D` column, which the workbook doesn't have. |
| `data/MarketDD.xlsx` | Crash duration (T1 down, T2 recovery, in trading days) and depth per swan; shapes the injected crash. Swan 10D corrected and Swan 11A added from the Black Swan Signatures workbook. |
| `data/archive/MarketDD_before_signature_fix.xlsx` | The original `MarketDD.xlsx` (Swan 10D T2 = 415, no Swan 11A). Used only by the Locked Sweep Report preset. |
| `data/optdata/*.xlsx` | `R_fwd_30_MA` (forward-return estimate) per ticker, one file per event. |
| `presets/locked_sweep_report.py` | Settings that reproduce the Locked Sweep Report. |
| `cache/prices/` | Yahoo daily closes per event (stocks + ^OEX), so runs don't depend on re-downloading. |
| `cache/dbi/` | 1,000-seed Davies-Bouldin rankings per (training swans, K). The slow part of preselection. |
| `results/` | Sweep outputs. `signature_avg_13p5mo/` is the full grid with the current defaults; `locked_sweep_reproduction/` is the Locked Sweep Report settings. |

---

## How one run works

For one event and one combination of (K, Seeds, MinAppear, w1):

1. **Preselect** *(notebook cells `a1c7e6cb`–`9acaecb3`)*. Take each stock's beta in the event's 5 training swans. Fit K-means with K clusters for seeds 1–1,000 and rank the seeds by Davies-Bouldin score. The best `Seeds` seeds each vote for the stocks in their lowest-beta cluster, plus the next-lowest cluster if the lowest has fewer than 10 stocks. Stocks with at least `MinAppear` votes (among the 22 most-voted) are kept. Each kept stock's beta for injection is its mean over the training swans.
2. **Inject a crash** *(`2bd97cf4`–`3be4c870`)*. Take real daily closes over the training window and multiply by `1 − beta × market_drop`. The market-drop curve ramps down to depth `dd` over `t1` trading days, recovers over `t2` days, then stays at zero. `t1`, `t2` and `dd` are averages from `MarketDD.xlsx` over every swan from Swan1 to the event's last training swan: Swan1–7 for Swan 8, Swan1–8 for Swan 9, Swan1–9 for Swan 10D, Swan1–10D for Swan 11A. They match the Black Swan Signatures workbook's "use for swan N injection" rows (69–75). See `T1_METHOD`.
3. **Risk statistics** *(`668e84bf`–`3fea5d8b`)*. From the injected prices, over the **last 365 days** of the window only: each stock's max drawdown (`fall`), the covariance of weekly returns × 51, and `E_2 = R_fwd_30_MA / fall`.
4. **Optimize** *(`b876261e`, `4e90df30`)*. `max_sharpe(rf=0.0)` on `E_2/100`, each weight capped at `w1`. There is no fallback, so a combination that can't solve is recorded as `opt_failed`. Weights under 3% are dropped and the rest renormalized.
5. **Evaluate** *(`11a8c553`–`b45e8a54`)*. Buy the basket (default: at the close on the **peak** date) and measure its drop from the peak close to the **bottom** close, using real (non-injected) prices. **PR = 1 − portfolio drop / ^OEX drop.**

A sweep repeats this for every combination in `GRID` on every event in `EVENTS_TO_RUN`.

---

## Changing the sweep

### Parameter grid
Edit `GRID` in `config.py`. Every combination of the lists is run on every event:

```python
GRID = {"K": [6, 7], "Seeds": [3, 5], "MinAppear": [3], "w1": [0.13, 0.17]}
```

Or override from the command line without editing the file:

```bash
python run_sweep.py --name k67_test --events "Swan 9" "Swan 10D" --K 6 7 --w1 0.13 0.17
```

### Event dates
Each event in `EVENTS` has:

| Field | Meaning |
|---|---|
| `peak`, `bottom` | The real crash being scored: PR measures the drop from the `peak` close to the `bottom` close. Both must be trading days. |
| `buy_date` *(optional)* | Close at which shares are bought. Defaults to `peak`. The original notebook bought one trading day earlier (`start_swan_date` = 2020-02-18 for Swan 9); because share counts are fixed at purchase, that shifts PR slightly (48.817% → 48.822% for the notebook's settings). |
| `train_swans` | The 5 swans whose betas drive preselection, oldest first. Must not include the event itself. |
| `optdata` | File in `data/optdata/` with `R_fwd_30_MA` per ticker. |
| `train_start`, `train_end` *(optional)* | Override the training window rule below for this event. |

Change any date and rerun. Prices are re-downloaded automatically if the new dates fall outside what's cached.

### Training window
The window supplies the injection and the risk statistics. Both of its end dates are **included**.

- **End:** the Friday before the week the swan starts, i.e. the week of `peak`. Swan 8 → 2018-09-28, Swan 9 → 2020-02-14, Swan 10D → 2021-12-31, Swan 11A → 2025-02-14.
- **Start:** `TRAINING_WINDOW_MONTHS` (13.5 = 13 months + 15 days) before the end.
- **Fitting the crash curve:** the window must hold the whole crash curve, `t1 + t2` trading days. If it doesn't and `EXTEND_WINDOW_TO_FIT_CURVE = True`, the start moves back just far enough. Currently only Swan 8 needs this: its curve is 74 + 216 = 290 trading days, 5 more than its 13.5-month window holds, so it starts 2017-08-07 instead of 2017-08-13. Swan 9, 10D and 11A fit with 12, 30 and 5 days to spare. If the flag is `False`, the curve is cut off at the front instead and the run is flagged `curve_truncated`. The start actually used is recorded in every run.

Two things to know about the dates:

- **`train_end` is included; the notebook's `end` is not.** `mvo4.ipynb` calls `yf.download(start, end)`, which stops the day *before* `end`. So the notebook's `end = dt(2022,1,3)` means "train through 2021-12-31", which is `train_end="2021-12-31"` here.
- **Only the last 365 days of the window feed the risk statistics** (about 252 trading days). The rest of the window only holds the early part of the injected crash curve.

### Crash-curve shape (`T1_METHOD`)
`t2` is the plain average of T2 and `dd` the plain average of Actual DD, over Swan1 to the event's last training swan. `t1` depends on `T1_METHOD`:

| `T1_METHOD` | t1 | Swan 8 / 9 / 10D / 11A |
|---|---|---|
| `"average"` *(default)* | Plain average of T1, rounded. Matches the Black Swan Signatures workbook and the notebook's first `t1` line. | 74 / 72 / 66 / 79 |
| `"weighted"` | T1 weighted by how correlated each swan's betas are with later swans, truncated. This is the notebook's second `t1` line, which overwrites the first and is what the notebook actually runs. | 73 / 68 / 58 / 96 |

`t2` = 216 / 200 / 188 / 198 and `dd` = 25.4% / 24.7% / 25.5% / 25.6% for both methods. Rounding follows the notebook's Python `round`, which rounds halves to even, so 198.5 becomes 198.

### Model settings
Also in `config.py`, all defaulting to the notebook's values: `RISK_FREE_RATE`, `MAX_PRESELECTED` (22), `MIN_CLUSTER_SIZE` (10), `N_DBI_SEEDS` (1,000), `WEIGHT_FLOOR` (3%), `COV_ANNUALIZATION` (51), `BENCHMARK` (^OEX).

### Presets
A preset is a settings file that starts from `config.py` and overrides part of it. Run one with `--config`:

```bash
python run_sweep.py --config presets/locked_sweep_report.py --name locked_report
```

`presets/locked_sweep_report.py` restores the Locked Sweep Report's weighted `t1`, the original `MarketDD.xlsx`, and its fixed training windows (Swan 8 2017-07-22 → 2018-09-28, Swan 9 2019-01-01 → 2020-02-15, Swan 10D 2020-11-16 → 2021-12-31, Swan 11A 2023-10-31 → 2025-01-31). Copy it to make your own.

### Adding a new event
1. Add an entry to `EVENTS` with its `peak`, `bottom`, 5 training swans and an optdata file. The training window follows automatically.
2. Put the optdata workbook (same format as the others: a `Symbol` column and an `R_fwd_30_MA` column) in `data/optdata/`.
3. Every training swan needs a beta column. If one isn't in `All_SwanBeta100_MVO2.xlsx`, add it to `EXTRA_BETA_COLUMNS` (e.g. `"Swan11A": "S11A"` pulls S11A's betas from `kappa_beta_per_swan.csv`).
4. The last training swan and every swan before it must be in `SWAN_ORDER` and have a row in `MarketDD.xlsx` (the crash-curve shape uses them). Swan 11A's row and beta column are already there, so a Swan 12 event training on Swan7–11A works as is.
5. Add the event to `EVENTS_TO_RUN`.

The first run of a new (training swans, K) pair computes its 1,000-seed ranking, about a minute per K.

---

## Outputs

`python run_sweep.py --name X` writes `results/X/`:

| File | Contents |
|---|---|
| `raw_runs.csv` | One row per (event, K, Seeds, MinAppear, w1): `status`, `PR`, portfolio and market drops, `n_preselect` (stocks into MVO), `n_final` (stocks weighted), `holdings` (JSON of ticker → weight), the training start actually used. |
| `summary.xlsx` | Top combos by mean PR and by worst case, every combo that solved on all events (with "15->7"-style stock counts per event), average PR by parameter value, holdings for the top combos, run-status counts, and the raw runs. |
| `config_used.py` | Copy of the settings file at run time (plus `config_base.py` when a preset was used). |
| `run_info.json` | Settings file, `T1_METHOD`, MarketDD file, grid, events, training windows actually used, crash-curve parameters, package versions. |

`status` values: `ok`; `no_preselect` (fewer than 5 stocks survived the vote, e.g. `MinAppear` > `Seeds`); `opt_failed` (`max_sharpe` couldn't solve, usually because `n_stocks × w1 < 1`); `no_price_data` / `no_expected_returns` / `too_little_history` (data gaps).

---

## Proving the sweep matches `mvo4.ipynb`

`reference/mvo4_notebook_reference.py` is the original notebook's code, copied cell by cell (each block labeled with its cell id). `parity_check.py` runs it and the engine on identical inputs (the same prices, data files and training windows), for every event, and compares the results. Because the notebook's executed `t1` line is the weighted one, the check runs the engine with `T1_METHOD = "weighted"`. It compares:

- the full 1,000-seed Davies-Bouldin ranking
- the preselected stocks
- `t1`, `t2`, `dd` and the number of training days
- every stock's drawdown, `E_2`, and the covariance matrix (to 1e-10)
- the final weights and the PR

The only changes to the notebook code are marked `# ADAPTED`:

1. `yf.download` is replaced by the same cached price table the engine uses, so both sides see identical prices. The training slice includes `train_end` (see "Dates"), and evaluation prices start at the event's `buy_date`.
2. Cell `7335a9dc`'s look-ahead bug is replaced by the notebook's own commented-out correction. The bug sets each stock's beta to Swan 9's own realized beta, which is the swan being predicted; the correction uses the mean over the training swans.
3. The beta table is passed with its swan columns in chronological order, so the notebook's `start_swan:end_swan` slices select exactly the training swans (including the merged-in `Swan10D`).
4. Parameters and the evaluation dates are function arguments instead of hard-coded cell-14 values.
5. Two positional lookups (`Series[0]`) are written as `.iloc[0]`. They mean the same in pandas 2.x; pandas 3 removes the old form.

**Result:** see "Verification results" below.

---

## Deliberate differences from the notebook

Methodology choices that change results:

- **Plain-average `t1`** by default (`T1_METHOD = "average"`), matching the Black Swan Signatures workbook. The notebook's executed line uses the weighted average.
- **Corrected crash timing.** Swan 10D's T1/T2 in `MarketDD.xlsx` corrected from 195/415 to the signature workbook's 196/297, and Swan 11A (34/54) added. The copies of `MarketDD.xlsx` in `Original MVODB4/MVO4/` and `New Training Docs/MVO2/` still have the old values.
- **Training window rule** (Friday before the swan's week, 13.5 months) instead of hand-set dates.

Fixes that don't change any current result, but would matter for new data:

- **`data_cleaning` indexing fixed.** The notebook's version uses the loop counter as the row number, so it would fill the wrong cells. None of the current training windows have missing prices, so it never runs. The reference keeps the original.
- **Look-ahead bug fixed** (cell `7335a9dc`, see above).
- **Window extension** so the crash curve always fits (`EXTEND_WINDOW_TO_FIT_CURVE`). The notebook would crash with a shape error if the curve were longer than the window.
- **`Swan10D` beta** comes from `kappa_beta_per_swan.csv`. The workbook's `Swan10a`/`Swan10b` columns are false bottoms and are never used.
- **Risk statistics are computed once per basket and reused across `w1`** in a sweep. They don't depend on `w1`, so the numbers are identical; it's just faster.
- **Small-basket guards.** A run with fewer than 5 stocks at any stage is skipped with a status instead of erroring.

---

## Caching and reproducibility

- **Prices** (`cache/prices/`) are the exact Yahoo closes the Locked Sweep Report used. Yahoo occasionally revises history, and two tickers (BK, WBA) no longer download at all, so keeping this cache is what makes results reproducible later. Delete it to re-download.
- **DBI rankings** (`cache/dbi/`) are keyed by the exact beta matrix, so editing the beta data automatically triggers a recompute.
- **The scikit-learn version matters.** Which seeds win the Davies-Bouldin ranking, and what each seed's K-means fit looks like, can change between scikit-learn versions, which changes the preselected stocks. The cache and all results here were produced with **scikit-learn 1.7.2** (`requirements.txt` lists every version). The original notebook pins 1.2.2, which has no wheel for Python 3.12+, so none of these results have been checked against it. An independent re-run on 1.8.0 got a different preselection for the same settings. If you're on a different version, `dbi_ranking` prints a note. For exact reproduction, install the pinned versions.

---

## Caveats

- **Everything is in-sample.** The "best" combination is picked by looking at all four events' outcomes, so treat it as a description of the parameter space, not a forecast.
- **Crash-curve weighting (only with `T1_METHOD = "weighted"`).** The weights favor the most recent training swan: it is only correlated with itself, so it always scores 1.0. That's inherited from the notebook.
- **Swan 11A's forward returns are two weeks older than its window end.** Its optdata snapshot is dated 2025-01-31, but the window now ends 2025-02-14. The other three snapshots match their window end exactly (2018-09-28, 2020-02-14, 2021-12-31). Replace the file if a 2025-02-14 snapshot exists.
- **Window length is a convention, not an optimum.** Only the last 365 days feed the risk statistics, so window length mainly changes how much of the injected crash lands in that final year.

---

## Runtime

With the shipped cache: one run takes about 1 second, the full 5,376-run default grid about 7 minutes, and `parity_check.py` about 1 minute per event. A new (training swans, K) pair adds about a minute for its DBI ranking.

---

## Verification results

Checked on 2026-09-23, Python 3.13.14 with the package versions in `requirements.txt`.

**1. Engine vs. original notebook code** (`python parity_check.py`, K=7, Seeds=5, MinAppear=3, w1=0.17, the notebook's own settings). Run with the current windows and the corrected `MarketDD.xlsx`, with `T1_METHOD = "weighted"` because that is the notebook's executed line:

| Event | DBI ranking | Preselected | t1 / t2 / dd | Training days | fall, E_2, covariance | Weights | PR (both sides) |
|---|---|---|---|---|---|---|---|
| Swan 8 | identical | 20, identical | 73 / 216 / 0.2539 | 289 | max diff 0 | identical | 41.070586% |
| Swan 9 | identical | 20, identical | 68 / 200 / 0.2471 | 284 | max diff 0 | identical | 47.946778% |
| Swan 10D | identical | 14, identical | 58 / 188 / 0.2547 | 284 | max diff 0 | identical | 45.160628% |
| Swan 11A | identical | 13, identical | 96 / 198 / 0.2564 | 294 | max diff 0 | identical | 98.901730% |

Every intermediate is bit-for-bit identical. The notebook code recomputes the DBI ranking live, so this also confirms the shipped `cache/dbi/` rankings match a fresh computation. The same check passed with the Locked Sweep Report's windows and the original MarketDD before these changes.

**2. The Locked Sweep Report preset vs. the report's own sweep.** `results/locked_sweep_reproduction/` (`--config presets/locked_sweep_report.py`) was compared run by run against the sweep behind the report. All 5,376 runs have the same status, the same 4,130 are solved, PR differs by 0.0 on every solved run, and the stock counts (into MVO, weighted) are identical.

**3. Against the notebook as originally executed.** Running `New Training Docs/MVODB/mvo4.ipynb` itself (with the cell-`7335a9dc` fix) for Swan 9 printed PR **0.4882206644308368**, with weights AMZN/COST/MDLZ/MRK/WMT 17.48%, LLY 8.19%, EXC 4.39%. With the notebook's own settings (`T1_METHOD = "weighted"`, training 2019-01-01 → 2020-02-15, `buy_date="2020-02-18"`), this folder gives **0.4882206644308362** and the same weights, equal to 15 significant digits. The prices the notebook downloaded match the cache to 1e-16. Section 5 of `mvo4_sweep.ipynb` reruns this.

**4. `MarketDD.xlsx` vs. the Black Swan Signatures workbook.** Swan1–9 T1, T2 and Actual DD matched exactly. Swan 10D was corrected (195/415/0.2720 → 196/297/0.2723) and Swan 11A added (34/54/0.199), copying the workbook's cells directly. The resulting plain averages equal the workbook's rows 69, 71 and 73 (T1 71.5 / 66.2 / 79.2; T2 199.5 / 187.6 / 198.5).
