"""
MVO4 sweep configuration -- THIS IS THE FILE TO EDIT.

Everything the sweep does is controlled from here: which parameter values to
sweep, which crash events to test against, the dates for each event, and the
model settings. Dates are plain "YYYY-MM-DD" strings. See README.md for what
each setting does and how to add a new event.
"""

# ---------------------------------------------------------------------------
# 1. Parameter grid -- every combination of these values is run on every event
# ---------------------------------------------------------------------------
GRID = {
    "K":         [5, 6, 7, 8, 9, 10, 11, 12],                     # number of K-means clusters
    "Seeds":     [3, 5, 7, 9, 11, 13],                            # top-DBI seeds that vote
    "MinAppear": [2, 3, 4, 5],                                    # votes a stock needs to be kept
    "w1":        [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17],      # max weight per stock
}

# ---------------------------------------------------------------------------
# 2. Crash events
#
#   peak / bottom           : the real crash. PR measures the basket's drop from
#                             the close on `peak` to the close on `bottom`.
#                             Both must be trading days.
#   buy_date (optional)     : close at which shares are bought; defaults to `peak`.
#                             The original mvo4.ipynb bought one trading day earlier
#                             (start_swan_date = 2020-02-18 for Swan 9), which moves
#                             PR slightly because share counts are fixed at purchase.
#   train_swans             : beta columns used for preselection (the 5 swans
#                             before the event, oldest first). Must be names from
#                             SWAN_ORDER below and must NOT include the event itself.
#   optdata                 : file in data/optdata/ holding R_fwd_30_MA per ticker.
#
# Training window (injection + risk statistics). Both dates are INCLUDED:
#   train_end   : the Friday before the week the swan starts (the week of `peak`).
#   train_start : TRAINING_WINDOW_MONTHS (13.5) before train_end, moved earlier
#                 only if the window doesn't hold the whole crash curve (t1 + t2
#                 trading days; see EXTEND_WINDOW_TO_FIT_CURVE).
#   Either can be overridden per event with train_start="YYYY-MM-DD" /
#   train_end="YYYY-MM-DD". (The notebook's yf.download(start, end) EXCLUDES
#   `end`, so a notebook end of dt(2020,2,15) is train_end="2020-02-14" here.)
# ---------------------------------------------------------------------------
EVENTS = {
    "Swan 8": dict(
        peak="2018-10-03", bottom="2018-12-24",
        train_swans=["Swan3", "Swan4", "Swan5", "Swan6", "Swan7"],
        optdata="Swan8_sp100_optdata4.xlsx",
    ),
    "Swan 9": dict(
        peak="2020-02-19", bottom="2020-03-23",
        # buy_date="2020-02-18",   # uncomment to match the original notebook exactly
        train_swans=["Swan4", "Swan5", "Swan6", "Swan7", "Swan8"],
        optdata="revised_swan9_sp100_optdata4.xlsx",
    ),
    "Swan 10D": dict(
        peak="2022-01-03", bottom="2022-10-12",
        train_swans=["Swan5", "Swan6", "Swan7", "Swan8", "Swan9"],
        optdata="revised_swan10d_sp100_optdata4.xlsx",
    ),
    "Swan 11A": dict(
        peak="2025-02-19", bottom="2025-04-08",
        train_swans=["Swan6", "Swan7", "Swan8", "Swan9", "Swan10D"],
        optdata="revised_swan11a_sp100_optdata4.xlsx",
    ),
}

# Which events a sweep runs, and the column order in every output.
EVENTS_TO_RUN = ["Swan 8", "Swan 9", "Swan 10D", "Swan 11A"]

# ---------------------------------------------------------------------------
# 3. Model settings
# ---------------------------------------------------------------------------
TRAINING_WINDOW_MONTHS = 13.5

# How the crash curve's length is set, over every swan from Swan1 to the event's
# last training swan (rows of MarketDD.xlsx). t2 and dd are always plain averages.
#   "average"  : t1 = plain average of T1 (the Black Swan Signatures workbook,
#                rows 69-75, e.g. 71.5 for Swan 9)
#   "weighted" : t1 = T1 weighted by beta correlation (what mvo4.ipynb cell
#                b807776f actually runs; used by the Locked Sweep Report)
T1_METHOD = "average"

RISK_FREE_RATE = 0.0            # max_sharpe(rf). No fallback: a combo that can't solve is marked opt_failed
MAX_PRESELECTED = 22            # notebook: selected_stock_number
MIN_CLUSTER_SIZE = 10           # lowest-beta cluster smaller than this -> next-lowest cluster is added
N_DBI_SEEDS = 1000              # K-means seeds scored by Davies-Bouldin before the top `Seeds` vote
KMEANS_N_INIT = 10
WEIGHT_FLOOR = 0.03             # weights below this are dropped, the rest renormalized
MIN_STOCKS = 5                  # fewer candidates than this -> run skipped
COV_ANNUALIZATION = 51          # weekly covariance x 51, as in the notebook

# If the training window has fewer than t1 + t2 (+ CURVE_BUFFER_TRADING_DAYS)
# trading days, move train_start back just far enough. If False, the curve is cut
# off at the front instead and the run is flagged curve_truncated.
EXTEND_WINDOW_TO_FIT_CURVE = True
CURVE_BUFFER_TRADING_DAYS = 0

BENCHMARK = "^OEX"              # market index for the PR denominator

# ---------------------------------------------------------------------------
# 4. Data files (paths relative to this folder)
# ---------------------------------------------------------------------------
BETA_WORKBOOK = "data/All_SwanBeta100_MVO2.xlsx"     # per-stock beta, one column per swan
EXTRA_BETAS_FILE = "data/kappa_beta_per_swan.csv"    # real drawdown betas for swans missing from the workbook
EXTRA_BETA_COLUMNS = {"Swan10D": "S10D",             # new column name -> Swan label in the CSV
                      "Swan11A": "S11A"}
EXTRA_BETA_UNIVERSE = "SP100"
MARKETDD_FILE = "data/MarketDD.xlsx"                 # T1 / T2 / Actual DD per swan (crash-curve shape)
OPTDATA_DIR = "data/optdata"

# Chronological order of swans. An event's crash curve averages over every swan
# from Swan1 up to its last training swan. Every name here must be a beta column
# AND a row in MarketDD.xlsx. Swan10a/Swan10b are false bottoms and are left out.
SWAN_ORDER = ["Swan1", "Swan2", "Swan3", "Swan4", "Swan5",
              "Swan6", "Swan7", "Swan8", "Swan9", "Swan10D", "Swan11A"]
