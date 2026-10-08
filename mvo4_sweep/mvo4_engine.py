"""
MVO4 sweep engine.

Each function is one step of New Training Docs/MVODB/mvo4.ipynb, with the
notebook cell id(s) it reproduces noted in its docstring. The only intended
differences from the notebook are listed in README.md ("Differences from the
notebook"); reference/mvo4_notebook_reference.py and parity_check.py prove the
rest is identical.

Settings come from config.py. Edit config.py (or the attributes of the `config`
module from a notebook) rather than this file.
"""
import datetime
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from pypfopt import EfficientFrontier

import config as C

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"

_memo = {}


def _path(rel):
    return HERE / rel


def ts(d):
    return pd.Timestamp(d)


def slug(name):
    return name.replace(" ", "_")


def clear_memory():
    """Forget loaded data/weights (call after editing config values in a notebook)."""
    _memo.clear()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def beta_table():
    """Per-stock beta by swan: the MVO2 workbook plus any EXTRA_BETA_COLUMNS
    (e.g. real Swan10D beta) merged in from EXTRA_BETAS_FILE."""
    if "beta" not in _memo:
        b = pd.read_excel(_path(C.BETA_WORKBOOK), index_col=0)
        if C.EXTRA_BETA_COLUMNS:
            kb = pd.read_csv(_path(C.EXTRA_BETAS_FILE))
            for col, label in C.EXTRA_BETA_COLUMNS.items():
                s = kb[(kb.Swan == label) & (kb.Universe == C.EXTRA_BETA_UNIVERSE)].set_index("Ticker")["Beta"]
                b[col] = s.reindex(b.index)
        missing = [s for s in C.SWAN_ORDER if s not in b.columns]
        if missing:
            raise ValueError(f"SWAN_ORDER names with no beta column: {missing}")
        _memo["beta"] = b
    return _memo["beta"]


def market_dd():
    if "marketdd" not in _memo:
        m = pd.read_excel(_path(C.MARKETDD_FILE), index_col=0)
        missing = [s for s in C.SWAN_ORDER if s not in m.index]
        if missing:
            raise ValueError(f"SWAN_ORDER names with no row in {C.MARKETDD_FILE}: {missing}")
        _memo["marketdd"] = m
    return _memo["marketdd"]


def r_fwd(event):
    """R_fwd_30_MA per ticker for an event (notebook cell 389215e9)."""
    key = ("rfwd", event)
    if key not in _memo:
        f = pd.read_excel(_path(C.OPTDATA_DIR) / C.EVENTS[event]["optdata"])
        f = f.set_index("Symbol") if "Symbol" in f.columns else f.set_index(f.columns[0])
        _memo[key] = f["R_fwd_30_MA"].astype(float)
    return _memo[key]


# ---------------------------------------------------------------------------
# Step 1a: eligible universe and swan weights   (cells a1c7e6cb, 6f9b6522, 70e36794)
# ---------------------------------------------------------------------------
def eligible_beta(train_swans):
    """Cell a1c7e6cb: keep stocks with beta in at least 2 of the training swans."""
    b = beta_table()
    sel = b.loc[:, train_swans]
    return b[sel.T.isnull().sum() < sel.shape[1] - 1]


def swan_weights(train_swans):
    """Cells 6f9b6522 + 70e36794: weight_swan for every swan from Swan1 to the last
    training swan = each swan's mean correlation with itself and later swans."""
    pb = eligible_beta(train_swans)
    cols = C.SWAN_ORDER[: C.SWAN_ORDER.index(train_swans[-1]) + 1]
    corr = pb[cols].corr()
    for i in range(corr.shape[1]):
        for j in range(i + 1, corr.shape[0]):
            corr.iloc[j, i] = np.nan
    avg = corr.mean(axis=1)
    return avg / avg.sum()


# ---------------------------------------------------------------------------
# Step 2a: crash curve shape   (cells b807776f, 451fc9c4)
# ---------------------------------------------------------------------------
def curve_params(train_swans, method=None):
    """Cell b807776f, over every swan from Swan1 to the last training swan:
    t2 = mean T2 (rounded), dd = mean Actual DD, and t1 by config.T1_METHOD:
      "average"  - mean T1, rounded (the notebook's first t1 line; the Black Swan
                   Signatures workbook's "use for swan N injection" rows)
      "weighted" - T1 weighted by weight_swan, truncated (the notebook's second
                   t1 line, which overwrites the first)."""
    method = method or C.T1_METHOD
    rows = C.SWAN_ORDER[: C.SWAN_ORDER.index(train_swans[-1]) + 1]
    md = market_dd()
    if method == "average":
        t1 = int(round(md.loc[rows, "T1"].mean(), 0))
    elif method == "weighted":
        t1 = int((md.loc[rows, "T1"].to_numpy() * swan_weights(train_swans).to_numpy()).sum())
    else:
        raise ValueError(f"T1_METHOD must be 'average' or 'weighted', not {method!r}")
    t2 = int(round(md.loc[rows, "T2"].mean(), 0))
    dd = float(md.loc[rows, "Actual DD"].mean())
    return max(t1, 2), t2, dd


def crash_curve(t1, t2, dd, total_day):
    """Cell 451fc9c4: t1 days ramping 0 -> dd, t2 days decaying back, then t0 zeros.
    If the curve is longer than the window (t0 < 0), its start is cut off and
    truncated=True (only happens when EXTEND_WINDOW_TO_FIT_CURVE is False)."""
    t0 = total_day - t1 - t2
    md = []
    j = 0
    inc = dd / (t1 - 1)
    for _ in range(t1):
        md.append(round(j, 4))
        j = j + inc
    base = (t2 + 1) ** (1 / (dd * 100))
    l1 = [math.log((i + 2), base) / 100 for i in range(t2)]
    for i in range(t2):
        md.append(round((dd - l1[i]), 4))
    truncated = t0 < 0
    if truncated:
        md = md[-total_day:]
    else:
        md.extend([0] * t0)
    return np.array(md, dtype=float).reshape(-1, 1), truncated


def friday_before_week(day):
    """The Friday before the Monday of `day`'s week."""
    day = ts(day)
    return day - datetime.timedelta(days=day.weekday() + 3)


def nominal_window(event):
    """(start, end) before any extension: end = the Friday before the peak's week,
    start = TRAINING_WINDOW_MONTHS earlier (13.5 = 13 months + 15 days). Either can
    be overridden per event with train_start / train_end in config.EVENTS."""
    ev = C.EVENTS[event]
    end = ts(ev["train_end"]) if "train_end" in ev else friday_before_week(ev["peak"])
    if "train_start" in ev:
        start = ts(ev["train_start"])
    else:
        months = int(C.TRAINING_WINDOW_MONTHS)
        extra_days = round((C.TRAINING_WINDOW_MONTHS - months) * 30)
        start = end - pd.DateOffset(months=months) - datetime.timedelta(days=extra_days)
    return ts(start), end


def training_window(event):
    """(start, end, extended). If EXTEND_WINDOW_TO_FIT_CURVE and the nominal window has
    fewer than t1 + t2 (+ CURVE_BUFFER_TRADING_DAYS) trading days, start moves back to
    the earliest trading day that gives exactly that many."""
    ev = C.EVENTS[event]
    start, end = nominal_window(event)
    if not C.EXTEND_WINDOW_TO_FIT_CURVE:
        return start, end, False
    t1, t2, _ = curve_params(ev["train_swans"])
    need = t1 + t2 + C.CURVE_BUFFER_TRADING_DAYS
    px, _ = prices(event)
    days = px.index[px.index <= end]
    if ((days >= start)).sum() >= need:
        return start, end, False
    if len(days) < need:
        raise ValueError(f"{event}: cached prices start {days[0].date()}, not early enough for a "
                         f"{need}-trading-day window; delete cache/prices/{slug(event)}.* to re-download")
    return days[-need], end, True


# ---------------------------------------------------------------------------
# Prices (Yahoo Close, auto_adjust=False), cached per event under cache/prices/
# ---------------------------------------------------------------------------
def prices(event):
    """(stock close prices for the whole beta universe, benchmark close series),
    covering the event's training window through bottom + 10 days."""
    key = ("px", event)
    if key in _memo:
        return _memo[key]
    ev = C.EVENTS[event]
    start, _ = nominal_window(event)
    # 60 extra days (~40 trading days) so training_window can extend the start
    # to fit the crash curve without another download
    need_start = start - datetime.timedelta(days=60)
    need_end = ts(ev["bottom"]) + datetime.timedelta(days=6)
    d = CACHE_DIR / "prices"
    d.mkdir(parents=True, exist_ok=True)
    px_f, bm_f, meta_f = d / f"{slug(event)}.parquet", d / f"{slug(event)}_benchmark.parquet", d / f"{slug(event)}.json"
    tickers = list(beta_table().index)
    if meta_f.exists() and px_f.exists() and bm_f.exists():
        meta = json.loads(meta_f.read_text())
        if (ts(meta["start"]) <= need_start and ts(meta["end"]) >= need_end
                and meta.get("benchmark") == C.BENCHMARK and set(tickers) <= set(meta["tickers"])):
            px = pd.read_parquet(px_f)
            bm = pd.read_parquet(bm_f).iloc[:, 0]
            _memo[key] = (px, bm)
            return px, bm
    import yfinance as yf
    dl_start = (need_start - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
    dl_end = (ts(ev["bottom"]) + datetime.timedelta(days=10)).strftime("%Y-%m-%d")
    print(f"  downloading prices for {event}: {len(tickers)} tickers, {dl_start} -> {dl_end}", flush=True)
    px = yf.download(tickers, start=dl_start, end=dl_end, auto_adjust=False, progress=False)["Close"]
    bm = yf.download(C.BENCHMARK, start=dl_start, end=dl_end, auto_adjust=False, progress=False)["Close"]
    bm = bm.iloc[:, 0] if isinstance(bm, pd.DataFrame) else bm
    px.to_parquet(px_f)
    bm.to_frame(C.BENCHMARK).to_parquet(bm_f)
    meta_f.write_text(json.dumps(dict(start=dl_start, end=dl_end, benchmark=C.BENCHMARK, tickers=tickers), indent=1))
    _memo[key] = (px, bm)
    return px, bm


# ---------------------------------------------------------------------------
# Step 1b: preselection   (cell b2d69e9c, method = "K_means_multiple")
# ---------------------------------------------------------------------------
def training_beta(train_swans):
    """Cell b2d69e9c: selected_beta = training-swan betas, stocks with all of them present."""
    return eligible_beta(train_swans).loc[:, train_swans].dropna()


def dbi_ranking(train_swans, K):
    """Cell b2d69e9c, first loop: fit K-means for seeds 1..N_DBI_SEEDS and sort seeds by
    Davies-Bouldin score (lowest first). Cached under cache/dbi/, keyed by the exact
    training beta matrix, so editing the beta data automatically invalidates it."""
    tb = training_beta(train_swans)
    h = hashlib.md5(tb.to_csv().encode()).hexdigest()[:10]
    d = CACHE_DIR / "dbi"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"dbi_{'-'.join(train_swans)}_K{K}_n{C.N_DBI_SEEDS}_{h}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            obj = pickle.load(fh)
        cached_v = obj.get("sklearn_version")
        if cached_v and cached_v != sklearn.__version__ and not _memo.get("warned_sklearn"):
            print(f"  note: DBI rankings in cache/ were computed with scikit-learn {cached_v}; "
                  f"installed is {sklearn.__version__}. Delete cache/dbi/ to recompute.", flush=True)
            _memo["warned_sklearn"] = True
        return obj["ranking"]
    seed_dbi = {}
    for r in range(1, C.N_DBI_SEEDS + 1):
        km = KMeans(n_clusters=K, random_state=r, n_init=C.KMEANS_N_INIT).fit(tb)
        seed_dbi[r] = metrics.davies_bouldin_score(tb, km.labels_)
    ranking = sorted(seed_dbi.items(), key=lambda x: x[1])
    with open(f, "wb") as fh:
        pickle.dump(dict(ranking=ranking, sklearn_version=sklearn.__version__), fh)
    return ranking


def preselect(train_swans, K, Seeds, MinAppear, return_votes=False):
    """Cell b2d69e9c, second loop: each of the top `Seeds` DBI seeds refits K-means and
    votes for its lowest-beta cluster (plus the next-lowest if the lowest has fewer than
    MIN_CLUSTER_SIZE stocks). Stocks with >= MinAppear votes, among the MAX_PRESELECTED
    most-voted, are kept. Returns the kept stocks' training betas plus 'avg' (their mean
    beta over the training swans = the injection beta, fixed cell 7335a9dc), or None if
    fewer than MIN_STOCKS survive."""
    tb = training_beta(train_swans)
    ranking = dbi_ranking(train_swans, K)
    stock_list = []
    for seed, _ in ranking[:Seeds]:
        sb = tb.copy()
        km = KMeans(n_clusters=K, random_state=seed, n_init=C.KMEANS_N_INIT).fit(sb)
        sb["Group"] = km.labels_
        res = km.cluster_centers_.mean(axis=1)
        res[res < 0] = 0
        sb["avg"] = res[km.labels_]
        sb = sb.sort_values(by=["avg"])
        first_group = sb[sb["Group"] == sb["Group"].iloc[0]]
        if first_group.shape[0] < C.MIN_CLUSTER_SIZE:
            second_group = sb[sb["Group"] == sb["Group"].iloc[first_group.shape[0]]]
            picked = pd.concat([first_group, second_group], axis=0)
        else:
            picked = first_group
        stock_list.extend(picked.index.values)
    values, counts = np.unique(stock_list, return_counts=True)
    sorted_count = sorted(zip(values, counts), key=lambda x: x[1], reverse=True)
    keep = sorted(s for s, c in sorted_count[: C.MAX_PRESELECTED] if c >= MinAppear)
    votes = pd.Series(dict(sorted_count), name="votes").sort_values(ascending=False, kind="stable")
    if len(keep) < C.MIN_STOCKS:
        return (None, votes) if return_votes else None
    pre = tb.loc[keep, train_swans].copy()
    pre["avg"] = pre[train_swans].mean(axis=1)
    return (pre, votes) if return_votes else pre


# ---------------------------------------------------------------------------
# Helpers copied from the notebook   (cells a375b8e8, 7fe29e3f, 74e1c1f8)
# ---------------------------------------------------------------------------
def data_cleaning(data):
    """Cell a375b8e8, with its row-index bug fixed (the notebook uses the loop counter
    as the row number). Fills a missing price with the geometric mean of its neighbours."""
    if data.isna().any().any():
        rows, cols = np.where(data.isna())
        for k in range(len(rows)):
            row, col = rows[k], cols[k]
            if row == len(data) - 1:
                data.iloc[row, col] = data.iloc[row - 1, col] ** 2 / data.iloc[row - 2, col]
            elif row == 0:
                data.iloc[row, col] = data.iloc[row + 1, col] ** 2 / data.iloc[row + 2, col]
            else:
                data.iloc[row, col] = math.sqrt(data.iloc[row - 1, col] * data.iloc[row + 1, col])


def get_price_years_ago(data, date, years):
    cutoff = date - datetime.timedelta(days=int(years * 365))
    return data[data.index <= cutoff]


def max_drawdown_absolute(returns):
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    end = dd.argmin()
    start = r.iloc[:end].argmax()
    return abs(mdd), start, end


# ---------------------------------------------------------------------------
# Steps 2b-3: injection and risk statistics   (cells 2bd97cf4 .. 3fea5d8b)
# ---------------------------------------------------------------------------
def risk_inputs(event, pre):
    """Everything the optimizer needs for one preselected basket. Does not depend on
    w1, so a sweep computes it once per (K, Seeds, MinAppear) and reuses it.
    Returns a dict (status 'ok'), or a dict with a failure status."""
    ev = C.EVENTS[event]
    start, end, extended = training_window(event)
    px, _ = prices(event)

    # cell 2bd97cf4: training-window prices for the preselected stocks
    raw = px.loc[start:end, [t for t in pre.index if t in px.columns]].dropna(axis=1, how="all")
    if raw.shape[1] < C.MIN_STOCKS:
        return dict(status="no_price_data")
    swan_beta = pre.loc[raw.columns, "avg"]

    # cells b807776f, 451fc9c4, 3be4c870: inject the synthetic crash
    total_day = raw.shape[0]
    t1, t2, dd = curve_params(ev["train_swans"])
    curve, truncated = crash_curve(t1, t2, dd, total_day)
    inj = pd.DataFrame((1 - np.array(swan_beta) * curve) * raw.to_numpy(), index=raw.index, columns=raw.columns)
    inj = inj.dropna(how="all", axis=1)
    if inj.shape[1] < C.MIN_STOCKS:
        return dict(status="no_price_data")

    # cell 668e84bf
    data_cleaning(inj)

    # cell 47a07252: weekly log returns over the last year
    one_year_ago_date = get_price_years_ago(inj, end, 1).index.values[-1]
    week = pd.to_datetime(inj.index.values).strftime("%W").values
    week_difference = np.delete(np.where(week != np.roll(week, 1)), 0, None)
    weekly_return = inj.iloc[week_difference - 1].pct_change().dropna()
    one_year_weekly_return = np.log(1 + weekly_return)
    one_year_weekly_return = one_year_weekly_return.iloc[one_year_weekly_return.index >= one_year_ago_date]
    if one_year_weekly_return.shape[0] < 10:
        return dict(status="too_little_history")

    # cell 25f95798: max drawdown per stock over the last year
    stock_return_one_year = inj.pct_change().dropna().loc[one_year_ago_date:, :]
    fall = pd.Series({c: max_drawdown_absolute(stock_return_one_year[c])[0] for c in stock_return_one_year.columns})

    # cells 389215e9, 3fea5d8b: E_2 = R_fwd / fall, weekly covariance x 51
    rf = r_fwd(event)
    common = [t for t in inj.columns if t in rf.index]
    fall = fall[common].replace(0, np.nan)
    cov = pd.DataFrame(np.cov(np.exp(one_year_weekly_return[common]) - 1, rowvar=False, bias=False) * C.COV_ANNUALIZATION,
                       index=common, columns=common)
    E_2 = (rf[common] / fall)
    valid = E_2.replace([np.inf, -np.inf], np.nan).dropna().index
    if len(valid) < C.MIN_STOCKS:
        return dict(status="no_expected_returns")
    return dict(status="ok", E_2=E_2[valid], cov=cov.loc[valid, valid], fall=fall[valid], R_fwd=rf[valid],
                n_preselect=inj.shape[1], train_start=start, train_end=end, window_extended=extended,
                total_day=total_day, t1=t1, t2=t2, dd=dd, curve=curve[:, 0], curve_truncated=truncated,
                raw_prices=raw, injected_prices=inj, one_year_ago_date=pd.Timestamp(one_year_ago_date))


# ---------------------------------------------------------------------------
# Step 4: optimization   (cells b876261e, 4e90df30)
# ---------------------------------------------------------------------------
def optimize(E_2, cov, w1):
    """max_sharpe(RISK_FREE_RATE) on E_2/100 with weights in [0, w1]; weights below
    WEIGHT_FLOOR dropped and the rest renormalized. None if the solver fails."""
    try:
        ef = EfficientFrontier(E_2 / 100, cov, weight_bounds=(0, w1))
        ef.max_sharpe(C.RISK_FREE_RATE)
    except Exception:
        return None
    w = pd.Series(ef.clean_weights())
    w = w[w >= C.WEIGHT_FLOOR]
    if w.empty:
        return None
    return w / w.sum()


# ---------------------------------------------------------------------------
# Step 5: evaluation   (cells 11a8c553 .. b45e8a54)
# ---------------------------------------------------------------------------
def evaluate(event, w):
    """Buy the basket at the close on `buy_date` (default: `peak`), then measure its drop
    from `peak` to `bottom` on real (not injected) prices. PR = 1 - portfolio drop /
    benchmark drop over the same dates."""
    ev = C.EVENTS[event]
    peak, bottom = ts(ev["peak"]), ts(ev["bottom"])
    buy = ts(ev.get("buy_date", ev["peak"]))
    px, bm = prices(event)
    win = px.loc[buy: bottom + datetime.timedelta(days=5), [t for t in w.index if t in px.columns]].ffill().bfill()
    w = w.reindex(win.columns).dropna()
    w = w / w.sum()
    shares = np.array(10000 * w.T) / np.array(win.iloc[0, :][w.index]).T
    pv = win[w.index] @ shares.T
    bwin = bm.loc[buy: bottom + datetime.timedelta(days=5)].ffill().bfill()
    port_dd = abs(pv.asof(bottom) / pv.asof(peak) - 1)
    mkt_dd = abs(float(bwin.asof(bottom)) / float(bwin.asof(peak)) - 1)
    return dict(PR=1 - port_dd / mkt_dd, port_dd=float(port_dd), mkt_dd=mkt_dd, weights=w)


# ---------------------------------------------------------------------------
# One full run, and the sweep
# ---------------------------------------------------------------------------
def run_one(event, K, Seeds, MinAppear, w1, detail=False):
    """Preselect -> inject -> risk stats -> optimize -> evaluate for one combination.
    detail=True also returns every intermediate (used by the notebook)."""
    ev = C.EVENTS[event]
    pre, votes = preselect(ev["train_swans"], K, Seeds, MinAppear, return_votes=True)
    out = dict(event=event, K=K, Seeds=Seeds, MinAppear=MinAppear, w1=w1)
    if pre is None:
        return dict(out, status="no_preselect", votes=votes) if detail else dict(out, status="no_preselect")
    ri = risk_inputs(event, pre)
    return _finish(out, pre, votes, ri, w1, detail)


def _finish(out, pre, votes, ri, w1, detail):
    if ri["status"] != "ok":
        res = dict(out, status=ri["status"])
    else:
        w = optimize(ri["E_2"], ri["cov"], w1)
        if w is None:
            res = dict(out, status="opt_failed", n_preselect=ri["n_preselect"])
        else:
            ev = evaluate(out["event"], w)
            res = dict(out, status="ok", PR=ev["PR"], port_dd=ev["port_dd"], mkt_dd=ev["mkt_dd"],
                       n_preselect=ri["n_preselect"], n_final=len(ev["weights"]), weights=ev["weights"])
    if "train_start" in ri:
        res.update(train_start=ri["train_start"].date(), window_extended=ri["window_extended"],
                   curve_truncated=ri["curve_truncated"])
    if detail:
        res.update(preselected=pre, votes=votes, risk=ri)
    return res


def run_sweep(grid=None, events=None, out_dir=None, checkpoint_every=200):
    """Run every combination in `grid` (default config.GRID) on every event (default
    config.EVENTS_TO_RUN). Writes raw_runs.csv to out_dir if given; returns the rows.
    Injection and risk stats are computed once per (event, K, Seeds, MinAppear) and
    reused across w1 -- same numbers as calling run_one for each, just faster."""
    grid = grid or C.GRID
    events = events or C.EVENTS_TO_RUN
    total = len(events) * len(grid["K"]) * len(grid["Seeds"]) * len(grid["MinAppear"]) * len(grid["w1"])
    rows, done, t0 = [], 0, time.time()
    out_dir = Path(out_dir) if out_dir else None
    for event in events:
        ev = C.EVENTS[event]
        prices(event)
        for K in grid["K"]:
            for Seeds in grid["Seeds"]:
                for MinAppear in grid["MinAppear"]:
                    pre = preselect(ev["train_swans"], K, Seeds, MinAppear)
                    ri = risk_inputs(event, pre) if pre is not None else None
                    for w1 in grid["w1"]:
                        out = dict(event=event, K=K, Seeds=Seeds, MinAppear=MinAppear, w1=w1)
                        r = dict(out, status="no_preselect") if pre is None else _finish(out, pre, None, ri, w1, False)
                        if "weights" in r:
                            r["holdings"] = json.dumps({k: round(float(v), 6) for k, v in r.pop("weights").items()})
                        rows.append(r)
                        done += 1
                        if done % checkpoint_every == 0:
                            el = time.time() - t0
                            print(f"  {done}/{total}  elapsed {el:.0f}s  eta {el / done * (total - done):.0f}s", flush=True)
                            if out_dir:
                                pd.DataFrame(rows).to_csv(out_dir / "raw_runs_partial.csv", index=False)
    df = pd.DataFrame(rows)
    if out_dir:
        df.to_csv(out_dir / "raw_runs.csv", index=False)
        p = out_dir / "raw_runs_partial.csv"
        if p.exists():
            p.unlink()
    print(f"  finished {len(df)} runs in {time.time() - t0:.0f}s; {(df.status == 'ok').sum()} solved", flush=True)
    return df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize(raw, events=None, top_n=25):
    """Tables from a raw_runs DataFrame: per-combo PR on every event (combos that solved
    on all events), top by mean and by worst case, marginal trends, and holdings."""
    events = events or [e for e in C.EVENTS_TO_RUN if e in set(raw.event)]
    idx = ["K", "Seeds", "MinAppear", "w1"]
    ok = raw[raw.status == "ok"]
    piv = ok.pivot_table(index=idx, columns="event", values="PR").reindex(columns=events).dropna()
    piv["mean_PR"] = piv[events].mean(axis=1)
    piv["median_PR"] = piv[events].median(axis=1)
    piv["min_PR"] = piv[events].min(axis=1)
    piv["worst_event"] = piv[events].idxmin(axis=1)
    for e in events:
        s = ok[ok.event == e].set_index(idx)
        piv[f"{e} stocks"] = (s["n_preselect"].astype(int).astype(str) + "->" + s["n_final"].astype(int).astype(str)).reindex(piv.index)
    piv = piv.reset_index()
    marg = []
    for p in idx:
        g = piv.groupby(p)[["mean_PR", "min_PR"]].mean()
        g["combos"] = piv.groupby(p).size()
        marg.append(g.reset_index().rename(columns={p: "value"}).assign(parameter=p))
    marg = pd.concat(marg)[["parameter", "value", "mean_PR", "min_PR", "combos"]]
    top_mean = piv.sort_values("mean_PR", ascending=False).head(top_n)
    top_min = piv.sort_values(["min_PR", "mean_PR"], ascending=False).head(top_n)
    hold = []
    picks = pd.concat([top_mean.head(10), top_min.head(10)]).drop_duplicates(idx)
    for _, c in picks.iterrows():
        sel = ok[(ok.K == c.K) & (ok.Seeds == c.Seeds) & (ok.MinAppear == c.MinAppear) & (ok.w1 == c.w1)]
        for _, r in sel.iterrows():
            for t, wt in sorted(json.loads(r.holdings).items(), key=lambda x: -x[1]):
                hold.append(dict(K=c.K, Seeds=c.Seeds, MinAppear=c.MinAppear, w1=c.w1, event=r.event,
                                 PR=r.PR, ticker=t, weight=wt))
    status = raw.groupby(["event", "status"]).size().unstack(fill_value=0)
    return dict(all_combos=piv, top_mean=top_mean, top_worst_case=top_min, marginal=marg,
                holdings=pd.DataFrame(hold), status=status)


def write_summary(raw, path, events=None):
    s = summarize(raw, events)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        s["top_mean"].to_excel(xw, sheet_name="Top by mean PR", index=False)
        s["top_worst_case"].to_excel(xw, sheet_name="Top by worst case", index=False)
        s["all_combos"].to_excel(xw, sheet_name="All complete combos", index=False)
        s["marginal"].to_excel(xw, sheet_name="Marginal trends", index=False)
        s["holdings"].to_excel(xw, sheet_name="Holdings (top combos)", index=False)
        s["status"].to_excel(xw, sheet_name="Run status")
        raw.to_excel(xw, sheet_name="Raw runs", index=False)
    return s


def environment():
    import pypfopt, yfinance
    return dict(python=__import__("sys").version.split()[0], pandas=pd.__version__, numpy=np.__version__,
                scikit_learn=sklearn.__version__, pypfopt=getattr(pypfopt, "__version__", "?"),
                yfinance=yfinance.__version__)
