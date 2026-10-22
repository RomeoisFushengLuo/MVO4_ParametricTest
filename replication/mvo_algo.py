"""Independent MVO4 implementation of REPLICATION_GUIDE.md, sections 1–9.

This replaces the uploaded experimental mvo_algo.py for the replication workflow.
It deliberately does not import the repository engine or execute Yahoo downloads.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
import importlib.metadata
import json
import math

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from pypfopt import EfficientFrontier
from pypfopt.exceptions import OptimizationError
from cvxpy.error import SolverError


SWAN_ORDER = [f"Swan{i}" for i in range(1, 10)] + ["Swan10D", "Swan11A"]
GRID = dict(K=list(range(5, 13)), Seeds=[3, 5, 7, 9, 11, 13],
            MinAppear=[2, 3, 4, 5], w1=[0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17])


@dataclass(frozen=True)
class Event:
    name: str
    peak: str
    bottom: str
    train_swans: tuple[str, ...]
    optdata: str


EVENTS = {
    "Swan 8": Event("Swan 8", "2018-10-03", "2018-12-24",
                    tuple(f"Swan{i}" for i in range(3, 8)), "Swan8_sp100_optdata4.xlsx"),
    "Swan 9": Event("Swan 9", "2020-02-19", "2020-03-23",
                    tuple(f"Swan{i}" for i in range(4, 9)), "revised_swan9_sp100_optdata4.xlsx"),
    "Swan 10D": Event("Swan 10D", "2022-01-03", "2022-10-12",
                      tuple(f"Swan{i}" for i in range(5, 10)), "revised_swan10d_sp100_optdata4.xlsx"),
    "Swan 11A": Event("Swan 11A", "2025-02-19", "2025-04-08",
                      ("Swan6", "Swan7", "Swan8", "Swan9", "Swan10D"),
                      "revised_swan11a_sp100_optdata4.xlsx"),
}
PARAMETERS = ["K", "Seeds", "MinAppear", "w1"]


class DataError(ValueError):
    """Missing or invalid data; must not be silently labeled an optimizer failure."""


def dbscores(beta: pd.DataFrame, number_of_clusters: int,
             number_iteration: int = 1000) -> list[tuple[int, float]]:
    """Score seeds 1 through number_iteration INCLUSIVE, using each actual seed."""
    if not 2 <= number_of_clusters < len(beta):
        raise ValueError("K must be at least 2 and smaller than the eligible universe")
    scores = []
    for seed in range(1, number_iteration + 1):
        fit = KMeans(n_clusters=number_of_clusters, random_state=seed, n_init=10).fit(beta)
        scores.append((seed, float(davies_bouldin_score(beta, fit.labels_))))
    return sorted(scores, key=lambda pair: pair[1])  # stable ties retain ascending seed


def preselect(beta: pd.DataFrame, ranking: list[tuple[int, float]], K: int,
              Seeds: int, MinAppear: int) -> tuple[pd.DataFrame, pd.Series]:
    """Guide §4: centroid voting, top 22, >= threshold, ticker order for ties."""
    if not 1 <= Seeds <= len(ranking) or MinAppear < 1:
        raise ValueError("Seeds must be within the ranking and MinAppear must be positive")
    counts: Counter = Counter()
    for seed, _ in ranking[:Seeds]:
        fit = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(beta)
        score = np.maximum(fit.cluster_centers_.mean(axis=1), 0)
        assigned = beta.assign(Group=fit.labels_, cluster_score=score[fit.labels_])
        assigned = assigned.sort_values("cluster_score")
        first = assigned[assigned.Group == assigned.Group.iloc[0]]
        chosen = first
        if len(first) < 10:
            second = assigned[assigned.Group == assigned.Group.iloc[len(first)]]
            chosen = pd.concat([first, second])
        counts.update(chosen.index)
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    votes = pd.Series(dict(ordered), name="votes", dtype=int)
    names = sorted(name for name, count in ordered[:22] if count >= MinAppear)
    selected = beta.loc[names].copy()
    selected["avg"] = selected.mean(axis=1)
    return selected, votes


def crash_curve(t1: int, t2: int, dd: float, total_days: int) -> np.ndarray:
    """Guide §5: four-decimal ramp/recovery at the FRONT of the window."""
    if t1 < 2 or t2 < 1 or not 0 < dd < 1 or total_days < t1 + t2:
        raise DataError("Invalid curve parameters or training window shorter than the curve")
    increment = dd / (t1 - 1)
    value = 0.0
    ramp = []
    for _ in range(t1):
        ramp.append(round(value, 4))
        value += increment
    base = (t2 + 1) ** (1 / (dd * 100))
    recovery = [round(dd - math.log(i + 2, base) / 100, 4) for i in range(t2)]
    return np.array(ramp + recovery + [0.0] * (total_days - t1 - t2))


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Fill isolated gaps geometrically; fail clearly on ambiguous multi-day gaps.

    At the first row use next**2/next_next; at the last use prev**2/prev_prev.
    The guide's 'prev' shorthand cannot apply literally to a first-row gap.
    """
    result = prices.copy().astype(float)
    for col in result:
        values = result[col].to_numpy(copy=True)
        missing = np.flatnonzero(np.isnan(values))
        for i in missing:
            if len(values) < 3:
                raise DataError(f"Not enough neighbors to fill {col}")
            if i == 0:
                a, b = values[1:3]
                fill = a * a / b if b > 0 else np.nan
            elif i == len(values) - 1:
                a, b = values[i - 1], values[i - 2]
                fill = a * a / b if b > 0 else np.nan
            else:
                a, b = values[i - 1], values[i + 1]
                fill = math.sqrt(a * b) if a > 0 and b > 0 else np.nan
            if not np.isfinite(fill) or fill <= 0:
                raise DataError(f"Cannot geometrically fill {col} at {result.index[i]}")
            values[i] = fill
        result[col] = values
    if not np.isfinite(result.to_numpy()).all() or (result <= 0).any().any():
        raise DataError("Training prices must be finite and positive after cleaning")
    return result


def max_drawdown_absolute(returns: pd.Series) -> float:
    """Notebook convention: cumulative wealth starts with the first retained return."""
    wealth = (1 + returns).cumprod()
    return float(abs((wealth / wealth.cummax() - 1).min()))


def optimize(expected: pd.Series, covariance: pd.DataFrame, w1: float
             ) -> tuple[pd.Series | None, str | None]:
    """Guide §7: rf=0 once; clean, discard <3%, then renormalize; no fallback."""
    if not 0 < w1 <= 1:
        raise ValueError("w1 must be in (0, 1]")
    if len(expected) * w1 < 1 - 1e-12:
        return None, "infeasible: n_stocks * w1 < 1"
    try:
        frontier = EfficientFrontier(expected / 100, covariance, weight_bounds=(0, w1))
        frontier.max_sharpe(risk_free_rate=0.0)
        weights = pd.Series(frontier.clean_weights(), dtype=float)
    except (OptimizationError, SolverError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"
    weights = weights[weights >= 0.03]
    if weights.empty or not np.isfinite(weights).all() or weights.sum() <= 0:
        return None, "no finite weights remain after the 3% floor"
    return weights / weights.sum(), None


class Replication:
    """Data loading and reusable intermediate stages for a full parameter sweep."""

    def __init__(self, source: str | Path | None = None, cache: str | Path | None = None):
        if sklearn.__version__ != "1.7.2":
            raise RuntimeError("Exact replication requires scikit-learn==1.7.2")
        self.source = (Path(source) if source else Path(__file__).resolve().parents[1] / "mvo4_sweep").resolve()
        self.cache = Path(cache).resolve() if cache else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.beta = pd.read_excel(self.source / "data/All_SwanBeta100_MVO2.xlsx", index_col=0)
        extra = pd.read_csv(self.source / "data/kappa_beta_per_swan.csv")
        for column, label in [("Swan10D", "S10D"), ("Swan11A", "S11A")]:
            rows = extra.loc[(extra.Universe == "SP100") & (extra.Swan == label)]
            self.beta[column] = rows.set_index("Ticker")["Beta"].reindex(self.beta.index)
        self.beta = self.beta[SWAN_ORDER].astype(float)
        self.market = pd.read_excel(self.source / "data/MarketDD.xlsx", index_col=0)
        self._prices = {}
        self._rankings = {}
        self._selection = {}
        self._risk = {}

    def training_beta(self, event: str) -> pd.DataFrame:
        return self.beta.loc[:, list(EVENTS[event].train_swans)].dropna().copy()

    def curve_params(self, event: str) -> tuple[int, int, float]:
        last = SWAN_ORDER.index(EVENTS[event].train_swans[-1])
        history = self.market.loc[SWAN_ORDER[:last + 1]]
        return round(history.T1.mean()), round(history.T2.mean()), float(history["Actual DD"].mean())

    def prices(self, event: str) -> tuple[pd.DataFrame, pd.Series]:
        if event not in self._prices:
            slug = event.replace(" ", "_")
            folder = self.source / "cache/prices"
            stocks = pd.read_parquet(folder / f"{slug}.parquet")
            benchmark = pd.read_parquet(folder / f"{slug}_benchmark.parquet").iloc[:, 0]
            if not stocks.index.is_monotonic_increasing or stocks.index.has_duplicates:
                raise DataError(f"{event}: cached price dates must be unique and increasing")
            if not benchmark.index.is_monotonic_increasing or benchmark.index.has_duplicates:
                raise DataError(f"{event}: benchmark dates must be unique and increasing")
            self._prices[event] = stocks, benchmark
        return self._prices[event]

    def training_window(self, event: str) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
        peak = pd.Timestamp(EVENTS[event].peak)
        end = peak - pd.Timedelta(days=peak.weekday() + 3)
        start = end - pd.DateOffset(months=13) - pd.Timedelta(days=15)
        prices, _ = self.prices(event)
        days = prices.index[prices.index <= end]
        t1, t2, _ = self.curve_params(event)
        extended = len(days[days >= start]) < t1 + t2
        if extended:
            if len(days) < t1 + t2:
                raise DataError(f"{event}: cache lacks enough days for the whole crash curve")
            start = days[-(t1 + t2)]
        elif prices.index[0] > start:
            raise DataError(f"{event}: cache does not cover the nominal training start")
        return start, end, extended

    def dbi_ranking(self, event: str, K: int) -> list[tuple[int, float]]:
        key = (event, K)
        if key not in self._rankings:
            beta = self.training_beta(event)
            digest = sha256(pd.util.hash_pandas_object(beta, index=True).values.tobytes()
                            + json.dumps([list(beta.columns), K, 1000, 10,
                                          sklearn.__version__, np.__version__]).encode()).hexdigest()
            path = self.cache / f"dbi_{digest}.json" if self.cache else None
            if path and path.exists():
                ranking = [(int(s), float(d)) for s, d in json.loads(path.read_text())]
                if len(ranking) != 1000 or sorted(s for s, _ in ranking) != list(range(1, 1001)):
                    raise DataError(f"Invalid ranking cache: {path}")
                if not all(np.isfinite(d) for _, d in ranking) or any(
                        ranking[i][1] > ranking[i + 1][1] for i in range(999)):
                    raise DataError(f"Invalid DBI scores: {path}")
            else:
                ranking = dbscores(beta, K)
                if path:
                    path.write_text(json.dumps(ranking))
            self._rankings[key] = ranking
        return self._rankings[key]

    def preselect(self, event: str, K: int, Seeds: int, MinAppear: int):
        key = (event, K, Seeds, MinAppear)
        if key not in self._selection:
            self._selection[key] = preselect(self.training_beta(event), self.dbi_ranking(event, K), K, Seeds, MinAppear)
        return self._selection[key]

    def forward_returns(self, event: str) -> pd.Series:
        """Event analyst inputs; subclasses may supply explicitly labeled snapshots."""
        optdata = pd.read_excel(self.source / "data/optdata" / EVENTS[event].optdata, index_col=0)
        return optdata["R_fwd_30_MA"].astype(float)

    def risk_inputs(self, event: str, selected: pd.DataFrame) -> dict:
        key = (event, tuple(selected.index))
        if key in self._risk:
            return self._risk[key]
        start, end, extended = self.training_window(event)
        stocks, _ = self.prices(event)
        raw = stocks.loc[start:end, selected.index].dropna(axis=1, how="all")
        if raw.shape[1] < 5:
            raise DataError(f"{event}: fewer than five stocks have training prices")
        t1, t2, dd = self.curve_params(event)
        curve = crash_curve(t1, t2, dd, len(raw))
        injected = clean_prices(raw * (1 - curve[:, None] * selected.loc[raw.columns, "avg"].to_numpy()))
        prior = injected.index[injected.index <= end - pd.Timedelta(days=365)]
        if len(prior) == 0:
            raise DataError(f"{event}: no price on or before the one-year cutoff")
        cutoff = prior[-1]
        daily = injected.pct_change(fill_method=None).dropna()
        fall = daily.loc[cutoff:].apply(max_drawdown_absolute)
        weeks = injected.index.strftime("%W").to_numpy()
        boundaries = np.flatnonzero(weeks[1:] != weeks[:-1]) + 1
        weekly = injected.iloc[boundaries - 1].pct_change(fill_method=None).dropna()
        weekly = weekly.loc[cutoff:]
        if len(weekly) < 2:
            raise DataError(f"{event}: insufficient weekly returns for covariance")
        # Use the notebook's log/exp round trip to minimize floating-point differences.
        simple = np.exp(np.log1p(weekly)) - 1
        covariance = pd.DataFrame(np.cov(simple, rowvar=False, bias=False) * 51,
                                  index=weekly.columns, columns=weekly.columns)
        forward = self.forward_returns(event).reindex(fall.index).astype(float)
        expected = (forward / fall).replace([np.inf, -np.inf], np.nan).dropna()
        names = expected.index
        covariance = covariance.loc[names, names]
        if not np.isfinite(covariance.to_numpy()).all():
            raise DataError(f"{event}: nonfinite covariance")
        result = dict(start=start, end=end, extended=extended, cutoff=cutoff, t1=t1, t2=t2, dd=dd,
                      total_days=len(raw), raw_prices=raw, injected_prices=injected,
                      curve=curve, weekly_returns=weekly, fall=fall.loc[names],
                      R_fwd=forward.loc[names], E_2=expected, cov=covariance)
        self._risk[key] = result
        return result

    def evaluate(self, event: str, weights: pd.Series) -> dict:
        stocks, benchmark = self.prices(event)
        ev = EVENTS[event]
        peak, bottom = pd.Timestamp(ev.peak), pd.Timestamp(ev.bottom)
        stocks = stocks.loc[peak:bottom, weights.index].ffill()
        benchmark = benchmark.loc[peak:bottom].ffill()
        for frame in [stocks, benchmark]:
            if peak not in frame.index or bottom not in frame.index:
                raise DataError(f"{event}: missing peak or bottom trading date")
            endpoints = frame.loc[[peak, bottom]].to_numpy()
            if not np.isfinite(endpoints).all() or np.any(endpoints <= 0):
                raise DataError(f"{event}: invalid evaluation endpoint prices")
        shares = 10000 * weights / stocks.loc[peak]
        values = stocks @ shares
        port_dd = abs(float(values.loc[bottom] / values.loc[peak] - 1))
        market_dd = abs(float(benchmark.loc[bottom] / benchmark.loc[peak] - 1))
        if market_dd == 0:
            raise DataError(f"{event}: zero benchmark drawdown makes PR undefined")
        return dict(PR=1 - port_dd / market_dd, port_dd=port_dd, market_dd=market_dd,
                    portfolio_values=values, benchmark_values=10000 * benchmark / benchmark.loc[peak])

    def run_one(self, event: str, K=7, Seeds=3, MinAppear=3, w1=0.17, detail=False) -> dict:
        selected, votes = self.preselect(event, K, Seeds, MinAppear)
        row = dict(event=event, K=K, Seeds=Seeds, MinAppear=MinAppear, w1=w1,
                   status="no_preselect", PR=np.nan, port_dd=np.nan, market_dd=np.nan,
                   n_preselect=len(selected), n_final=0, holdings="{}", error=None)
        if detail:
            row.update(preselected=selected, votes=votes)
        if len(selected) < 5:
            return row
        risk = self.risk_inputs(event, selected)
        row.update(train_start=str(risk["start"].date()), train_end=str(risk["end"].date()))
        if detail:
            row["risk"] = risk
        if len(risk["E_2"]) < 5:
            row["status"] = "no_expected_returns"
            return row
        weights, error = optimize(risk["E_2"], risk["cov"], w1)
        if weights is None:
            row.update(status="opt_failed", error=error)
            return row
        evaluated = self.evaluate(event, weights)
        row.update(status="ok", PR=evaluated["PR"], port_dd=evaluated["port_dd"],
                   market_dd=evaluated["market_dd"], n_final=len(weights), holdings=json.dumps(weights.to_dict()))
        if detail:
            row.update(weights=weights, evaluation=evaluated)
        return row

    def run_sweep(self, grid=None, events=None) -> pd.DataFrame:
        grid = GRID if grid is None else grid
        events = list(EVENTS) if events is None else events
        rows = []
        for event in events:
            for K in grid["K"]:
                for Seeds, MinAppear, w1 in product(grid["Seeds"], grid["MinAppear"], grid["w1"]):
                    rows.append(self.run_one(event, K, Seeds, MinAppear, w1))
                print(f"{event}, K={K}: {len(rows)} runs completed", flush=True)
        return pd.DataFrame(rows)

    def export_checkpoints(self, path: str | Path, **parameters) -> None:
        """Export all four events' intermediate data for any requested combination."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        overview = []
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for event in EVENTS:
                result = self.run_one(event, detail=True, **parameters)
                slug = event.replace("Swan ", "S")
                ranking = self.dbi_ranking(event, result["K"])
                pd.DataFrame(ranking, columns=["seed", "DBI"]).to_excel(writer, sheet_name=f"{slug} DBI", index=False)
                selected = result["preselected"]
                stats = self.training_beta(event)
                stats["votes"] = result["votes"].reindex(stats.index).fillna(0).astype(int)
                stats["selected"] = stats.index.isin(selected.index)
                stats["injection_beta"] = selected["avg"]
                if "risk" in result:
                    risk = result["risk"]
                    for name in ["fall", "R_fwd", "E_2"]:
                        stats[name] = risk[name]
                    stats["final_weight"] = result.get("weights", pd.Series(dtype=float)).reindex(stats.index).fillna(0)
                    risk["cov"].to_excel(writer, sheet_name=f"{slug} covariance")
                    pd.DataFrame({"market_drop": risk["curve"]}, index=risk["raw_prices"].index).to_excel(writer, sheet_name=f"{slug} curve")
                    risk["injected_prices"].to_excel(writer, sheet_name=f"{slug} injected prices")
                    risk["weekly_returns"].to_excel(writer, sheet_name=f"{slug} weekly returns")
                    overview.append(dict(event=event, status=result["status"], start=risk["start"],
                                         end=risk["end"], cutoff=risk["cutoff"], days=risk["total_days"],
                                         t1=risk["t1"], t2=risk["t2"], dd=risk["dd"], PR=result["PR"]))
                else:
                    overview.append(dict(event=event, status=result["status"]))
                stats.to_excel(writer, sheet_name=f"{slug} votes and stats")
            pd.DataFrame(overview).to_excel(writer, sheet_name="Overview", index=False)
            pd.DataFrame([parameters]).to_excel(writer, sheet_name="Parameters", index=False)


def summarize(raw: pd.DataFrame, events=None) -> dict[str, pd.DataFrame]:
    events = list(EVENTS) if events is None else list(events)
    solved = raw.loc[raw.status == "ok"]
    wide = solved.pivot(index=PARAMETERS, columns="event", values="PR").reindex(columns=events)
    complete = wide.dropna().copy()
    complete["mean_PR"] = complete[events].mean(axis=1)
    complete["min_PR"] = complete[events].min(axis=1)
    complete = complete.reset_index()
    by_mean = complete.sort_values(["mean_PR", "min_PR"], ascending=False, kind="stable")
    by_worst = complete.sort_values(["min_PR", "mean_PR"], ascending=False, kind="stable")
    marginals = []
    for parameter in PARAMETERS:
        group = complete.groupby(parameter).agg(mean_PR=("mean_PR", "mean"),
                                                min_PR=("min_PR", "mean"), combos=("mean_PR", "size")).reset_index()
        group.insert(0, "parameter", parameter)
        group = group.rename(columns={parameter: "value"})
        marginals.append(group)
    return dict(all_combos=complete, top_mean=by_mean, top_worst_case=by_worst,
                marginal=pd.concat(marginals, ignore_index=True),
                status_counts=raw.groupby(["event", "status"]).size().rename("count").reset_index())


def write_outputs(raw: pd.DataFrame, output: str | Path, events=None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "raw_runs.csv", index=False)
    tables = summarize(raw, events)
    with pd.ExcelWriter(output / "summary.xlsx", engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
        raw.to_excel(writer, sheet_name="Raw runs", index=False)
    return tables


def versions() -> dict:
    import platform
    return dict(python=platform.python_version(), **{name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scikit-learn", "scipy", "PyPortfolioOpt", "cvxpy"]})
