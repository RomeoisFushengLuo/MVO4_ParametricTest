"""Download frozen Yahoo Close caches and current analyst inputs for SP500 work.

Run --prepare-only without Yahoo access; the default run first checks the network.
Existing SP100 caches are preserved. Download errors and partial coverage stay visible.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONSTITUENTS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
WINDOWS = {
    "Swan_8": dict(start="2017-05-16", end_exclusive="2019-01-03", train_start="2017-08-07", train_end="2018-09-28", peak="2018-10-03", bottom="2018-12-24", betas=["Swan3", "Swan4", "Swan5", "Swan6", "Swan7"]),
    "Swan_9": dict(start="2018-10-03", end_exclusive="2020-04-02", train_start="2018-12-30", train_end="2020-02-14", peak="2020-02-19", bottom="2020-03-23", betas=["Swan4", "Swan5", "Swan6", "Swan7", "Swan8"]),
    "Swan_10D": dict(start="2020-08-18", end_exclusive="2022-10-22", train_start="2020-11-15", train_end="2021-12-31", peak="2022-01-03", bottom="2022-10-12", betas=["Swan5", "Swan6", "Swan7", "Swan8", "S9"]),
    "Swan_11A": dict(start="2023-08-22", end_exclusive="2025-04-18", train_start="2023-12-30", train_end="2025-02-14", peak="2025-02-19", bottom="2025-04-08", betas=["Swan6", "Swan7", "Swan8", "S9", "S10D"]),
}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
    temporary.replace(path)


def preflight():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.load(response)
    if not result.get("chart", {}).get("result"):
        raise RuntimeError("Yahoo preflight returned no SPY chart data")


def prepare(root, constituents=None):
    for directory in ["data", "raw/history", "raw/analysts", "cache/prices", "reports", "provenance"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    destination = root / "data/current_constituents.csv"
    if not destination.exists():
        if constituents:
            shutil.copy2(constituents, destination)
        else:
            with urllib.request.urlopen(CONSTITUENTS_URL, timeout=20) as response:
                destination.write_bytes(response.read())
        write_json(root / "provenance/constituents.json", dict(
            source=CONSTITUENTS_URL, retrieved_at_utc=utcnow(),
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            note="Current source snapshot; not historical event-date membership"))
    current = pd.read_csv(destination)
    beta_rows = pd.read_csv(REPO / "mvo4_sweep/data/kappa_beta_per_swan.csv")
    beta = beta_rows[beta_rows.Universe == "SP500"].pivot(index="Ticker", columns="Swan", values="Beta")
    beta.to_csv(root / "data/historical_beta_sp500.csv")
    historical_names = set(beta.index)
    current_names = set(current.Symbol)
    # Constituents use BRK.B/BF.B while the beta inputs use BRK-B/BF-B.
    # Collapse those share-class aliases without losing the original beta labels.
    groups = {}
    for name in sorted(historical_names | current_names):
        groups.setdefault(name.replace(".", "-"), []).append(name)
    records = []
    for symbol, aliases in sorted(groups.items()):
        beta_aliases = [name for name in aliases if name in historical_names]
        if len(beta_aliases) > 1:
            raise ValueError(f"Multiple beta labels normalize to {symbol}: {beta_aliases}")
        records.append(dict(Ticker=beta_aliases[0] if beta_aliases else symbol,
                            YahooSymbol=symbol, source_aliases=json.dumps(aliases),
                            in_existing_beta_universe=bool(beta_aliases),
                            in_current_constituents_snapshot=any(name in current_names for name in aliases)))
    universe = pd.DataFrame(records)
    universe.to_csv(root / "data/download_universe.csv", index=False)
    write_json(root / "data/event_windows.json", WINDOWS)
    return universe, beta


def history(symbol, root, end):
    path = root / "raw/history" / f"{symbol}.parquet"
    if path.exists():
        return pd.read_parquet(path), "cached"
    ticker = yf.Ticker(symbol)
    data = ticker.history(start="2017-05-16", end=end, interval="1d", auto_adjust=False,
                          actions=True, raise_errors=True, timeout=25)
    if data.empty or "Close" not in data or data.Close.dropna().empty:
        raise ValueError("No daily Close history returned")
    data.index = pd.DatetimeIndex(data.index).tz_localize(None).normalize()
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("History has duplicate or unsorted dates")
    data.to_parquet(path)
    write_json(path.with_suffix(".json"), dict(source="Yahoo Finance via yfinance", symbol=symbol,
               retrieved_at_utc=utcnow(), requested_start="2017-05-16", requested_end_exclusive=end,
               auto_adjust=False, yfinance_version=yf.__version__, rows=len(data),
               first_date=str(data.index.min().date()), last_date=str(data.index.max().date()),
               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return data, "downloaded"


def analyst(symbol, root):
    path = root / "raw/analysts" / f"{symbol}.json"
    if path.exists():
        return json.loads(path.read_text()), "cached"
    info = yf.Ticker(symbol).get_info()
    keys = ["symbol", "currency", "financialCurrency", "currentPrice", "regularMarketPrice",
            "regularMarketTime", "targetMeanPrice", "targetMedianPrice", "targetLowPrice",
            "targetHighPrice", "numberOfAnalystOpinions", "recommendationKey"]
    if not info or not any(info.get(k) is not None for k in ["targetMeanPrice", "targetMedianPrice"]):
        raise ValueError("No current analyst target returned")
    result = {k: info.get(k) for k in keys}
    result.update(retrieved_at_utc=utcnow(), source="Yahoo Finance current quote/financialData",
                  historical_point_in_time=False)
    write_json(path, result)
    return result, "downloaded"


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def assemble(root, universe, beta):
    closes = {}
    for row in universe.itertuples():
        path = root / "raw/history" / f"{row.YahooSymbol}.parquet"
        if path.exists():
            closes[row.Ticker] = pd.read_parquet(path)["Close"]
    stocks = pd.DataFrame(closes).sort_index()
    if not stocks.empty:
        stocks.to_parquet(root / "cache/all_stock_closes.parquet")
    benchmarks = {}
    for symbol in ["SPY", "^GSPC"]:
        path = root / "raw/history" / f"{symbol}.parquet"
        if path.exists():
            benchmarks[symbol] = pd.read_parquet(path)["Close"]
    if benchmarks:
        pd.DataFrame(benchmarks).to_parquet(root / "cache/benchmark_closes.parquet")
    coverage = []
    for event, window in WINDOWS.items():
        eligible = beta[window["betas"]].dropna().index
        flags = pd.DataFrame(index=eligible)
        flags.index.name = "Ticker"
        flags["has_history_file"] = eligible.isin(stocks.columns)
        flags["complete_training_and_endpoints"] = False
        # Preserve the benchmark trading calendar; do not invent prices or fill pre-IPO history.
        calendar = benchmarks.get("SPY", benchmarks.get("^GSPC"))
        if calendar is not None:
            dates = calendar.loc[window["start"]:pd.Timestamp(window["end_exclusive"]) - pd.Timedelta(days=1)].dropna().index
            px = stocks.reindex(index=dates, columns=universe.Ticker)
            px.to_parquet(root / "cache/prices" / f"{event}.parquet")
            for symbol, series in benchmarks.items():
                series.reindex(dates).to_frame(symbol).to_parquet(root / "cache/prices" / f"{event}_{symbol.replace('^', '')}.parquet")
            train = px.loc[window["train_start"]:window["train_end"]]
            endpoints = px.reindex([pd.Timestamp(window["peak"]), pd.Timestamp(window["bottom"])])
            good = train.notna().all() & train.gt(0).all() & endpoints.notna().all() & endpoints.gt(0).all()
            if len(train):
                flags["complete_training_and_endpoints"] = flags.index.isin(good.index[good])
            write_json(root / "cache/prices" / f"{event}.json", dict(
                source="Yahoo Finance", auto_adjust=False, event=event, window=window,
                tickers=universe.Ticker.tolist(), benchmark_symbols=list(benchmarks),
                assembled_at_utc=utcnow(), historical_price_rows=len(dates)))
        flags.to_csv(root / "reports" / f"{event}_price_coverage.csv")
        coverage.append(dict(event=event, eligible_beta_stocks=len(eligible),
                             history_files=int(flags.has_history_file.sum()),
                             complete_training_and_endpoints=int(flags.complete_training_and_endpoints.sum()),
                             benchmark_available=calendar is not None))
    pd.DataFrame(coverage).to_csv(root / "reports/price_coverage_summary.csv", index=False)
    analyst_rows = []
    for row in universe.itertuples():
        path = root / "raw/analysts" / f"{row.YahooSymbol}.json"
        if not path.exists():
            continue
        info = json.loads(path.read_text())
        prices = closes.get(row.Ticker, pd.Series(dtype=float)).dropna().tail(30)
        average = float(prices.mean()) if len(prices) == 30 else np.nan
        target = info.get("targetMeanPrice")
        # Align current targets with CURRENT prices, not 2018/2020 event prices.
        fresh_prices = bool(len(prices) and (datetime.now(timezone.utc).date() - prices.index[-1].date()).days <= 14)
        valid = info.get("currency") == "USD" and finite(target) and float(target) > 0 and finite(average) and average > 0 and fresh_prices
        analyst_rows.append(dict(Symbol=row.Ticker, YahooSymbol=row.YahooSymbol,
                snapshot_retrieved_at_utc=info["retrieved_at_utc"], currency=info.get("currency"),
                P_fwd=target, target_median=info.get("targetMedianPrice"),
                target_low=info.get("targetLowPrice"), target_high=info.get("targetHighPrice"),
                analyst_count=info.get("numberOfAnalystOpinions"),
                current_quote=info.get("currentPrice"), MA30_Close=average,
                MA30_start=str(prices.index[0].date()) if len(prices) else None,
                MA30_end=str(prices.index[-1].date()) if len(prices) else None,
                R_fwd_30_MA=float(target) / average - 1 if valid else np.nan,
                historical_point_in_time=False, usable=valid))
    pd.DataFrame(analyst_rows, columns=["Symbol", "YahooSymbol", "snapshot_retrieved_at_utc", "currency",
                 "P_fwd", "target_median", "target_low", "target_high", "analyst_count", "current_quote",
                 "MA30_Close", "MA30_start", "MA30_end", "R_fwd_30_MA", "historical_point_in_time", "usable"]
                 ).to_csv(root / "data/current_analyst_inputs.csv", index=False)
    return dict(price_history_files=len(closes), benchmarks=list(benchmarks),
                analyst_targets=len(analyst_rows), usable_analyst_inputs=sum(r["usable"] for r in analyst_rows),
                event_coverage=coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--constituents-file", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Process only the first N stocks for a real smoke download")
    args = parser.parse_args()
    root = args.output.resolve()
    universe, beta = prepare(root, args.constituents_file)
    status = dict(updated_at_utc=utcnow(), requested_stock_symbols=len(universe),
                  beta_universe=int(universe.in_existing_beta_universe.sum()),
                  current_constituents=int(universe.in_current_constituents_snapshot.sum()),
                  current_analysts_are_not_historical=True)
    if args.prepare_only or args.assemble_only:
        status.update(state="prepared_no_download" if args.prepare_only else "assembled_local_files", **assemble(root, universe, beta))
        write_json(root / "reports/download_status.json", status)
        print(json.dumps(status, indent=2))
        return
    try:
        preflight()
    except Exception as error:
        status.update(state="blocked_before_download", error=f"{type(error).__name__}: {error}", **assemble(root, universe, beta))
        write_json(root / "reports/download_status.json", status)
        print(json.dumps(status, indent=2))
        raise SystemExit(2)
    yf.set_tz_cache_location(str(root / ".runtime/yfinance"))
    end = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
    failures = []
    consecutive_network_failures = 0
    symbols = ["SPY", "^GSPC"] + universe.YahooSymbol.tolist()[:args.limit]
    for i, symbol in enumerate(symbols, 1):
        for kind, operation in [("prices", lambda: history(symbol, root, end)),
                                ("analysts", lambda: analyst(symbol, root))]:
            if kind == "analysts" and symbol in ["SPY", "^GSPC"]:
                continue
            try:
                _, outcome = operation()
                print(f"{i}/{len(symbols)} {symbol} {kind}: {outcome}", flush=True)
                consecutive_network_failures = 0
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                failures.append(dict(symbol=symbol, operation=kind, error=message, at_utc=utcnow()))
                pd.DataFrame(failures).to_csv(root / "reports/download_errors.csv", index=False)
                print(f"{symbol} {kind}: {message[:240]}", flush=True)
                if any(marker in message.lower() for marker in ["429", "too many requests", "403", "forbidden", "rate limit"]):
                    consecutive_network_failures += 1
                if consecutive_network_failures >= 3:
                    status.update(state="paused_access_or_rate_limit", **assemble(root, universe, beta))
                    write_json(root / "reports/download_status.json", status)
                    raise SystemExit(2)
            time.sleep(0.5)
        if i % 25 == 0:
            status.update(state="in_progress", processed=i, **assemble(root, universe, beta))
            write_json(root / "reports/download_status.json", status)
    status.update(state="completed_with_gaps" if failures else "completed", failures=len(failures),
                  completed_at_utc=utcnow(), **assemble(root, universe, beta))
    write_json(root / "reports/download_status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
