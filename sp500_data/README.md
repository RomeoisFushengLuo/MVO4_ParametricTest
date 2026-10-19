# SP500 price and current analyst acquisition

The live download is currently blocked by the environment egress proxy with
`Tunnel connection failed: 403 Forbidden`. This is not evidence of a Yahoo
subscription requirement or ticker failure. `reports/download_status.json`
records the actual result: zero new Yahoo stock histories, benchmarks, or
analyst snapshots have been downloaded so far. Empty analyst/coverage tables
are explicit pending outputs, not completed inputs.

The downloaded constituents CSV is a current source snapshot from
https://github.com/datasets/s-and-p-500-companies. It contains 503 securities.
The existing SP500 beta file contains 502 symbols. Their normalized union has
518 symbols. BRK.B/BRK-B and BF.B/BF-B are share-class notation aliases; the
plan preserves the beta-file names and records both source aliases. This list
is not a verified historical event-date constituent history.

## Resume

For the downloadable ZIP, extract it and open a terminal in the extracted
`SP500_Downloader` folder. Keep `sp500_data` and `mvo4_sweep` beside each other.
Then run these commands using Python 3.12:

```bash
python -m pip install -r sp500_data/requirements.txt
python sp500_data/download_sp500.py --limit 1
python sp500_data/download_sp500.py
```

The ZIP includes the required SP500 beta source CSV. It contains downloader code,
the constituents snapshot, and preparation reports; it does not contain newly
downloaded Yahoo stock prices or analyst targets. The cloud network restriction
described above does not establish whether Yahoo is reachable from your computer.

The prepared environment uses:

```bash
cd /workspace/MVO4_ParametricTest
/workspace/venvs/sp500-download/bin/python sp500_data/download_sp500.py --limit 1
/workspace/venvs/sp500-download/bin/python sp500_data/download_sp500.py
```

The first command is a real one-stock smoke download plus SPY and ^GSPC.
Run the second after the smoke download works. On another machine, install
`sp500_data/requirements.txt` in a virtual environment first.

Yahoo destinations needed in environment settings:

- query1.finance.yahoo.com
- query2.finance.yahoo.com
- fc.yahoo.com
- guce.yahoo.com
- finance.yahoo.com

These additions were saved to the environment configuration draft. Saving a
draft does not activate network changes. Review and save in environment settings,
then publish the environment; retry the smoke download after runtime access changes.
Existing package-manager network presets are preserved.

## Data conventions

- Source: Yahoo Finance through yfinance 1.3.0, daily interval,
  `auto_adjust=False`. Keep Yahoo's Close and Adj Close separately, along with
  available dividends/splits; do not substitute adjusted Close for Close.
- Request histories from 2017-05-16 through the current date. Freeze per-symbol
  raw histories and metadata on first successful fetch; subsequent runs reuse
  them. UTC retrieval times, requested ranges, and file hashes are recorded.
- Download both SPY and ^GSPC so the later SP500 experiment can choose a
  benchmark consistent with its beta/drawdown convention. No automatic change
  is made to the existing SP100 engine or crash-shape workbook.
- Assemble four event caches, retaining missing values. Do not backfill
  pre-IPO history or silently replace a delisted stock with a different company.
  Cache existence alone does not indicate complete coverage; inspect reports.
- Current analyst data includes mean/median/low/high targets, available analyst
  counts, quote currency, and retrieval times. Missing targets remain missing.
- The derived current `R_fwd_30_MA` is mean target / most recent 30-trading-day
  Close average - 1. Targets must be positive, quote currency USD, and the latest
  price at most 14 calendar days old. Other cases remain unusable.
- Current analyst inputs are marked `historical_point_in_time=False`. They are
  not written over the original event-date workbooks or silently divided by
  historical 2018/2020/2022/2025 prices. Reusing these current return estimates
  in historical tests is an explicit experimental choice, not historical replication.
- Errors are recorded, and repeated access/rate-limit failures pause the run.
  Re-running resumes successful frozen files. A normal run may still finish with
  gaps, particularly for delisted/renamed stocks and missing analyst coverage.

The original SP100 caches remain under `mvo4_sweep/cache/prices/`, unchanged.
Offline cache/coverage tests can be run with:

```bash
cd sp500_data
/workspace/venvs/sp500-download/bin/python -m unittest test_download_sp500 -v
```

Offline tests do not validate live Yahoo access or returned vendor data.
