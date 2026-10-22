# SP500 experiment with uploaded caches

Read `REPORT.md` for the completed experiment, or `results/reference/` for the
four-event reference combination. All price and analyst inputs are frozen copies
of the user's uploads. This runner performs no network requests.

From the repository root, use Python 3.12 and the pinned scientific dependencies:

```bash
python -m pip install -r replication/requirements.txt
python -m unittest discover -s replication/tests -v
cd sp500_experiment
python -m unittest test_sp500 -v
cd ..
python sp500_experiment/run_sp500.py --reference-only --output sp500_experiment/results/new_reference
python sp500_experiment/run_sp500.py --output sp500_experiment/results/new_full
```

Outputs must be new directories to avoid overwriting previous evidence. Default
inputs and ranking caches are resolved relative to the script, not the terminal.
The runner caps OpenMP/BLAS at one thread for these small clustering matrices.

It subclasses the tested `replication/mvo_algo.py` pipeline. The only shared-engine
change is a `forward_returns(event)` input hook; the original SP100 data path and
behavior are preserved. The subclass supplies SP500 betas, ^GSPC prices, current
analyst estimates, and explicit complete-historical-price eligibility.

The guide's T1/T2 durations are a shared timing convention. Historical SP500
index drawdown depths come from the original beta/depth CSV. Current analyst
estimates are used for every event and marked as non-historical. These choices
make this an experimental historical test rather than an exact historical
replication or a clean comparison of SP100 versus SP500 universe effects.

`inputs/market_sp500.csv` is prepared from the original SP100 guide workbook's
T1/T2 and the SP500 index_dd rows in `inputs/beta_and_depth_source.csv`. Uploaded
files are not overwritten; the original source workbook and all uploaded inputs
are retained for review. `input_archives.json` records original archive hashes.
