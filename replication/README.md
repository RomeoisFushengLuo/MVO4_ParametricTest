# MVO4 replication deliverable

`../REPLICATION_GUIDE.md` is the specification. This is an independent implementation,
not a wrapper around `mvo4_sweep/mvo4_engine.py`. The attached experimental Python
module and event notebooks are replaced by one shared pipeline and four revised
notebooks. Originals and the existing repository implementation are untouched.

## Run

From this directory, with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --prefix "$VIRTUAL_ENV" --name mvo4 --display-name "MVO4"
python -m unittest discover -s tests -v
python run_replication.py --output results/reference --checkpoints
python run_replication.py --full --output results/full
python validate_replication.py --full-results results/full/raw_runs.csv --output results/validation
python make_checkpoints.py --K 12 --Seeds 3 --MinAppear 3 --w1 0.11 --output results/k12_checkpoints.xlsx
```

For the prepared cloud machine, use `source /workspace/mvo4-runtime/activate.sh`.
All data paths default to the sibling `mvo4_sweep` directory. Supply `--source`
when that directory lives elsewhere. Price caches are required; there are no live
downloads. Exact replication enforces scikit-learn 1.7.2, the version used for the
repository's saved results. The guide and README disagree about whether 1.8.0 is
equivalent, so this implementation uses their common reproducible version.

Run `mvo4_Swan8.ipynb`, `mvo4_Swan9.ipynb`, `mvo4_Swan10.ipynb`, or
`mvo4_Swan11.ipynb` from this directory using the MVO4 kernel. Their event names
are Swan 8, Swan 9, Swan 10D, and Swan 11A respectively. Each notebook runs a
reference combination and a small sweep. Large sweeps and exports are explicit.

## Files and outputs

- `mvo_algo.py`: historical inputs, seed ranking, voting, injection, risk inputs,
  optimization, evaluation, sweep, summaries, and checkpoint export.
- `run_replication.py`: CLI; default is K=7, Seeds=3, MinAppear=3, w1=0.17 on four events.
- `make_checkpoints.py`: export a complete set of intermediate sheets for one combination.
- `validate_replication.py`: compare reference-combination intermediates against the
  existing engine, and optionally compare every sweep row with saved results.
- `tests/test_pipeline.py`: regression tests for seed ranges, thresholds, missing
  prices, crash timing, optimization rules, training leakage, and summaries.
- `IMPLEMENTATION_REPORT.md`: differences from the uploads, evidence, and limitations.

A run writes `raw_runs.csv`, `summary.xlsx`, and `run_info.json`; `--checkpoints`
also writes `replication_checkpoints.xlsx`. Summaries include complete combinations,
mean/worst-case rankings, marginal parameter results, status counts, and all raw rows.
Failed optimization is retained as `opt_failed` with an error explanation. Fewer
than five preselected stocks is `no_preselect`; insufficient valid forward-return
inputs is `no_expected_returns`. Unexpected input defects raise `DataError` and
stop the run rather than silently hiding broken inputs. Generated DBI caches live
under `.cache/dbi`, independently keyed by input matrix, parameters, and versions.

This is a replacement API, not a drop-in compatibility layer for old notebooks.
The obsolete Holt/GARCH/quandl features, monthly covariance, alternate PR objectives,
and risk-free-rate retry logic are omitted because they do not implement the guide.
Keep the original uploads if those separate experiments are needed.
