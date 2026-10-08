"""
Settings that reproduce the Locked Sweep Report exactly:

    python run_sweep.py --config presets/locked_sweep_report.py --name locked_report

Differences from the current config.py defaults:
  - t1 weighted by beta correlation (mvo4.ipynb's executed line), not the plain average
  - the original MarketDD.xlsx, before Swan10D was corrected to the Black Swan
    Signatures workbook (T1 195 / T2 415 instead of 196 / 297; no Swan11A row)
  - the report's fixed training windows instead of the Friday-before / 13.5-month rule
Everything else is inherited from config.py.
"""
import copy

from config import *  # noqa: F401,F403
from config import EVENTS as _EVENTS

T1_METHOD = "weighted"
MARKETDD_FILE = "data/archive/MarketDD_before_signature_fix.xlsx"
SWAN_ORDER = ["Swan1", "Swan2", "Swan3", "Swan4", "Swan5",
              "Swan6", "Swan7", "Swan8", "Swan9", "Swan10D"]      # archive has no Swan11A row
EXTRA_BETA_COLUMNS = {"Swan10D": "S10D"}

EVENTS = copy.deepcopy(_EVENTS)
for _name, (_start, _end) in {
    "Swan 8":   ("2017-07-22", "2018-09-28"),
    "Swan 9":   ("2019-01-01", "2020-02-15"),
    "Swan 10D": ("2020-11-16", "2021-12-31"),
    "Swan 11A": ("2023-10-31", "2025-01-31"),
}.items():
    EVENTS[_name].update(train_start=_start, train_end=_end)
