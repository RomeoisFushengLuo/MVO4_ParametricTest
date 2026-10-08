"""
Parity check: does the sweep engine reproduce the original mvo4.ipynb logic?

For each event it runs
  (a) mvo4_engine.run_one           -- what the sweep uses, and
  (b) reference/mvo4_notebook_reference.run_notebook_logic
                                    -- the notebook's own cells, copied verbatim
on the same prices and data files, and compares every intermediate: the
1,000-seed DBI ranking, the preselected stocks, the crash curve (t1, t2, dd),
each stock's drawdown, E_2, the covariance matrix, the final weights and the PR.

The reference recomputes the full 1,000-seed K-means ranking every time (as the
notebook does), so each event takes about a minute.

    python parity_check.py                      # all events, notebook's own settings
    python parity_check.py --events "Swan 9" --K 7 --Seeds 3 --MinAppear 3 --w1 0.17
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "reference"))

import config as C  # noqa: E402
import mvo4_engine as E  # noqa: E402
from mvo4_notebook_reference import run_notebook_logic  # noqa: E402

TOL = 1e-10


def _maxdiff(a, b):
    a, b = a.astype(float), b.astype(float)
    return float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))) if a.size else 0.0


def compare(event, K=7, Seeds=5, MinAppear=3, w1=0.17):
    """One row per check: what the engine got, what the notebook code got, and whether they match.
    Runs the engine with T1_METHOD = "weighted", because that is the t1 line the notebook
    executes (config's default "average" is the notebook's first, overwritten line)."""
    saved = C.T1_METHOD
    C.T1_METHOD = "weighted"
    try:
        return _compare(event, K, Seeds, MinAppear, w1)
    finally:
        C.T1_METHOD = saved


def _compare(event, K, Seeds, MinAppear, w1):
    ev = C.EVENTS[event]
    tr = ev["train_swans"]
    i0 = C.SWAN_ORDER.index(tr[0])
    if C.SWAN_ORDER[i0: i0 + len(tr)] != tr:
        raise ValueError(f"{event}: train_swans must be consecutive in SWAN_ORDER for the notebook's label slicing")

    eng = E.run_one(event, K, Seeds, MinAppear, w1, detail=True)
    start, end, _ = E.training_window(event)
    px, bm = E.prices(event)
    ref = run_notebook_logic(
        portfolio_beta=E.beta_table()[C.SWAN_ORDER].copy(), market_drop_info=E.market_dd(),
        optdata_path=E._path(C.OPTDATA_DIR) / ev["optdata"], px_all=px, oex_all=bm,
        start=start, end=end, peak=E.ts(ev["peak"]), bottom=E.ts(ev["bottom"]),
        start_swan=tr[0], end_swan=tr[-1], N=K, Seeds=Seeds, Min_appear=MinAppear, w1=w1,
        selected_stock_number=C.MAX_PRESELECTED, start_swan_date=E.ts(ev.get("buy_date", ev["peak"])))

    ri = eng["risk"]
    rows = []

    def add(check, engine_val, notebook_val, ok):
        rows.append(dict(event=event, check=check, engine=engine_val, notebook=notebook_val, match=bool(ok)))

    eng_rank = E.dbi_ranking(tr, K)
    add("DBI seed ranking (all 1,000)", f"top {Seeds}: {[s for s, _ in eng_rank[:Seeds]]}",
        f"top {Seeds}: {[s for s, _ in ref['dbi_ranking'][:Seeds]]}",
        [s for s, _ in eng_rank] == [s for s, _ in ref["dbi_ranking"]]
        and np.allclose([d for _, d in eng_rank], [d for _, d in ref["dbi_ranking"]], atol=TOL))
    pre = list(eng["preselected"].index)
    add("preselected stocks", f"{len(pre)}: {', '.join(pre)}", f"{len(ref['preselected'])}: {', '.join(ref['preselected'])}",
        pre == ref["preselected"])
    add("crash curve t1 / t2 / dd", f"{ri['t1']} / {ri['t2']} / {ri['dd']:.6f}",
        f"{ref['t1']} / {ref['t2']} / {ref['dd']:.6f}",
        ri["t1"] == ref["t1"] and ri["t2"] == ref["t2"] and abs(ri["dd"] - ref["dd"]) < TOL)
    add("training days in window", ri["total_day"], ref["total_day"], ri["total_day"] == ref["total_day"])
    idx = ri["fall"].index
    d = _maxdiff(ri["fall"], ref["fall"].reindex(idx))
    add("max drawdown per stock (fall)", f"{len(idx)} stocks", f"max |diff| {d:.2e}", d < TOL)
    d = _maxdiff(ri["E_2"], ref["E_2"].reindex(idx))
    add("E_2 = R_fwd / fall", f"{len(idx)} stocks", f"max |diff| {d:.2e}", d < TOL)
    d = _maxdiff(ri["cov"], ref["cov"].loc[idx, idx])
    add("covariance matrix", f"{len(idx)}x{len(idx)}", f"max |diff| {d:.2e}", d < TOL)
    ew, rw = eng.get("weights", pd.Series(dtype=float)), ref["weights"]
    same_names = sorted(ew.index) == sorted(rw.index)
    d = _maxdiff(ew.sort_index(), rw.sort_index()) if same_names else float("nan")
    add("final weights", ", ".join(f"{t} {v:.1%}" for t, v in ew.sort_values(ascending=False).items()),
        ", ".join(f"{t} {v:.1%}" for t, v in rw.sort_values(ascending=False).items()), same_names and d < 1e-8)
    add("PR", f"{eng.get('PR', float('nan')):.6%}", f"{ref['PR']:.6%}", abs(eng.get("PR", np.nan) - ref["PR"]) < 1e-8)
    return pd.DataFrame(rows)


def run(events=None, **params):
    events = events or C.EVENTS_TO_RUN
    out = []
    for e in events:
        t = time.time()
        r = compare(e, **params)
        out.append(r)
        print(f"{e}: {'ALL MATCH' if r.match.all() else 'MISMATCH'} ({time.time() - t:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", nargs="*")
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--Seeds", type=int, default=5)
    ap.add_argument("--MinAppear", type=int, default=3)
    ap.add_argument("--w1", type=float, default=0.17)
    a = ap.parse_args()
    res = run(a.events, K=a.K, Seeds=a.Seeds, MinAppear=a.MinAppear, w1=a.w1)
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(res.to_string(index=False))
    print("\nRESULT:", "every check matches" if res.match.all() else f"{(~res.match).sum()} check(s) differ")
    sys.exit(0 if res.match.all() else 1)
