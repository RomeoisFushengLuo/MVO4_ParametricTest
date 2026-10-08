"""
Run the MVO4 parameter sweep from the command line.

    python run_sweep.py                                  # full grid + events from config.py
    python run_sweep.py --name my_test                   # results go to results/my_test/
    python run_sweep.py --events "Swan 9" --K 6 7 --w1 0.13 0.17   # override parts of the grid
    python run_sweep.py --config presets/locked_sweep_report.py    # a different settings file

Anything not given on the command line comes from config.py. Output, in
results/<name>/:
    raw_runs.csv        one row per (event, K, Seeds, MinAppear, w1), with PR,
                        stock counts and the final holdings
    summary.xlsx        top combos by mean and worst case, all complete combos,
                        marginal trends, holdings for the top combos, run status
    config_used.py      a copy of config.py as it was for this run
    run_info.json       grid, events, effective training windows, package versions
"""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import mvo4_engine as E  # noqa: E402


def load_config(path):
    """Use a different settings file (e.g. presets/locked_sweep_report.py) instead of config.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sweep_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    E.C = mod
    E.clear_memory()
    return mod


def main():
    global C
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="settings file to use instead of config.py, e.g. presets/locked_sweep_report.py")
    ap.add_argument("--name", help="results folder name (default: timestamp)")
    ap.add_argument("--events", nargs="*", help='event names from config.EVENTS, e.g. "Swan 9"')
    ap.add_argument("--K", nargs="*", type=int)
    ap.add_argument("--Seeds", nargs="*", type=int)
    ap.add_argument("--MinAppear", nargs="*", type=int)
    ap.add_argument("--w1", nargs="*", type=float)
    a = ap.parse_args()
    config_path = Path(a.config).resolve() if a.config else HERE / "config.py"
    if a.config:
        C = load_config(config_path)

    grid = dict(C.GRID)
    for k in grid:
        if getattr(a, k):
            grid[k] = getattr(a, k)
    events = a.events or C.EVENTS_TO_RUN
    unknown = [e for e in events if e not in C.EVENTS]
    if unknown:
        sys.exit(f"Unknown event(s) {unknown}. Defined in config.EVENTS: {list(C.EVENTS)}")

    name = a.name or datetime.datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
    out = HERE / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out / "config_used.py")
    if a.config:
        shutil.copy(HERE / "config.py", out / "config_base.py")   # the preset inherits from this

    windows = {}
    for e in events:
        s, t, ext = E.training_window(e)
        t1, t2, dd = E.curve_params(C.EVENTS[e]["train_swans"])
        windows[e] = dict(train_start=str(s.date()), train_end=str(t.date()), extended_to_fit_curve=ext,
                          peak=C.EVENTS[e]["peak"], bottom=C.EVENTS[e]["bottom"], t1=t1, t2=t2, dd=dd)
        print(f"{e}: training {s.date()} -> {t.date()}{' (extended to fit curve)' if ext else ''}, "
              f"evaluated {C.EVENTS[e]['peak']} -> {C.EVENTS[e]['bottom']}")
    n = len(events) * len(grid["K"]) * len(grid["Seeds"]) * len(grid["MinAppear"]) * len(grid["w1"])
    print(f"running {n} combinations -> {out}")
    (out / "run_info.json").write_text(json.dumps(dict(
        started=datetime.datetime.now().isoformat(timespec="seconds"), config=str(config_path.relative_to(HERE)),
        t1_method=C.T1_METHOD, marketdd_file=C.MARKETDD_FILE, grid=grid, events=events,
        windows=windows, environment=E.environment()), indent=1))

    raw = E.run_sweep(grid, events, out)
    s = E.write_summary(raw, out / "summary.xlsx", events)
    top = s["top_mean"].head(5)
    if len(top):
        cols = ["K", "Seeds", "MinAppear", "w1"] + events + ["mean_PR", "min_PR"]
        print("\ntop 5 by mean PR:\n" + top[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\ntop 5 by worst case:\n" + s["top_worst_case"].head(5)[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out / 'raw_runs.csv'} and {out / 'summary.xlsx'}")


if __name__ == "__main__":
    main()
