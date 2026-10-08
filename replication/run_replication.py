"""Run a guide-compliant reference combination, full sweep, or checkpoint export."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path

# Small matrices are faster and more reproducible with one BLAS/OpenMP thread.
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

from mvo_algo import EVENTS, GRID, Replication, versions, write_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="mvo4_sweep directory containing data and price caches")
    parser.add_argument("--output", type=Path, default=Path("results") / datetime.now().strftime("replication_%Y%m%d_%H%M%S"))
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent / ".cache/dbi")
    parser.add_argument("--full", action="store_true", help="all 5,376 combinations")
    parser.add_argument("--events", nargs="+", choices=list(EVENTS), default=list(EVENTS))
    parser.add_argument("--K", type=int, nargs="+", default=[7])
    parser.add_argument("--Seeds", type=int, nargs="+", default=[3])
    parser.add_argument("--MinAppear", type=int, nargs="+", default=[3])
    parser.add_argument("--w1", type=float, nargs="+", default=[0.17])
    parser.add_argument("--checkpoints", action="store_true", help="export one combination across all four events")
    args = parser.parse_args()
    if args.full and args.events != list(EVENTS):
        parser.error("--full runs all four events; omit --events")
    if args.full and any(getattr(args, p) != default for p, default in
                         [("K", [7]), ("Seeds", [3]), ("MinAppear", [3]), ("w1", [0.17])]):
        parser.error("--full uses the guide's fixed grid; omit parameter overrides")
    if args.checkpoints and (args.full or any(len(getattr(args, p)) != 1 for p in GRID)):
        parser.error("--checkpoints requires exactly one combination and no --full")
    args.output.mkdir(parents=True, exist_ok=False)
    model = Replication(args.source, args.cache)
    grid = GRID if args.full else {p: getattr(args, p) for p in GRID}
    raw = model.run_sweep(grid, args.events)
    tables = write_outputs(raw, args.output, args.events)
    metadata = dict(guide="REPLICATION_GUIDE.md", source=str(model.source), grid=grid,
                    events=args.events, environment=versions(), runs=len(raw),
                    solved=int(raw.status.eq("ok").sum()), complete_combos=len(tables["all_combos"]),
                    windows={event: dict(start=str(model.training_window(event)[0].date()),
                                         end=str(model.training_window(event)[1].date()),
                                         extended=model.training_window(event)[2]) for event in args.events})
    (args.output / "run_info.json").write_text(json.dumps(metadata, indent=2))
    if args.checkpoints:
        model.export_checkpoints(args.output / "replication_checkpoints.xlsx", **{p: grid[p][0] for p in grid})
    print(f"{len(raw)} runs; {metadata['solved']} solved; {metadata['complete_combos']} complete combinations")
    print(tables["top_worst_case"].head(5).to_string(index=False))
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
