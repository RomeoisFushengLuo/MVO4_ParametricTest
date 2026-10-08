"""Export intermediate checkpoints, as requested by REPLICATION_GUIDE.md."""
import argparse
import os
from pathlib import Path

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

from mvo_algo import Replication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent / ".cache/dbi")
    parser.add_argument("--output", type=Path, default=Path("replication_checkpoints.xlsx"))
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--Seeds", type=int, default=3)
    parser.add_argument("--MinAppear", type=int, default=3)
    parser.add_argument("--w1", type=float, default=.17)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new --output path")
    model = Replication(args.source, args.cache)
    model.export_checkpoints(args.output, K=args.K, Seeds=args.Seeds, MinAppear=args.MinAppear, w1=args.w1)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
