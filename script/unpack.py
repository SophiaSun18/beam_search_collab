#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "h5py",
#   "numpy",
# ]
# ///

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def write_fvecs(path: Path, array: np.ndarray) -> None:
    data = np.asarray(array, dtype=np.float32, order="C")
    if data.ndim != 2:
        raise ValueError(f"expected 2D array for {path.name}, got shape {data.shape}")
    dim = np.int32(data.shape[1])
    with path.open("wb") as fout:
        for row in data:
            fout.write(dim.tobytes())
            fout.write(np.ascontiguousarray(row).tobytes())


def write_ivecs(path: Path, array: np.ndarray) -> None:
    data = np.asarray(array, dtype=np.int32, order="C")
    if data.ndim != 2:
        raise ValueError(f"expected 2D array for {path.name}, got shape {data.shape}")
    dim = np.int32(data.shape[1])
    with path.open("wb") as fout:
        for row in data:
            fout.write(dim.tobytes())
            fout.write(np.ascontiguousarray(row).tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export ANN-benchmark style HDF5 datasets into fvecs/ivecs files."
    )
    parser.add_argument("input", type=Path, help="Input HDF5 file")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--base-name", default="base", help="Output prefix for training vectors")
    parser.add_argument("--query-name", default="query", help="Output prefix for query vectors")
    parser.add_argument("--gt-name", default="groundtruth", help="Output prefix for groundtruth ids")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as fin:
        required = ["train", "test", "neighbors"]
        missing = [name for name in required if name not in fin]
        if missing:
            raise KeyError(f"missing datasets in {args.input}: {', '.join(missing)}")

        train = fin["train"][:]
        test = fin["test"][:]
        neighbors = fin["neighbors"][:]

        write_fvecs(args.output_dir / f"{args.base_name}.fvecs", train)
        write_fvecs(args.output_dir / f"{args.query_name}.fvecs", test)
        write_ivecs(args.output_dir / f"{args.gt_name}.ivecs", neighbors)

        if "distances" in fin:
            distances = np.asarray(fin["distances"][:], dtype=np.float32, order="C")
            np.save(args.output_dir / f"{args.gt_name}_distances.npy", distances)

    print(f"wrote {args.output_dir / f'{args.base_name}.fvecs'}")
    print(f"wrote {args.output_dir / f'{args.query_name}.fvecs'}")
    print(f"wrote {args.output_dir / f'{args.gt_name}.ivecs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
