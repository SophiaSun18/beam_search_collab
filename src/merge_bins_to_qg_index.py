#!/usr/bin/env python3
"""Merge graph/RaBitQ/sign bins into a QG-style codebook index.

The output layout is the inverse of split_qg_index.py:
  [uint32 entry_point]
  repeated N times:
    raw_vector(float32 dim)
    packed_codes(uint64 degree * padded_dim/64)
    factors(float32 3 * degree)
    neighbor_ids(uint32 degree)
  rotator_signs(float32 padded_dim)
  optional quantizer_tail(int32 quant_tag, int32 bits, int32 nlevels, float32 levels...)
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


def next_pow2(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def read_i32_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError(f"{path} is too small to contain a 2-int header")
    return struct.unpack("<ii", raw)


def fvecs_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        raw = f.read(4)
    if len(raw) != 4:
        raise ValueError(f"{path} is too small to be an fvecs file")
    dim = struct.unpack("<i", raw)[0]
    if dim <= 0:
        raise ValueError(f"{path} has invalid fvecs dimension {dim}")
    record_bytes = (dim + 1) * 4
    size = path.stat().st_size
    if size % record_bytes != 0:
        raise ValueError(
            f"{path} size {size} is not divisible by fvecs record size {record_bytes}"
        )
    return size // record_bytes, dim


def compute_l2_centroid_entry(base: Path, n: int, dim: int, block_rows: int) -> int:
    base_mmap = np.memmap(base, dtype=np.float32, mode="r", shape=(n, dim + 1))
    sums = np.zeros(dim, dtype=np.float64)
    for start in range(0, n, block_rows):
        stop = min(n, start + block_rows)
        block = np.asarray(base_mmap[start:stop, 1:], dtype=np.float32)
        sums += block.sum(axis=0, dtype=np.float64)
    centroid = (sums / float(n)).astype(np.float32)

    best_id = 0
    best_dist = np.float32(np.inf)
    for start in range(0, n, block_rows):
        stop = min(n, start + block_rows)
        block = np.asarray(base_mmap[start:stop, 1:], dtype=np.float32)
        diff = block - centroid
        dists = np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
        local = int(np.argmin(dists))
        local_dist = dists[local]
        if local_dist < best_dist:
            best_dist = local_dist
            best_id = start + local
    return best_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge standalone graph/RaBitQ/sign bins into a QG codebook index."
    )
    parser.add_argument("--base", required=True, type=Path, help="Input base .fvecs")
    parser.add_argument("--graph", required=True, type=Path, help="Input graph bin")
    parser.add_argument("--rabitq", required=True, type=Path, help="Input RaBitQ table bin")
    parser.add_argument("--sign", required=True, type=Path, help="Input sign table bin")
    parser.add_argument("--output", required=True, type=Path, help="Output QG index")
    parser.add_argument(
        "--entry-point",
        type=int,
        default=None,
        help="Entry point to write; default computes nearest point to the base centroid.",
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        default=65536,
        help="Rows per streaming block while computing/writing.",
    )
    parser.add_argument(
        "--quant-tail",
        choices=("none", "rbq", "tbq"),
        default="none",
        help="Append the quantizer descriptor tail expected by newer FlashGANN loaders.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=1,
        choices=(1, 2, 4),
        help="Quantizer bit width to write in the optional tail.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.force:
        print(f"output exists: {args.output} (use --force)", file=sys.stderr)
        return 1
    layout_tokens = ("fastscan", "sequential", "sequential.LUT")
    if not any(token in args.output.name for token in layout_tokens):
        print(
            "warning: output filename does not include a layout token "
            "(fastscan, sequential, or sequential.LUT)",
            file=sys.stderr,
        )
    if args.block_rows <= 0:
        print("--block-rows must be positive", file=sys.stderr)
        return 1

    n, dim = fvecs_shape(args.base)
    padded_dim = next_pow2(dim)
    bitcode_words = padded_dim // 64

    graph_n, degree = read_i32_header(args.graph)
    if graph_n != n:
        print(f"graph n={graph_n} does not match base n={n}", file=sys.stderr)
        return 1
    graph_expected = 8 + n * degree * 4
    if args.graph.stat().st_size != graph_expected:
        print(
            f"graph size mismatch: expected {graph_expected}, got {args.graph.stat().st_size}",
            file=sys.stderr,
        )
        return 1

    code_n, code_width = read_i32_header(args.rabitq)
    code_u64_per_node = degree * bitcode_words
    factor_floats = 3 * degree
    factor_u64_words = (factor_floats * 4 + 7) // 8
    expected_code_width = code_u64_per_node + factor_u64_words
    if code_n != n or code_width != expected_code_width:
        print(
            f"RaBitQ shape mismatch: got {code_n}x{code_width}, "
            f"expected {n}x{expected_code_width}",
            file=sys.stderr,
        )
        return 1

    sign_rows, sign_cols = read_i32_header(args.sign)
    if not ((sign_rows == padded_dim and sign_cols == 1) or (sign_rows == 1 and sign_cols == padded_dim)):
        print(
            f"sign shape mismatch: got {sign_rows}x{sign_cols}, "
            f"expected {padded_dim}x1 or 1x{padded_dim}",
            file=sys.stderr,
        )
        return 1

    tail_levels = []
    if args.quant_tail == "tbq":
        raise NotImplementedError("TBQ tail level tables are not generated by this merger yet")
    tail_bytes = 0 if args.quant_tail == "none" else 12 + len(tail_levels) * 4

    row_offset_floats = dim + (code_u64_per_node * 2) + factor_floats + degree
    expected_output_size = 4 + n * row_offset_floats * 4 + padded_dim * 4 + tail_bytes

    entry_point = args.entry_point
    if entry_point is None:
        print("Computing centroid entry point...")
        entry_point = compute_l2_centroid_entry(args.base, n, dim, args.block_rows)
    if entry_point < 0 or entry_point >= n:
        print(f"entry point {entry_point} is outside [0, {n})", file=sys.stderr)
        return 1

    print(
        f"Merging bins into QG index: n={n}, dim={dim}, padded_dim={padded_dim}, "
        f"degree={degree}, row_offset={row_offset_floats}, entry_point={entry_point}"
    )

    base_mmap = np.memmap(args.base, dtype=np.float32, mode="r", shape=(n, dim + 1))
    graph_mmap = np.memmap(args.graph, dtype=np.uint32, mode="r", offset=8, shape=(n, degree))
    rabitq_mmap = np.memmap(
        args.rabitq, dtype=np.uint64, mode="r", offset=8, shape=(n, code_width)
    )
    sign_mmap = np.memmap(args.sign, dtype=np.float32, mode="r", offset=8, shape=(padded_dim,))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_name(args.output.name + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    with tmp_output.open("wb") as out:
        out.write(struct.pack("<I", entry_point))
        for start in range(0, n, args.block_rows):
            stop = min(n, start + args.block_rows)
            rows = stop - start
            merged = np.empty((rows, row_offset_floats), dtype=np.float32)
            merged[:, :dim] = base_mmap[start:stop, 1:]
            code_floats = rabitq_mmap[start:stop, :code_u64_per_node].view(np.float32)
            merged[:, dim : dim + code_u64_per_node * 2] = code_floats.reshape(
                rows, code_u64_per_node * 2
            )
            factor_start = dim + code_u64_per_node * 2
            factor_region = rabitq_mmap[start:stop, code_u64_per_node:].view(np.float32)
            merged[:, factor_start : factor_start + factor_floats] = factor_region.reshape(
                rows, factor_u64_words * 2
            )[:, :factor_floats]
            neighbor_start = factor_start + factor_floats
            neighbor_floats = graph_mmap[start:stop].view(np.float32)
            merged[:, neighbor_start : neighbor_start + degree] = neighbor_floats
            merged.tofile(out)
        sign_mmap.tofile(out)
        if args.quant_tail != "none":
            quant_tag = 0 if args.quant_tail == "rbq" else 1
            out.write(struct.pack("<iii", quant_tag, args.bits, len(tail_levels)))
            if tail_levels:
                np.asarray(tail_levels, dtype=np.float32).tofile(out)

    actual_size = tmp_output.stat().st_size
    if actual_size != expected_output_size:
        tmp_output.unlink(missing_ok=True)
        print(
            f"output size mismatch: expected {expected_output_size}, got {actual_size}",
            file=sys.stderr,
        )
        return 1
    tmp_output.replace(args.output)
    print(f"Wrote {args.output} ({actual_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
