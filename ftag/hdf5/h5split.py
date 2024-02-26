"""A script to split a large h5 file into smaller h5 files along the first index."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from ftag.cli_utils import HelpFormatter
from ftag.hdf5 import H5Reader, H5Writer


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="path to source h5 file")
    parser.add_argument(
        "--dst",
        type=Path,
        help=(
            "output directory contaning split files. by default use a new directory in the same"
            " directory as the source file"
        ),
    )
    parser.add_argument(
        "-n", "--jets_per_file", type=int, default=1_000_000, help="number of jets per output file"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=100_000,
        help="number of jets to read/write at a time",
    )
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    src = args.src
    dst = args.dst
    if dst is None:
        dst = src.parent / f"split_{src.stem}"
    jets_per_file = args.jets_per_file

    print(f"\nSplitting: {src}")
    print(f"Destination: {dst}")
    with h5py.File(src, "r") as f:
        total_jets = next(iter(f.values())).shape[0]

    num_full_files = total_jets // jets_per_file
    remainder = total_jets % jets_per_file
    num_total_files = num_full_files + (1 if remainder != 0 else 0)
    print(f"\n{total_jets:,} jets will be split across {num_total_files:,} files\n")

    reader = H5Reader(src, batch_size=args.batch_size, shuffle=False)
    variables = dict.fromkeys(reader.dtypes().keys())
    for i in range(num_total_files):
        start = i * jets_per_file
        out = dst / f"{src.stem}-split_{i}.h5"
        num = jets_per_file if i < num_full_files else remainder
        writer = H5Writer.from_file(src, dst=out, num_jets=num, shuffle=False)
        for batch in reader.stream(variables=variables, num_jets=num, start=start):
            writer.write(batch)
            total_written = start + writer.num_written
            pct_done = total_written / total_jets
            print(f"\rProcessed {total_written:,}/{total_jets:,} jets ({pct_done:.1%})", end="")
        writer.copy_attrs(src)
        writer.close()
    print("\nDone!\n")


if __name__ == "__main__":
    main()
