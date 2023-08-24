"""A script to move (i.e. rename) a dataset in an h5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fname", required=True, type=Path, help="path to h5 file")
    parser.add_argument(
        "--src",
        required=True,
        type=str,
        help="path within h5 file to source dataset, starts with '/'",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=str,
        help="path within h5 file to destination dataset, starts with '/'",
    )
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    print(f"Moving {args.src} to {args.dst} in {args.fname}")
    f = h5py.File(args.fname, "a")
    f.move(args.src, args.dst)
    f.close()
