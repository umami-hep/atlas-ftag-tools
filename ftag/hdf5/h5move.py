"""A script to move (i.e. rename) a dataset in an h5 file."""

from __future__ import annotations

import argparse

import h5py

from ftag.cli_utils import HelpFormatter, valid_path


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument("--fname", required=True, type=valid_path, help="path to h5 file")
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


def main(args=None) -> None:
    args = parse_args(args)
    print(f"Moving {args.src} to {args.dst} in {args.fname}")
    f = h5py.File(args.fname, "a")
    f.move(args.src, args.dst)
    f.close()
