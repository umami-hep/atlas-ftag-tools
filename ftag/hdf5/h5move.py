"""A script to move (i.e. rename) a dataset in an h5 file."""

from __future__ import annotations

import argparse
from typing import Any

import h5py

from ftag.cli_utils import HelpFormatter, valid_path


def parse_args(args: Any | None) -> argparse.Namespace:
    """Parse command line arguments.

    Parameters
    ----------
    args : Any | None
        Command line arguments

    Returns
    -------
    argparse.Namespace
        Namespace with the parsed command line arguments.
    """
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


def main(args: Any | None = None) -> None:
    """Run VDS creation.

    Parameters
    ----------
    args : Any | None, optional
        Command line arguments, by default None
    """
    args = parse_args(args)
    print(f"Moving {args.src} to {args.dst} in {args.fname}")
    f = h5py.File(args.fname, "a")
    f.move(args.src, args.dst)
    f.close()
