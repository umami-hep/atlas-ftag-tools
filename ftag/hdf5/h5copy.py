"""A script to copy a dataset from one h5 file to another."""

from __future__ import annotations

import argparse

import h5py

from ftag.cli_utils import HelpFormatter, valid_path


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument(
        "--src_fname", required=True, type=valid_path, help="path to the source h5 file"
    )
    parser.add_argument(
        "--src",
        required=True,
        type=str,
        help="path within source h5 file to dataset, starts with '/'",
    )
    parser.add_argument(
        "--dst_fname",
        required=True,
        type=str,
        help="path to the destination h5 file, which will be created if it does not exist",
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="path within destination h5 file to paste dataset, starts with '/' (defaults to src)",
    )
    return parser.parse_args(args)


def copy_dataset(src_fname, src_path, dst_fname, dst_path):
    # Open the source file in read mode
    with h5py.File(src_fname, "r") as src_file, h5py.File(dst_fname, "a") as dst_file:
        if src_path not in src_file:
            raise KeyError(f"Dataset {src_path} not found in {src_fname}")

        # Get the source dataset
        src_data = src_file[src_path]

        # Check if the destination dataset exists
        if dst_path in dst_file:
            raise FileExistsError(f"Destination dataset {dst_path} already exists in {dst_fname}")

        # Copy the dataset to the destination
        src_file.copy(src_data, dst_file, dst_path)
        print(f"Copied dataset {src_path} from {src_fname} to {dst_path} in {dst_fname}")


def main(args=None) -> None:
    args = parse_args(args)
    dst = args.dst or args.src
    copy_dataset(args.src_fname, args.src, args.dst_fname, dst)


if __name__ == "__main__":
    main()
