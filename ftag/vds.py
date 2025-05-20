from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import h5py


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Create a lightweight wrapper around a set of h5 files"
    )
    parser.add_argument("pattern", type=Path, help="quotes-enclosed glob pattern of files to merge")
    parser.add_argument("output", type=Path, help="path to output virtual file")
    parser.add_argument("--use_regex", help="if provided pattern is a regex", action="store_true")
    parser.add_argument("--regex_path", type=str, required="--regex" in sys.argv, default=None)
    return parser.parse_args(args)


def get_virtual_layout(fnames: list[str], group: str):
    # get sources
    sources = []
    total = 0
    for fname in fnames:
        with h5py.File(fname) as f:
            vsource = h5py.VirtualSource(f[group])
            total += vsource.shape[0]
            sources.append(vsource)

    # define layout of the vds
    with h5py.File(fnames[0]) as f:
        dtype = f[group].dtype
        shape = f[group].shape
    shape = (total,) + shape[1:]
    layout = h5py.VirtualLayout(shape=shape, dtype=dtype)

    # fill the vds
    idx = 0
    for source in sources:
        length = source.shape[0]
        layout[idx : idx + length] = source
        idx += length

    return layout


def glob_re(pattern, regex_path):
    return list(filter(re.compile(pattern).match, os.listdir(regex_path)))


def regex_files_from_dir(reg_matched_fnames, regex_path):
    parent_dir = regex_path or str(Path.cwd())
    full_paths = [parent_dir + "/" + fname for fname in reg_matched_fnames]
    paths_to_glob = [fname + "/*.h5" if Path(fname).is_dir() else fname for fname in full_paths]
    nested_fnames = [glob.glob(fname) for fname in paths_to_glob]
    return sum(nested_fnames, [])


def create_virtual_file(
    pattern: Path | str,
    out_fname: Path | None = None,
    use_regex: bool = False,
    regex_path: str | None = None,
    overwrite: bool = False,
):
    # get list of filenames
    pattern_str = str(pattern)
    if use_regex:
        reg_matched_fnames = glob_re(pattern_str, regex_path)
        print("reg matched fnames: ", reg_matched_fnames)
        fnames = regex_files_from_dir(reg_matched_fnames, regex_path)
    else:
        fnames = glob.glob(pattern_str)
    if not fnames:
        raise FileNotFoundError(f"No files matched pattern {pattern}")
    print("Files to merge to vds: ", fnames)

    # infer output path if not given
    if out_fname is None:
        assert len({Path(fname).parent for fname in fnames}) == 1
        out_fname = Path(fnames[0]).parent / "vds" / "vds.h5"
    else:
        out_fname = Path(out_fname)

    # check if file already exists
    if not overwrite and out_fname.is_file():
        return out_fname

    # identify common groups across all files
    common_groups: set[str] = set()
    for fname in fnames:
        with h5py.File(fname) as f:
            groups = set(f.keys())
            common_groups = groups if not common_groups else common_groups.intersection(groups)

    if not common_groups:
        raise ValueError("No common groups found across files")

    # create virtual file
    out_fname.parent.mkdir(exist_ok=True)
    with h5py.File(out_fname, "w") as f:
        for group in common_groups:
            layout = get_virtual_layout(fnames, group)
            f.create_virtual_dataset(group, layout)
            attrs_dict: dict = {}
            for fname in fnames:
                with h5py.File(fname) as g:
                    for name, value in g[group].attrs.items():
                        if name not in attrs_dict:
                            attrs_dict[name] = []
                        attrs_dict[name].append(value)
            for name, value in attrs_dict.items():
                if len(value) > 0:
                    f[group].attrs[name] = value[0]

    return out_fname


def main(args=None) -> None:
    args = parse_args(args)
    matching_mode = "Applying regex to" if args.use_regex else "Globbing"
    print(f"{matching_mode} {args.pattern}...")
    create_virtual_file(
        args.pattern,
        args.output,
        use_regex=args.use_regex,
        regex_path=args.regex_path,
        overwrite=True,
    )
    with h5py.File(args.output) as f:
        key = next(iter(f.keys()))
        num = len(f[key])
    print(f"Virtual dataset '{key}' has {num:,} entries")
    print(f"Saved virtual file to {args.output.resolve()}")


if __name__ == "__main__":
    main()
