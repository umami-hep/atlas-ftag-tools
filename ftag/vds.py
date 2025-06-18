from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Create a lightweight HDF5 wrapper (virtual datasets + "
        "summed cutBookkeeper counts) around a set of .h5 files"
    )
    parser.add_argument(
        "pattern",
        type=Path,
        help="quotes-enclosed glob pattern of files to merge, "
        "or a regex if --use_regex is given",
    )
    parser.add_argument("output", type=Path, help="path to output virtual file")
    parser.add_argument(
        "--use_regex",
        action="store_true",
        help="treat PATTERN as a regular expression instead of a glob",
    )
    parser.add_argument(
        "--regex_path",
        type=str,
        required="--use_regex" in (args or sys.argv),
        default=None,
        help="directory whose entries the regex is applied to "
        "(defaults to the current working directory)",
    )
    return parser.parse_args(args)


def get_virtual_layout(fnames: list[str], group: str) -> h5py.VirtualLayout:
    """Concatenate group from multiple files into a single VirtualDataset.

    Parameters
    ----------
    fnames : list[str]
        List with the file names
    group : str
        Name of the group that is concatenated

    Returns
    -------
    h5py.VirtualLayout
        Virtual layout of the new virtual dataset
    """
    sources = []
    total = 0

    # Loop over the input files
    for fname in fnames:
        with h5py.File(fname, "r") as f:
            # Get the file and append its length
            vsrc = h5py.VirtualSource(f[group])
            total += vsrc.shape[0]
            sources.append(vsrc)

    # Define the layout of the output vds
    with h5py.File(fnames[0], "r") as f:
        dtype = f[group].dtype
        shape = f[group].shape

    # Update the shape finalize the output layout
    shape = (total,) + shape[1:]
    layout = h5py.VirtualLayout(shape=shape, dtype=dtype)

    # Fill the vds
    idx = 0
    for vsrc in sources:
        length = vsrc.shape[0]
        layout[idx : idx + length] = vsrc
        idx += length

    return layout


def glob_re(pattern: str | None, regex_path: str | None) -> list[str] | None:
    """Return list of filenames that match REGEX pattern inside regex_path.

    Parameters
    ----------
    pattern : str
        Pattern for the input files
    regex_path : str
        Regex path for the input files

    Returns
    -------
    list[str]
        List of the file basenames that matched the regex pattern
    """
    if pattern is None or regex_path is None:
        return None

    return list(filter(re.compile(pattern).match, os.listdir(regex_path)))


def regex_files_from_dir(
    reg_matched_fnames: list[str] | None,
    regex_path: str | None,
) -> list[str] | None:
    """Turn a list of basenames into full paths; dive into sub-dirs if needed.

    Parameters
    ----------
    reg_matched_fnames : list[str]
        List of the regex matched file names
    regex_path : str
        Regex path for the input files

    Returns
    -------
    list[str]
        List of file paths (as strings) that matched the regex and any subsequent
        globbing inside matched directories.
    """
    if reg_matched_fnames is None or regex_path is None:
        return None

    parent_dir = regex_path or str(Path.cwd())
    full_paths = [Path(parent_dir) / fname for fname in reg_matched_fnames]
    paths_to_glob = [str(fp / "*.h5") if fp.is_dir() else str(fp) for fp in full_paths]
    nested_fnames = [glob.glob(p) for p in paths_to_glob]
    return sum(nested_fnames, [])


def sum_counts_once(counts: np.ndarray) -> np.ndarray:
    """Reduce the arrays in the counts dataset for one file to a scalar via summation.

    Parameters
    ----------
    counts : np.ndarray
        Array from the h5py dataset (counts) from the cutBookkeeper groups

    Returns
    -------
    np.ndarray
        Array with the summed variables for the file
    """
    dtype = counts.dtype
    summed = np.zeros((), dtype=dtype)
    for field in dtype.names:
        summed[field] = counts[field].sum()
    return summed


def check_subgroups(fnames: list[str], group_name: str = "cutBookkeeper") -> list[str]:
    """Check which subgroups are available for the bookkeeper.

    Find the intersection of sub-group names that have a 'counts' dataset
    in every input file. (Using the intersection makes the script robust
    even if a few files are missing a variation.)

    Parameters
    ----------
    fnames : list[str]
        List of the input files
    group_name : str, optional
        Group name in the h5 files of the bookkeeper, by default "cutBookkeeper"

    Returns
    -------
    set[str]
        Returns the files with common sub-groups

    Raises
    ------
    KeyError
        When a file does not have a bookkeeper
    ValueError
        When no common bookkeeper sub-groups were found
    """
    common: set[str] | None = None
    for fname in fnames:
        with h5py.File(fname, "r") as f:
            if group_name not in f:
                raise KeyError(f"{fname} has no '{group_name}' group")
            these = {
                name
                for name, item in f[group_name].items()
                if isinstance(item, h5py.Group) and "counts" in item
            }
            common = these if common is None else common & these
    if not common:
        raise ValueError("No common cutBookkeeper sub-groups with 'counts' found")
    return sorted(common)


def aggregate_cutbookkeeper(
    fnames: list[str],
    group_name: str = "cutBookkeeper",
) -> dict[str, np.ndarray] | None:
    """Aggregate the cutBookkeeper in the input files.

    For every input file:
    For every sub-group (nominal, sysUp, sysDown, …):
    1. Sum the 4-entry record array inside each file into 1 record
    1. Add those records from all files together into grand total
    Returns a dict  {subgroup_name: scalar-record-array}

    Parameters
    ----------
    fnames : list[str]
        List of the input files

    Returns
    -------
    dict[str, np.ndarray] | None
        Dict with the accumulated cutBookkeeper groups. If the cut bookkeeper
        is not in the files, return None.
    """
    if any(group_name not in h5py.File(f, "r") for f in fnames):
        return None

    subgroups = check_subgroups(fnames, group_name=group_name)

    # initialise an accumulator per subgroup (dtype taken from 1st file)
    accum: dict[str, np.ndarray] = {}
    with h5py.File(fnames[0], "r") as f0:
        for sg in subgroups:
            dtype = f0[f"{group_name}/{sg}/counts"].dtype
            accum[sg] = np.zeros((), dtype=dtype)

    # add each files contribution field-wise
    for fname in fnames:
        with h5py.File(fname, "r") as f:
            for sg in subgroups:
                per_file = sum_counts_once(f[f"{group_name}/{sg}/counts"][()])
                for fld in accum[sg].dtype.names:
                    accum[sg][fld] += per_file[fld]

    return accum


def create_virtual_file(
    pattern: Path | str,
    out_fname: Path | str | None = None,
    use_regex: bool = False,
    regex_path: str | None = None,
    overwrite: bool = False,
    bookkeeper_name: str = "cutBookkeeper",
) -> Path:
    """Create the virtual dataset file for the given inputs.

    Parameters
    ----------
    pattern : Path | str
        Pattern of the input files used. Wildcard is supported
    out_fname : Path | str | None, optional
        Output path to which the virtual dataset file is written. By default None
    use_regex : bool, optional
        If you want to use regex instead of glob, by default False
    regex_path : str | None, optional
        Regex logic used to define the input files, by default None
    overwrite : bool, optional
        Decide, if an existing output file is overwritten, by default False
    bookkeeper_name : str, optional
        Name of the cut bookkeeper in the h5 files.

    Returns
    -------
    Path
        Path object of the path to which the output file is written

    Raises
    ------
    FileNotFoundError
        If not input files were found for the given pattern
    ValueError
        If no output file is given and the input comes from multiple directories
    """
    # Get list of filenames
    pattern_str = str(pattern)

    # Use regex to find input files else use glob
    if use_regex is True:
        matched = glob_re(pattern_str, regex_path)
        fnames = regex_files_from_dir(matched, regex_path)
    else:
        fnames = glob.glob(pattern_str)

    # Throw error if no input files were found
    if not fnames:
        raise FileNotFoundError(f"No files matched pattern {pattern!r}")

    # Infer output path if not given
    if out_fname is None:
        if len({Path(f).parent for f in fnames}) != 1:
            raise ValueError("Give --output when files reside in multiple dirs")
        out_fname = Path(fnames[0]).parent / "vds" / "vds.h5"
    else:
        out_fname = Path(out_fname)

    # If overwrite is not active and a file exists, stop here
    if not overwrite and out_fname.is_file():
        return out_fname

    # Identify common groups across all files
    common_groups: set[str] = set()
    for fname in fnames:
        with h5py.File(fname, "r") as f:
            groups = set(f.keys())
            common_groups = groups if not common_groups else common_groups & groups

    # Ditch the bookkeeper. We will process it separately
    common_groups.discard("cutBookkeeper")

    # Check that the directory of the output file exists
    out_fname.parent.mkdir(parents=True, exist_ok=True)

    # Build the output file
    with h5py.File(out_fname, "w") as fout:
        # Build "standard" groups
        for gname in sorted(common_groups):
            layout = get_virtual_layout(fnames, gname)
            fout.create_virtual_dataset(gname, layout)

            # Copy first-file attributes to VDS root object
            with h5py.File(fnames[0], "r") as f0:
                for k, v in f0[gname].attrs.items():
                    fout[gname].attrs[k] = v

        # Build the cutBookkeeper
        counts_total = aggregate_cutbookkeeper(fnames=fnames, group_name=bookkeeper_name)
        if counts_total is not None:
            for sg, record in counts_total.items():
                grp = fout.require_group(f"{bookkeeper_name}/{sg}")
                grp.create_dataset("counts", data=record, shape=(), dtype=record.dtype)

    return out_fname


def main(args=None) -> None:
    args = parse_args(args)
    matching_mode = "Applying regex to" if args.use_regex else "Globbing"
    print(f"{matching_mode} {args.pattern} ...")
    out_path = create_virtual_file(
        pattern=args.pattern,
        out_fname=args.output,
        use_regex=args.use_regex,
        regex_path=args.regex_path,
        overwrite=True,
    )

    with h5py.File(out_path, "r") as f:
        key = next(iter(f.keys()))
        print(f"Virtual dataset '{key}' has {len(f[key]):,} entries")

    print(f"Saved virtual file to {out_path.resolve()}")


if __name__ == "__main__":
    main()
