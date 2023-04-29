from __future__ import annotations

import glob
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import h5py


def filter_events(
    fname: str, group: str, filter_fraction: float, filtering_var: str
) -> Tuple[np.ndarray, int]:
    with h5py.File(fname, "r") as f:
        filtering_var = f[group][filtering_var][:]
        num_total_var = len(filtering_var)

        mask = np.random.rand(num_total_var) < filter_fraction
        filtered_indices = np.where(mask)[0]

    return filtered_indices, num_total_var


def get_virtual_layout(
    fnames: list[str], group: str, event_ratio: Tuple[int, int], filtering_var: Optional[str]
):
    sources = []
    total = 0
    for fname in fnames:
        if filtering_var:
            indices, num_total_events = filter_events(fname, group, event_ratio, filtering_var)
            with h5py.File(fname) as f:
                vsource = h5py.VirtualSource(f[group], shape=(len(indices),), sel=indices)
        else:
            with h5py.File(fname) as f:
                vsource = h5py.VirtualSource(f[group])
                num_total_events = vsource.shape[0]

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


def create_virtual_file(
    pattern: Path | str,
    out_fname: Path | None = None,
    overwrite: bool = False,
    event_ratio: Tuple[int, int] = (5, 1),
    filtering_var: Optional[str] = None,
):
    # get list of filenames
    fnames = glob.glob(str(pattern))
    if not fnames:
        raise FileNotFoundError(f"No files matched pattern {pattern}")

    # infer output path if not given
    if out_fname is None:
        assert len(set(Path(fname).parent for fname in fnames)) == 1
        out_fname = Path(fnames[0]).parent / "vds" / "vds.h5"
    else:
        out_fname = Path(out_fname)

    # check if file already exists
    if not overwrite and out_fname.is_file():
        return out_fname

    # create virtual file
    out_fname.parent.mkdir(exist_ok=True)
    with h5py.File(out_fname, "w") as f:
        for group in h5py.File(fnames[0]):
            layout = get_virtual_layout(fnames, group, event_ratio, filtering_var)
            f.create_virtual_dataset(group, layout)

    return out_fname


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a lightweight wrapper around a set of h5 files"
    )
    parser.add_argument("pattern", type=Path, help="quotes-enclosed glob pattern of files to merge")
    parser.add_argument("output", type=Path, help="path to output virtual file")
    parser.add_argument(
        "--filtering-var",
        default=None,
        help="variable to use for event filtering (None for no filtering)",
    )
    args = parser.parse_args()

    print(f"Globbing {args.pattern}...")
    create_virtual_file(
        args.pattern,
        args.output,
        overwrite=True,
        event_ratio=(5, 1),
        filtering_var=args.filtering_var,
    )
    with h5py.File(args.output) as f:
        key = list(f.keys())[0]
        num = len(f[key])
    print(f"Virtual dataset '{key}' has {num:,} entries")
    print(f"Saved virtual file to {args.output.resolve()}")


if __name__ == "__main__":
    main()
