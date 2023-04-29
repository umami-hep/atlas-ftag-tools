from __future__ import annotations

import glob
from pathlib import Path

import h5py
import numpy as np


def filter_events(
    fname: str, group: str, filter_fraction: float, filtering_var: str | None
) -> np.ndarray:
    with h5py.File(fname, "r") as f:
        filtering_var = f[group][filtering_var][:]
        num_total_var = len(filtering_var)

        mask = np.random.rand(num_total_var) < filter_fraction
        filtered_indices = np.where(mask)[0]

    return filtered_indices


def get_filtered_chunks(
    fnames: list[str], group: str, filter_fraction: float, filtering_var: str | None
):
    filtered_chunks = []
    for fname in fnames:
        if filtering_var:
            print("Filtering events")
            indices = filter_events(fname, group, filter_fraction, filtering_var)
            print("Got indices")
        else:
            with h5py.File(fname, "r") as f:
                indices = np.arange(f[group].shape[0])

        filtered_chunks.append((fname, indices))

    return filtered_chunks


def get_contiguous_chunks(indices: np.ndarray) -> list[slice]:
    contiguous_chunks = []
    start = indices[0]
    end = start
    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end += 1
        else:
            contiguous_chunks.append(slice(start, end + 1))
            start = indices[i]
            end = start
    contiguous_chunks.append(slice(start, end + 1))
    return contiguous_chunks


def create_virtual_dataset(
    fnames: list[str], group: str, filter_fraction: float, filtering_var: str | None
):
    filtered_chunks = get_filtered_chunks(fnames, group, filter_fraction, filtering_var)

    sources = []
    total = 0
    for fname, indices in filtered_chunks:
        print(f"Adding {len(indices)} events from {fname}")
        with h5py.File(fname, "r") as f:
            src = h5py.VirtualSource(f[group])
            contiguous_chunks = get_contiguous_chunks(indices)
            for chunk in contiguous_chunks:
                sources.append(src[chunk])
                total += chunk.stop - chunk.start

    with h5py.File(fnames[0], "r") as f:
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
    filter_fraction: float = 0.2,
    filtering_var: str | None = None,
):
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
            layout = create_virtual_dataset(fnames, group, filter_fraction, filtering_var)
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
    parser.add_argument(
        "--filter-fraction",
        type=float,
        default=0.2,
        help="fraction of events to keep when filtering",
    )
    args = parser.parse_args()

    print(f"Globbing {args.pattern}...")
    create_virtual_file(
        args.pattern,
        args.output,
        overwrite=True,
        filter_fraction=args.filter_fraction,
        filtering_var=args.filtering_var,
    )
    with h5py.File(args.output) as f:
        key = list(f.keys())[0]
        num = len(f[key])
    print(f"Virtual dataset '{key}' has {num:,} entries")
    print(f"Saved virtual file to {args.output.resolve()}")


if __name__ == "__main__":
    main()
