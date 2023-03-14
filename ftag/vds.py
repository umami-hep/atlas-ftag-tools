import glob
from pathlib import Path

import h5py


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


def create_virtual_file(pattern: str, out_fname: Path | None = None, overwrite: bool = False):
    # get list of filenames
    fnames = glob.glob(str(pattern))
    if not fnames:
        raise FileNotFoundError(f"No files matched pattern {pattern}")

    # infer output path if not given
    if out_fname is None:
        assert len(set(Path(fname).parent for fname in fnames)) == 1
        out_fname = Path(fnames[0]).parent / "vds" / "vds.h5"

    if not overwrite and out_fname.is_file():
        return out_fname

    # create virtual file
    out_fname.parent.mkdir(exist_ok=True)
    with h5py.File(out_fname, "w") as f:
        for group in h5py.File(fnames[0]):
            layout = get_virtual_layout(fnames, group)
            f.create_virtual_dataset(group, layout)

    return out_fname
