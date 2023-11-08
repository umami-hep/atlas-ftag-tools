from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

import ftag


@dataclass
class H5Writer:
    """Writes jets to an HDF5 file.

    Parameters
    ----------
    dst : Path | str
        Path to the output file.
    dtypes : dict[str, np.dtype]
        Dictionary of group names and their corresponding dtypes.
    num_jets : int
        Number of jets to write.
    shapes : dict[str, int], optional
        Dictionary of group names and their corresponding shapes.
    jets_name : str, optional
        Name of the jets group. Default is "jets".
    add_flavour_label : bool, optional
        Whether to add a flavour label to the jets group. Default is False.
    compression : str, optional
        Compression algorithm to use. Default is "lzf".
    precision : str | None, optional
        Precision to use. Default is None.
    shuffle : bool, optional
        Whether to shuffle the jets before writing. Default is True.
    """

    dst: Path | str
    dtypes: dict[str, np.dtype]
    shapes: dict[str, tuple[int, ...]]
    jets_name: str = "jets"
    add_flavour_label: bool = False
    compression: str = "lzf"
    precision: str | None = None
    shuffle: bool = True

    def __post_init__(self):
        self.num_written = 0
        self.rng = np.random.default_rng(42)
        self.num_jets = [shape[0] for shape in self.shapes.values()]
        assert len(set(self.num_jets)) == 1, "Must have same number of jets per group"
        self.num_jets = self.num_jets[0]

        self.dst = Path(self.dst)
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.dst, "w")
        self.add_attr("writer_version", ftag.__version__)

        for name, dtype in self.dtypes.items():
            self.create_ds(name, dtype)

    @classmethod
    def from_file(cls, source: Path, num_jets: int | None = None, **kwargs) -> H5Writer:
        with h5py.File(source, "r") as f:
            dtypes = {name: ds.dtype for name, ds in f.items()}
            shapes = {name: ds.shape for name, ds in f.items()}
            if num_jets is not None:
                shapes = {name: (num_jets,) + shape[1:] for name, shape in shapes.items()}
            compression = [ds.compression for ds in f.values()]
            assert len(set(compression)) == 1, "Must have same compression for all groups"
            compression = compression[0]
            if compression not in kwargs:
                kwargs["compression"] = compression
        return cls(dtypes=dtypes, shapes=shapes, **kwargs)

    def create_ds(self, name: str, dtype: np.dtype) -> None:
        if name == self.jets_name and self.add_flavour_label:
            dtype = np.dtype(dtype.descr + [("flavour_label", "i4")])

        # optimal chunking is around 100 jets, only aply for track groups
        shape = self.shapes[name]
        chunks = (100,) + shape[1:] if shape[1:] else None

        # note: enabling the hd5 shuffle filter doesn't improve anything
        self.file.create_dataset(
            name, dtype=dtype, shape=shape, compression=self.compression, chunks=chunks
        )

    def close(self) -> None:
        with h5py.File(self.dst) as f:
            written = len(f[self.jets_name])
        if self.num_written != written:
            raise ValueError(
                f"Attemped to close file {self.dst} when only {self.num_written:,} out of"
                f" {written:,} jets have been written"
            )
        self.file.close()

    def get_attr(self, name, group=None):
        with h5py.File(self.dst) as f:
            obj = f[group] if group else f
            return obj.attrs[name]

    def add_attr(self, name, data, group=None) -> None:
        obj = self.file[group] if group else self.file
        obj.attrs.create(name, data)

    def copy_attrs(self, fname: Path) -> None:
        with h5py.File(fname) as f:
            for name, value in f.attrs.items():
                self.add_attr(name, value)
            for name, ds in f.items():
                for attr_name, value in ds.attrs.items():
                    self.add_attr(attr_name, value, group=name)

    def write(self, data: dict[str, np.array]) -> None:
        if (total := self.num_written + len(data[self.jets_name])) > self.num_jets:
            raise ValueError(
                f"Attempted to write more jets than expected: {total:,} > {self.num_jets:,}"
            )
        idx = np.arange(len(data[self.jets_name]))
        if self.shuffle:
            self.rng.shuffle(idx)
            data = {name: array[idx] for name, array in data.items()}

        low = self.num_written
        high = low + len(idx)
        for group in self.dtypes:
            self.file[group][low:high] = data[group]
        self.num_written += len(idx)
