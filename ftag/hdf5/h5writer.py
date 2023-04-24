from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

import ftag
from ftag.hdf5 import get_dtype


@dataclass
class H5Writer:
    src: Path | str
    dst: Path | str
    variables: dict
    num_jets: int
    jets_name: str = "jets"
    add_flavour_label: bool = False
    compression: str = "lzf"
    precision: str | None = None
    shuffle: bool = True
    num_written: int = 0
    rng = np.random.default_rng(42)

    def __post_init__(self):
        self.src = Path(self.src)
        self.dst = Path(self.dst)
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.dst, "w")
        self.add_attr("srcfile", str(self.src))
        self.add_attr("writer_version", ftag.__version__)
        for name, var in self.variables.items():
            self.create_ds(name, var)

    def create_ds(self, name: str, variables: list[str]) -> None:
        with h5py.File(self.src) as f:
            dtype = get_dtype(f[name], variables, self.precision)
            if name == self.jets_name and self.add_flavour_label:
                dtype = np.dtype(dtype.descr + [("flavour_label", "i4")])
            num_tracks = f[name].shape[1:]
            shape = (self.num_jets,) + num_tracks

        # optimal chunking is around 100 jets
        chunks = (100,) + num_tracks if num_tracks else None

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
        for n in self.variables:
            self.file[n][low:high] = data[n]
        self.num_written += len(idx)
