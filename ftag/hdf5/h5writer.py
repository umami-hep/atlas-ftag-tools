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
    full_precision_vars : list[str] | None, optional
        List of variables to store in full precision. Default is None.
    shuffle : bool, optional
        Whether to shuffle the jets before writing. Default is True.

    """

    dst: Path | str
    dtypes: dict[str, np.dtype]
    shapes: dict[str, tuple[int, ...]]
    jets_name: str = "jets"
    add_flavour_label: bool = False
    compression: str = "lzf"
    precision: str = "full"
    full_precision_vars: list[str] | None = None
    shuffle: bool = True
    num_jets: int | None = None  # Allow dynamic mode by defaulting to None

    def __post_init__(self):
        self.num_written = 0
        self.rng = np.random.default_rng(42)

        # Infer number of jets from shapes if not explicitly passed
        inferred_num_jets = [shape[0] for shape in self.shapes.values()]
        if self.num_jets is None:
            assert len(set(inferred_num_jets)) == 1, "Shapes must agree in first dimension"
            self.fixed_mode = False
        else:
            self.fixed_mode = True
            for name in self.shapes:
                self.shapes[name] = (self.num_jets,) + self.shapes[name][1:]

        if self.precision == "full":
            self.fp_dtype = np.float32
        elif self.precision == "half":
            self.fp_dtype = np.float16
        elif self.precision is None:
            self.fp_dtype = None
        else:
            raise ValueError(f"Invalid precision: {self.precision}")

        self.dst = Path(self.dst)
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.dst, "w")
        self.add_attr("writer_version", ftag.__version__)

        for name, dtype in self.dtypes.items():
            self.create_ds(name, dtype)

    @classmethod
    def from_file(
        cls, source: Path, num_jets: int | None = 0, variables=None, **kwargs
    ) -> H5Writer:
        with h5py.File(source, "r") as f:
            dtypes = {name: ds.dtype for name, ds in f.items()}
            shapes = {name: ds.shape for name, ds in f.items()}

            if variables:
                new_dtye = {}
                new_shape = {}
                for name, ds in f.items():
                    if name not in variables:
                        continue
                    new_dtye[name] = ftag.hdf5.get_dtype(
                        ds,
                        variables=variables[name],
                        precision=kwargs.get("precision"),
                        full_precision_vars=kwargs.get("full_precision_vars"),
                    )
                    new_shape[name] = ds.shape
                dtypes = new_dtye
                shapes = new_shape
            if num_jets != 0:
                shapes = {name: (num_jets,) + shape[1:] for name, shape in shapes.items()}
            compression = [ds.compression for ds in f.values()]
            assert len(set(compression)) == 1, "Must have same compression for all groups"
            compression = compression[0]
            if "compression" not in kwargs:
                kwargs["compression"] = compression
        return cls(dtypes=dtypes, shapes=shapes, **kwargs)

    def create_ds(self, name: str, dtype: np.dtype) -> None:
        if name == self.jets_name and self.add_flavour_label and "flavour_label" not in dtype.names:
            dtype = np.dtype([*dtype.descr, ("flavour_label", "i4")])

        fp_vars = self.full_precision_vars or []
        # If no precision is defined, or the field is in full_precision_vars, or its non-float,
        # keep it at the original dtype
        dtype = np.dtype([
            (
                field,
                (
                    self.fp_dtype
                    if (self.fp_dtype and field not in fp_vars and np.issubdtype(dt, np.floating))
                    else dt
                ),
            )
            for field, dt in dtype.descr
        ])

        shape = self.shapes[name]
        chunks = (100,) + shape[1:] if shape[1:] else None

        if self.fixed_mode:
            self.file.create_dataset(
                name, dtype=dtype, shape=shape, compression=self.compression, chunks=chunks
            )
        else:
            maxshape = (None,) + shape[1:]
            self.file.create_dataset(
                name,
                dtype=dtype,
                shape=(0,) + shape[1:],
                maxshape=maxshape,
                compression=self.compression,
                chunks=chunks,
            )

    def close(self) -> None:
        if self.fixed_mode:
            written = len(self.file[self.jets_name])
            if self.num_written != written:
                raise ValueError(
                    f"Attempted to close file {self.dst} when only {self.num_written:,} out of"
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

    def write(self, data: dict[str, np.ndarray]) -> None:
        batch_size = len(data[self.jets_name])
        idx = np.arange(batch_size)
        if self.shuffle:
            self.rng.shuffle(idx)
            data = {name: array[idx] for name, array in data.items()}

        low = self.num_written
        high = low + batch_size

        if self.fixed_mode and high > self.num_jets:
            raise ValueError(
                f"Attempted to write more jets than expected: {high:,} > {self.num_jets:,}"
            )

        for group in self.dtypes:
            ds = self.file[group]
            if not self.fixed_mode:
                ds.resize((high,) + ds.shape[1:])
            ds[low:high] = data[group]

        self.num_written += batch_size
