from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np

import ftag
from ftag.hdf5.h5utils import extract_group_full, write_group_full


@dataclass
class H5Writer:
    """Write jet-based data to an HDF5 file.

    This class creates one dataset per entry in ``dtypes``/``shapes`` and
    supports both fixed-size and dynamically growing output files. Floating-point
    fields can optionally be downcast before writing, selected metadata groups
    can be copied from an existing file, and several HDF5 compression backends
    are supported.

    Attributes
    ----------
    dst : Path | str
        Path to the output file.
    dtypes : dict[str, np.dtype]
        Mapping from dataset name to output dtype.
    shapes : dict[str, tuple[int, ...]]
        Mapping from dataset name to output shape. All datasets must agree in
        their first dimension unless ``num_jets`` is explicitly given.
    jets_name : str, optional
        Name of the jet dataset. This dataset is used to determine batch sizes
        during writing. Default is ``"jets"``.
    add_flavour_label : bool, optional
        If ``True``, append a ``"flavour_label"`` field of type ``i4`` to the
        jet dataset if it is not already present. Default is ``False``.
    compression : str | None, optional
        Compression algorithm to use. Supported values are ``None``,
        ``"none"``, ``"gzip"``, ``"lzf"``, ``"lz4"``, and ``"zstd"``.
        Default is ``"lz4"``.
    compression_opts : int | None, optional
        Optional compression level or backend-specific compression setting.
        For ``"gzip"``, this is passed as ``compression_opts`` to HDF5.
        For plugin-based compressors such as ``"lz4"`` and ``"zstd"``, this is
        interpreted as the plugin compression level and folded into the filter
        object. Ignored for compressors that do not support an explicit level.
        Default is ``None``.
    precision : str | None, optional
        Floating-point storage precision for output fields. Supported values are

        - ``"full"``: cast floating-point fields to ``np.float32``
        - ``"half"``: cast floating-point fields to ``np.float16``
        - ``None``: keep original floating-point dtypes

        Default is ``"full"``.
    full_precision_vars : list[str] | None, optional
        Variables that should keep their original dtype even when ``precision``
        requests downcasting. Default is ``None``.
    shuffle : bool, optional
        If ``True``, shuffle each batch before writing. Default is ``True``.
    num_jets : int | None, optional
        Expected total number of jets to write. If given, datasets are created
        in fixed-size mode. If ``None``, datasets are created in dynamic mode
        and resized during writing. Default is ``None``.
    groups : dict[str, dict] | None, optional
        Mapping of metadata group names to extracted group contents to be copied
        into the output file. Default is ``None``.

    Raises
    ------
    ValueError
        If an unsupported precision or compression setting is provided.
    AssertionError
        If dataset shapes disagree in their first dimension when ``num_jets``
        is not explicitly specified.
    """

    dst: Path | str
    dtypes: dict[str, np.dtype]
    shapes: dict[str, tuple[int, ...]]
    jets_name: str = "jets"
    add_flavour_label: bool = False
    compression: str | None = "lz4"
    compression_opts: int | None = None
    precision: str = "full"
    full_precision_vars: list[str] | None = None
    shuffle: bool = True
    num_jets: int | None = None
    groups: dict[str, dict] | None = None

    def __post_init__(self) -> None:
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
                self.shapes[name] = (self.num_jets, *self.shapes[name][1:])

        if self.precision == "full":
            self.fp_dtype = np.float32
        elif self.precision == "half":
            self.fp_dtype = np.float16
        elif self.precision is None:
            self.fp_dtype = None
        else:
            raise ValueError(f"Invalid precision: {self.precision}")

        self.compression, self.compression_opts = self._resolve_compression(self.compression)

        self.dst = Path(self.dst)
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.dst, "w")
        self.add_attr("writer_version", ftag.__version__)

        for name, dtype in self.dtypes.items():
            self.create_ds(name, dtype)
        if self.groups:
            self.save_groups(self.groups)

    def _resolve_compression(
        self, compression: str | None
    ) -> tuple[str | object | None, int | None]:
        """Resolve a user-facing compression setting into HDF5 write arguments.

        This method converts a human-readable compression identifier into the
        values used when calling :meth:`h5py.File.create_dataset`. Built-in HDF5
        filters such as ``"gzip"`` and ``"lzf"`` are returned as strings,
        optionally together with an HDF5 ``compression_opts`` value. Plugin-based
        filters such as ``"lz4"`` and ``"zstd"`` are converted into the
        corresponding filter objects provided by ``hdf5plugin``. In that case,
        any user-provided ``compression_opts`` value is absorbed into the plugin
        object and the returned HDF5 ``compression_opts`` is ``None``.

        Parameters
        ----------
        compression : str | None
            Compression algorithm identifier. Supported values are

            - ``None`` or ``"none"``: disable compression
            - ``"gzip"``: gzip/deflate compression
            - ``"lzf"``: built-in fast compression
            - ``"lz4"``: LZ4 compression via ``hdf5plugin``
            - ``"zstd"``: Zstandard compression via ``hdf5plugin``

        Returns
        -------
        tuple[str | object | None, int | None]
            Two-element tuple ``(compression, compression_opts)`` suitable for
            passing to :meth:`h5py.File.create_dataset`.

            - For no compression, returns ``(None, None)``.
            - For built-in HDF5 filters, returns the filter name and optional
              HDF5 ``compression_opts``.
            - For plugin filters, returns the instantiated plugin object and
              ``None``.

        Raises
        ------
        ValueError
            If the provided compression identifier is not supported.
        """
        if compression is None or compression == "none":
            return None, None

        if compression == "gzip":
            return "gzip", self.compression_opts

        if compression == "lzf":
            return "lzf", None

        if compression == "lz4":
            if self.compression_opts is None:
                return hdf5plugin.LZ4(), None
            return hdf5plugin.LZ4(clevel=self.compression_opts), None

        if compression == "zstd":
            if self.compression_opts is None:
                return hdf5plugin.Zstd(), None
            return hdf5plugin.Zstd(clevel=self.compression_opts), None

        raise ValueError(f"Unsupported compression: {compression}")

    @classmethod
    def from_file(
        cls,
        source: Path | str,
        num_jets: int | None = 0,
        variables: dict[str, list[str] | None] | None = None,
        copy_groups: bool = True,
        **kwargs: Any,
    ) -> H5Writer:
        """Construct a writer from the structure of an existing HDF5 file.

        This class method inspects an input file and derives output dataset
        dtypes, shapes, compression, and optionally metadata groups from it.
        It can be used to create a writer that mirrors the input file layout,
        optionally restricted to a subset of variables and/or a different number
        of output jets.

        Parameters
        ----------
        source : Path | str
            Source HDF5 file from which to infer the output structure.
        num_jets : int | None, optional
            If non-zero, override the first dimension of all dataset shapes with
            this value. If ``0``, keep the original dataset lengths. Default is
            ``0``.
        variables : dict[str, list[str] | None] | None, optional
            Optional mapping from dataset name to a list of variables to keep.
            If provided, output dtypes are reduced accordingly. Default is
            ``None``.
        copy_groups : bool, optional
            If ``True``, copy non-dataset groups from the source file into the
            created writer. Default is ``True``.
        **kwargs : Any
            Additional keyword arguments forwarded to the class constructor.
            This can be used, for example, to override ``compression``,
            ``compression_opts``, ``precision``, or ``full_precision_vars``.

        Returns
        -------
        H5Writer
            Writer initialized from the source file structure.

        Raises
        ------
        TypeError
            If an object in the source file is neither an HDF5 dataset nor group.
        AssertionError
            If the source file datasets do not all use the same compression and
            no explicit ``compression`` override is provided.
        """
        with h5py.File(source, "r") as f:
            dtypes = {}
            shapes = {}
            compression = []
            groups = {}
            for name, ds in f.items():
                if isinstance(ds, h5py.Group):
                    if copy_groups:
                        groups[name] = extract_group_full(ds)
                    continue
                if not isinstance(ds, h5py.Dataset):
                    raise TypeError(
                        f"Unsupported type {type(ds)} for dataset {name} in file {source}"
                    )
                dtypes[name] = ds.dtype
                shapes[name] = ds.shape
                compression.append(ds.compression)

            if variables:
                new_dtype = {}
                new_shape = {}
                for name, ds in f.items():
                    if name not in variables:
                        continue
                    new_dtype[name] = ftag.hdf5.get_dtype(
                        ds,
                        variables=variables[name],
                        precision=kwargs.get("precision"),
                        full_precision_vars=kwargs.get("full_precision_vars"),
                    )
                    new_shape[name] = ds.shape
                dtypes = new_dtype
                shapes = new_shape

            if num_jets != 0:
                shapes = {name: (num_jets, *shape[1:]) for name, shape in shapes.items()}

            assert len(set(compression)) == 1, "Must have same compression for all groups"
            compression = compression[0]
            if "compression" not in kwargs:
                kwargs["compression"] = compression

        return cls(dtypes=dtypes, shapes=shapes, groups=groups, **kwargs)

    def save_groups(self, groups: dict[str, dict]) -> None:
        """Write extracted metadata groups into the output file.

        Parameters
        ----------
        groups : dict[str, dict]
            Mapping from group name to extracted group contents.
        """
        for name, group_data in groups.items():
            if name not in self.file:
                write_group_full(self.file.create_group(name), group_data)

    def create_ds(self, name: str, dtype: np.dtype) -> None:
        """Create one output dataset.

        Parameters
        ----------
        name : str
            Dataset name.
        dtype : np.dtype
            Input dtype definition for the dataset.
        """
        if name == self.jets_name and self.add_flavour_label and "flavour_label" not in dtype.names:
            dtype = np.dtype([*dtype.descr, ("flavour_label", "i4")])

        fp_vars = self.full_precision_vars or []
        dtype = np.dtype(
            [
                (
                    field,
                    (
                        self.fp_dtype
                        if (
                            self.fp_dtype
                            and field not in fp_vars
                            and np.issubdtype(dt, np.floating)
                        )
                        else dt
                    ),
                )
                for field, dt in dtype.descr
            ]
        )

        shape = self.shapes[name]
        chunks = (100, *shape[1:]) if shape[1:] else None
        kwargs = {
            "dtype": dtype,
            "compression": self.compression,
            "chunks": chunks,
        }
        if self.compression_opts is not None:
            kwargs["compression_opts"] = self.compression_opts

        if self.fixed_mode:
            self.file.create_dataset(name, shape=shape, **kwargs)
        else:
            maxshape = (None, *shape[1:])
            self.file.create_dataset(
                name,
                shape=(0, *shape[1:]),
                maxshape=maxshape,
                **kwargs,
            )

    def close(self) -> None:
        """Close the output file.

        Raises
        ------
        ValueError
            If the writer is closed before the expected number of jets has been
            written in fixed-size mode.
        """
        if self.fixed_mode:
            written = len(self.file[self.jets_name])
            if self.num_written != written:
                raise ValueError(
                    f"Attempted to close file {self.dst} when only {self.num_written:,} out of"
                    f" {written:,} jets have been written"
                )
        self.file.close()

    def get_attr(self, name, group=None):
        """Return an attribute from the output file or one of its groups."""
        with h5py.File(self.dst) as f:
            obj = f[group] if group else f
            return obj.attrs[name]

    def add_attr(self, name, data, group=None) -> None:
        """Add an attribute to the output file or one of its groups."""
        obj = self.file[group] if group else self.file
        obj.attrs.create(name, data)

    def copy_attrs(self, fname: Path) -> None:
        """Copy file- and dataset-level attributes from another HDF5 file.

        Parameters
        ----------
        fname : Path
            Path to the source HDF5 file.
        """
        with h5py.File(fname) as f:
            for name, value in f.attrs.items():
                self.add_attr(name, value)
            for name, ds in f.items():
                for attr_name, value in ds.attrs.items():
                    self.add_attr(attr_name, value, group=name)

    def write(self, data: dict[str, np.ndarray]) -> None:
        """Write one batch of data to the output file.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Mapping from dataset name to batch array.

        Raises
        ------
        ValueError
            If writing this batch would exceed ``num_jets`` in fixed-size mode.
        """
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
                ds.resize((high, *ds.shape[1:]))
            ds[low:high] = data[group]

        self.num_written += batch_size
