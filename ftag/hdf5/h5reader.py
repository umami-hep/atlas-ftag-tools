from __future__ import annotations

import logging as log
import math
from collections.abc import Generator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np

from ftag.cuts import Cuts
from ftag.hdf5.h5utils import get_dtype
from ftag.sample import Sample
from ftag.transform import Transform


@dataclass
class H5SingleReader:
    fname: Path | str
    batch_size: int = 100_000
    jets_name: str = "jets"
    precision: str | None = None
    shuffle: bool = True
    do_remove_inf: bool = False
    transform: Transform | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(42)
        self.sample = Sample(self.fname)
        fname = self.sample.virtual_file()
        if len(fname) != 1:
            raise ValueError("H5SingleReader should only read a single file")
        self.fname = fname[0]

    @cached_property
    def num_jets(self) -> int:
        with h5py.File(self.fname) as f:
            return len(f[self.jets_name])

    def get_attr(self, name, group=None):
        with h5py.File(self.fname) as f:
            obj = f[group] if group else f
            return obj.attrs[name]

    def empty(self, ds: h5py.Dataset, variables: list[str]) -> np.ndarray:
        return np.array(0, dtype=get_dtype(ds, variables, self.precision, transform=self.transform))

    def read_chunk(self, ds: h5py.Dataset, array: np.ndarray, low: int) -> np.ndarray:
        high = min(low + self.batch_size, self.num_jets)
        shape = (high - low,) + ds.shape[1:]
        array.resize(shape, refcheck=False)
        ds.read_direct(array, np.s_[low:high])
        return array

    def remove_inf(self, data: dict) -> dict:
        keep_idx = np.full(len(data[self.jets_name]), True)
        for name, array in data.items():
            for var in array.dtype.names:
                isinf = np.isinf(array[var])
                isinf = isinf if name == self.jets_name else isinf.any(axis=-1)
                keep_idx = keep_idx & ~isinf
                if num_inf := isinf.sum():
                    log.warning(
                        f"{num_inf} inf values detected for variable {var} in"
                        f" {name} array. Removing the affected jets."
                    )
        return {name: array[keep_idx] for name, array in data.items()}

    def stream(
        self,
        variables: dict | None = None,
        num_jets: int | None = None,
        cuts: Cuts | None = None,
        start: int = 0,
    ) -> Generator:
        if num_jets is None:
            num_jets = self.num_jets

        if num_jets > self.num_jets:
            log.warning(
                f"{num_jets:,} jets requested but only {self.num_jets:,} available in {self.fname}."
                " Set to maximum available number!"
            )
            num_jets = self.num_jets

        if variables is None:
            variables = {self.jets_name: None}

        total = 0
        with h5py.File(self.fname) as f:
            arrays = {name: self.empty(f[name], var) for name, var in variables.items()}
            data = {name: self.empty(f[name], var) for name, var in variables.items()}

            # get indices
            indices = list(range(start, self.num_jets + start, self.batch_size))
            if self.shuffle:
                self.rng.shuffle(indices)

            # loop over batches and read file
            for low in indices:
                for name in variables:
                    data[name] = self.read_chunk(f[name], arrays[name], low)

                # apply selections
                if cuts:
                    idx = cuts(data[self.jets_name]).idx
                    data = {name: array[idx] for name, array in data.items()}

                # check for inf and remove
                if self.do_remove_inf:
                    data = self.remove_inf(data)

                # apply transform
                if self.transform:
                    data = self.transform(data)

                # check for completion
                total += len(data[self.jets_name])
                if total >= num_jets:
                    keep = num_jets - (total - len(data[self.jets_name]))
                    data = {name: array[:keep] for name, array in data.items()}
                    yield data
                    break

                yield data


@dataclass
class H5Reader:
    """Reads data from multiple HDF5 files.

    Parameters
    ----------
    fname : Path | str | list[Path | str]
        Path to the HDF5 file or list of paths
    batch_size : int, optional
        Number of jets to read at a time, by default 100_000
    jets_name : str, optional
        Name of the jets dataset, by default "jets"
    precision : str | None, optional
        Cast floats to given precision, by default None
    shuffle : bool, optional
        Read batches in a shuffled order, by default True
    weights : list[float] | None, optional
        Weights for different input datasets, by default None
    do_remove_inf : bool, optional
        Remove jets with inf values, by default False
    transform : Transform | None, optional
        Transform to apply to data, by default None
    equal_jets : bool, optional
        Take the same number of jets (weighted) from each sample, by default True
        If False, use all jets in each sample.
    """

    fname: Path | str | list[Path | str]
    batch_size: int = 100_000
    jets_name: str = "jets"
    precision: str | None = None
    shuffle: bool = True
    weights: list[float] | None = None
    do_remove_inf: bool = False
    transform: Transform | None = None
    equal_jets: bool = True

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(42)
        if not self.equal_jets:
            log.warning(
                "equal_jets is set to False, which will result in different number of jets taken"
                " from each sample. Be aware that this can affect the resampling, so make sure you"
                " know what you are doing."
            )

        if isinstance(self.fname, (str, Path)):
            self.fname = [self.fname]

        # calculate batch sizes
        if self.weights is None:
            self.weights = [1 / len(self.fname)] * len(self.fname)
        self.batch_sizes = [int(w * self.batch_size) for w in self.weights]

        # create readers
        self.readers = [
            H5SingleReader(
                f,
                b,
                self.jets_name,
                self.precision,
                self.shuffle,
                self.do_remove_inf,
                self.transform,
            )
            for f, b in zip(self.fname, self.batch_sizes)
        ]

    @property
    def num_jets(self) -> int:
        return sum(r.num_jets for r in self.readers)

    @property
    def files(self) -> list[Path]:
        return [Path(r.fname) for r in self.readers]

    def dtypes(self, variables: dict[str, list[str]] | None = None) -> dict[str, np.dtype]:
        dtypes = {}
        with h5py.File(self.files[0]) as f:
            if variables is None:
                for key in f:
                    dtype = f[key].dtype
                    if self.transform:
                        dtype = self.transform.map_dtype(key, dtype)
                    dtypes[key] = dtype
            else:
                for name, var in variables.items():
                    ds = f[name]
                    dtype = get_dtype(ds, var, self.precision, transform=self.transform)
                    dtypes[name] = dtype
        return dtypes

    def shapes(self, num_jets: int, groups: list[str] | None = None) -> dict[str, tuple[int, ...]]:
        if groups is None:
            groups = [self.jets_name]
        shapes = {}
        with h5py.File(self.files[0]) as f:
            for group in groups:
                shape = f[group].shape
                shapes[group] = (num_jets,) + shape[1:]
        return shapes

    def stream(
        self,
        variables: dict | None = None,
        num_jets: int | None = None,
        cuts: Cuts | None = None,
        start: int = 0,
    ) -> Generator:
        """Generate batches of selected jets.

        Parameters
        ----------
        variables : dict | None, optional
            Dictionary of variables to for each group, by default use all jet variables.
        num_jets : int | None, optional
            Total number of selected jets to generate, by default all.
        cuts : Cuts | None, optional
            Selection cuts to apply, by default None
        start : int, optional
            Starting index of the first jet to read, by default 0

        Yields
        ------
        Generator
            Generator of batches of selected jets.
        """
        # Check if number of jets is given, if not, set to maximum available
        if num_jets is None:
            num_jets = self.num_jets

        # Check if variables if given, if not, set to all
        if variables is None:
            variables = {self.jets_name: None}

        if self.jets_name not in variables or variables[self.jets_name] is not None:
            jet_vars = variables.get(self.jets_name, [])
            variables[self.jets_name] = list(jet_vars) + (cuts.variables if cuts else [])

        # get streams for selected jets from each reader
        streams = [
            r.stream(variables, int(r.num_jets / self.num_jets * num_jets), cuts, start)
            for r in self.readers
        ]

        while True:
            samples = []
            # Track which streams have been exhausted
            streams_done = [False] * len(streams)

            # for each unexhausted stream, get the next sample
            for i, stream in enumerate(streams):
                if not streams_done[i]:
                    try:
                        samples.append(next(stream))

                    # if equal_jets is True, we can stop when any stream is done
                    # otherwise if sample is exhausted, mark it as done
                    except StopIteration:
                        if self.equal_jets:
                            return
                        streams_done[i] = True

                # if equal_jets is False, we need to keep going until all streams are done
                if all(streams_done):
                    return

            # combine samples and shuffle
            data = {name: np.concatenate([s[name] for s in samples]) for name in variables}
            if self.shuffle:
                idx = np.arange(len(data[self.jets_name]))
                self.rng.shuffle(idx)
                data = {name: array[idx] for name, array in data.items()}

            # yield batch
            yield data

    def load(
        self, variables: dict | None = None, num_jets: int | None = None, cuts: Cuts | None = None
    ) -> dict:
        """Load multiple batches of selected jets into memory.

        Parameters
        ----------
        variables : dict | None, optional
            Dictionary of variables to for each group, by default use all jet variables.
        num_jets : int | None, optional
            Total number of selected jets to load, by default all.
        cuts : Cuts | None, optional
            Selection cuts to apply, by default None

        Returns
        -------
        dict
            Dictionary of arrays for each group.
        """
        # handle default arguments
        if num_jets == -1:
            num_jets = self.num_jets
        if variables is None:
            variables = {self.jets_name: None}

        # get data from each sample
        data: dict[str, list] = {name: [] for name in variables}
        for batch in self.stream(variables, num_jets, cuts):
            for name, array in batch.items():
                if name in data:
                    data[name].append(array)

        # concatenate batches
        return {name: np.concatenate(array) for name, array in data.items()}

    def estimate_available_jets(self, cuts: Cuts, num: int = 1_000_000) -> int:
        """Estimate the number of jets available after selection cuts.

        Parameters
        ----------
        cuts : Cuts
            Selection cuts to apply.
        num : int, optional
            Number of jets to use for the estimation, by default 1_000_000.

        Returns
        -------
        int
            Estimated number of jets available after selection cuts, rounded down.
        """
        # if equal jets is True, available jets is based on the smallest sample
        if self.equal_jets:
            num_jets = []
            for r in self.readers:
                stream = r.stream({self.jets_name: cuts.variables}, num)
                all_jets = np.concatenate([batch[self.jets_name].copy() for batch in stream])
                frac_selected = len(cuts(all_jets).values) / len(all_jets)
                num_jets.append(frac_selected * r.num_jets)
            estimated_num_jets = min(num_jets) * len(self.readers)
        # otherwise, available jets is based on all samples
        else:
            all_jets = self.load({self.jets_name: cuts.variables}, num)[self.jets_name]
            frac_selected = len(cuts(all_jets).values) / len(all_jets)
            estimated_num_jets = frac_selected * self.num_jets
        return math.floor(estimated_num_jets * 0.99)
