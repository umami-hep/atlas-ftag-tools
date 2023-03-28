from __future__ import annotations

import logging as log
import math
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ftag.cuts import Cuts
from ftag.hdf5.h5utils import get_dtype
from ftag.vds import create_virtual_file


@dataclass
class H5SingleReader:
    fname: Path | str
    batch_size: int = 100_000
    jets_name: str = "jets"
    precision: str | None = None
    shuffle: bool = True

    def __post_init__(self) -> None:
        self.fname = str(self.fname)
        if "*" in self.fname:
            self.fname = create_virtual_file(self.fname)

    @property
    def num_jets(self) -> int:
        with h5py.File(self.fname) as f:
            return len(f[self.jets_name])

    def get_attr(self, name, group=None):
        with h5py.File(self.fname) as f:
            obj = f[group] if group else f
            return obj.attrs[name]

    def empty(self, ds: h5py.Dataset, variables: list[str]) -> np.ndarray:
        return np.array(0, dtype=get_dtype(ds, variables, self.precision))

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
                keep_idx = keep_idx & ~isinf.any(axis=-1)
                if num_inf := isinf.sum():
                    log.warn(
                        f"{num_inf} inf values detected for variable {var} in"
                        f" {name} array. Removing the affected jets."
                    )
        return {name: array[keep_idx] for name, array in data.items()}

    def stream(
        self, variables: dict | None = None, num_jets: int | None = None, cuts: Cuts | None = None
    ) -> Generator:
        if num_jets is None:
            num_jets = self.num_jets
        if num_jets > self.num_jets:
            raise ValueError(
                f"{num_jets:,} jets requested but only {self.num_jets:,} available in {self.fname}"
            )

        if variables is None:
            variables = {self.jets_name: None}

        total = 0
        rng = np.random.default_rng(42)
        with h5py.File(self.fname) as f:
            data = {name: self.empty(f[name], var) for name, var in variables.items()}

            # get indices
            indices = list(range(0, self.num_jets, self.batch_size))
            if self.shuffle:
                rng.shuffle(indices)

            # loop over batches and read file
            for low in indices:
                for name in variables:
                    data[name] = self.read_chunk(f[name], data[name], low)

                # apply selections
                if cuts:
                    idx = cuts(data[self.jets_name]).idx
                    data = {name: array[idx] for name, array in data.items()}

                # check for inf and remove
                data = self.remove_inf(data)

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
    fname: Path | str | list[Path | str]
    batch_size: int = 100_000
    jets_name: str = "jets"
    precision: str | None = None
    shuffle: bool = True
    weights: list[float] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.fname, (str, Path)):
            self.fname = [self.fname]

        # calculate batch sizes
        if self.weights is None:
            self.weights = [1 / len(self.fname)] * len(self.fname)
        self.batch_sizes = [int(w * self.batch_size) for w in self.weights]

        # create readers
        self.readers = [
            H5SingleReader(fname, batch_size, self.jets_name, self.precision, self.shuffle)
            for fname, batch_size in zip(self.fname, self.batch_sizes)
        ]

    @property
    def vds_path(self) -> Path:
        # TODO: this is mostly needed to get the src dtype,
        # there is probably a better way of exposing this directly
        return Path(self.readers[0].fname)

    @property
    def num_jets(self) -> int:
        return sum(r.num_jets for r in self.readers)

    def get_dtype(self, name: str) -> np.dtype:
        with h5py.File(self.readers[0].fname) as f:
            return f[name].dtype

    def stream(
        self, variables: dict | None = None, num_jets: int | None = None, cuts: Cuts | None = None
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

        Yields
        ------
        Generator
            Generator of batches of selected jets.
        """
        if num_jets is None:
            num_jets = self.num_jets
        if variables is None:
            variables = {self.jets_name: None}
        if self.jets_name not in variables or variables[self.jets_name] is not None:
            jet_vars = variables.get(self.jets_name, [])
            variables[self.jets_name] = list(jet_vars) + (cuts.variables if cuts else [])

        # get streams for selected jets from each reader
        streams = [
            r.stream(variables, int(r.num_jets / self.num_jets * num_jets), cuts)
            for r in self.readers
        ]

        rng = np.random.default_rng(42)
        while True:
            # yeild from each stream
            samples = []
            for stream in streams:
                try:
                    samples.append(next(stream))
                except StopIteration:
                    return

            # combine samples and shuffle
            data = {name: np.concatenate([s[name] for s in samples]) for name in variables}
            if self.shuffle:
                for name in variables:
                    rng.shuffle(data[name])

            # select
            yield data

    def load(
        self, variables: dict | None = None, num_jets: int | None = None, cuts: Cuts | None = None
    ) -> dict:
        if variables is None:
            variables = {self.jets_name: None}
        data: dict[str, list] = {name: [] for name in variables}
        for sample in self.stream(variables, num_jets, cuts):
            for name, array in sample.items():
                if name in data:
                    data[name].append(array)
        return {name: np.concatenate(array) for name, array in data.items()}

    def estimate_available_jets(self, cuts: Cuts, num: int = 1_000_000) -> int:
        """Estimate the number of jets available after selection cuts, rounded down."""
        all_jets = self.load({self.jets_name: cuts.variables}, num)[self.jets_name]
        estimated_num_jets = len(cuts(all_jets).values) / len(all_jets) * self.num_jets
        return math.floor(estimated_num_jets / 1_000) * 1_000
