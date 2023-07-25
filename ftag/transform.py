from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["Transform"]

Batch = dict[str, np.ndarray]


@dataclass
class Transform:
    variable_name_map: dict[str, dict[str, str]] | None = None
    ints_map: dict[str, dict[str, dict[int, int]]] | None = None
    floats_map: dict[str, dict[str, str | Callable]] | None = None

    def __post_init__(self):
        if self.variable_name_map is None:
            self.variable_name_map = {}
        if self.ints_map is None:
            self.ints_map = {}
        if self.floats_map is None:
            self.floats_map = {}

        # convert string to callable
        for group, map_dict in self.floats_map.items():
            for variable, func in map_dict.items():
                self.floats_map[group][variable] = getattr(np, func)

    def __call__(self, batch: Batch) -> Batch:
        return self.map_floats(self.map_ints(self.map_variables(batch)))

    def map_dtype(self, name: str, dtype: np.dtype, inverse=False) -> np.dtype:
        """
        Rename variables in a dtype.

        Parameters
        ----------
        name : str
            Name of the group.
        dtype : np.dtype
            Structured numpy array dtype.
        inverse : bool, optional
            If True, apply the inverse mapping, by default False.

        Returns
        -------
        np.dtype
            A new dtype with renamed variables.
        """
        assert self.variable_name_map is not None
        names = list(dtype.names)
        remap = self.variable_name_map.get(name)
        if not remap:
            return dtype
        for old, new in remap.items():
            if old in names and new in names:
                raise ValueError(f"Variables {old, new} already exists in {name}.")
        if inverse:
            remap = {v: k for k, v in remap.items()}
        return np.dtype([(remap.get(name, name), dtype[name]) for name in names])

    def map_variables(self, batch: Batch) -> Batch:
        """
        Rename variables in a batch of data.

        Parameters
        ----------
        batch : Batch
            Dict of structured numpy arrays.

        Returns
        -------
        Batch
            Dict of structured numpy arrays with renamed variables.
        """
        assert self.variable_name_map is not None
        for group in self.variable_name_map:
            if group in batch:
                batch[group] = batch[group].astype(self.map_dtype(group, batch[group].dtype))
        return batch

    def map_ints(self, batch: Batch) -> Batch:
        """
        Map integer values to new values.

        Parameters
        ----------
        batch : Batch
            Dict of structured numpy arrays.

        Returns
        -------
        Batch
            Dict of structured numpy arrays with mapped integer values.
        """
        assert self.ints_map is not None
        for group, map_dict in self.ints_map.items():
            if group not in batch:
                continue
            for variable, int_map in map_dict.items():
                if variable not in batch[group].dtype.names:
                    continue
                data = batch[group][variable]
                for old, new in int_map.items():
                    data[data == old] = new
        return batch

    def map_floats(self, batch: Batch) -> Batch:
        """
        Transform float values.

        Parameters
        ----------
        batch : Batch
            Dict of structured numpy arrays.

        Returns
        -------
        Batch
            Dict of structured numpy arrays with transformed float values.
        """
        assert self.floats_map is not None
        for group, map_dict in self.floats_map.items():
            if group not in batch:
                continue
            for variable, func in map_dict.items():
                assert callable(func)
                batch[group][variable] = func(batch[group][variable])
        return batch
