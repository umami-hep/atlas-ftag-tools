from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["Transform"]


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

    def __call__(self, batch: dict) -> dict:
        return self.map_ints(self.rename_variables(batch))

    def rename_variables(self, batch: dict) -> dict:
        """
        Rename variables in a batch of data.

        Parameters
        ----------
        batch : dict
            Dict of structured numpy arrays.

        Returns
        -------
        dict
            Dict of structured numpy arrays with renamed variables.
        """
        assert self.variable_name_map is not None
        for group, remap in self.variable_name_map.items():
            dtype = batch[group].dtype
            names = list(dtype.names)
            for old, new in remap.items():
                if new in names:
                    raise ValueError(f"Variable {new} already exists in {group}.")
                if old in names:
                    names[names.index(old)] = new
            dtype.names = names

        return batch

    def map_ints(self, batch: dict) -> dict:
        """
        Map integer values to new values.

        Parameters
        ----------
        batch : dict
            Dict of structured numpy arrays.

        Returns
        -------
        dict
            Dict of structured numpy arrays with mapped integer values.
        """
        assert self.ints_map is not None
        for group, map_dict in self.ints_map.items():
            for variable, int_map in map_dict.items():
                if variable not in batch[group].dtype.names:
                    continue
                data = batch[group][variable]
                for old, new in int_map.items():
                    data[data == old] = new
        return batch

    def transform_floats(self, batch: dict) -> dict:
        """
        Transform float values.

        Parameters
        ----------
        batch : dict
            Dict of structured numpy arrays.

        Returns
        -------
        dict
            Dict of structured numpy arrays with transformed float values.
        """
        assert self.floats_map is not None
        for group, map_dict in self.floats_map.items():
            for variable, func in map_dict.items():
                assert callable(func)
                batch[group][variable] = func(batch[group][variable])
        return batch
