from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Transform"]

Batch = dict[str, np.ndarray]


@dataclass
class Transform:
    """Apply variable name remapping, integer remapping, and float transformations.

    The Transform class provides a unified mechanism to perform:
    - variable renaming (variable_map)
    - integer value remapping (ints_map)
    - float transformations (floats_map)

    Each transformation is applied to a *batch* consisting of a dictionary of
    structured numpy arrays.

    Attributes
    ----------
    variable_map : dict[str, dict[str, str]]
        A nested mapping where variable_map[group][old] = new specifies how
        variable names should be renamed inside a given group. If None, no
        variable renaming is applied.

    ints_map : dict[str, dict[str, dict[int, int]]]
        A nested mapping where ints_map[group][variable][old] = new specifies
        how integer values should be remapped. If None, no integer remapping
        is applied.

    floats_map : dict[str, dict[str, str | Callable]]
        A nested mapping where floats_map[group][variable] = func specifies
        a float transformation function. func may either be:
        - a callable
        - a string giving the name of a numpy function (e.g. "log")

        Strings are resolved to numpy.<func> automatically.

    variable_map_inv : dict[str, dict[str, str]]
        Automatically generated inverse of variable_map used for reverse
        variable lookup in :meth:`map_variable_names`.
    """

    variable_map: dict[str, dict[str, str]] = field(default_factory=dict)
    ints_map: dict[str, dict[str, dict[int, int]]] = field(default_factory=dict)
    floats_map: dict[str, dict[str, str | Callable]] = field(default_factory=dict)

    variable_map_inv: dict[str, dict[str, str]] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize internal maps and convert float transformation strings.

        This method ensures that variable_map, ints_map, and
        floats_map are always dictionaries (never None), constructs the
        inverse variable map, and converts any string-based float
        transformations into their numpy equivalents.
        """
        # Normalize maps
        self.variable_map = self.variable_map or {}
        self.ints_map = self.ints_map or {}
        self.floats_map = self.floats_map or {}

        # Inverse variable mapping
        self.variable_map_inv = {
            group: {new: old for old, new in mapping.items()}
            for group, mapping in self.variable_map.items()
        }

        # Convert float-string mappings to numpy functions
        for group, map_dict in self.floats_map.items():
            for variable, func in map_dict.items():
                if isinstance(func, str):
                    self.floats_map[group][variable] = getattr(np, func)

    def __call__(self, batch: Batch) -> Batch:
        """Apply integer remapping, float transformations, and variable renaming.

        Parameters
        ----------
        batch : Batch
            A mapping from group name to structured numpy arrays.

        Returns
        -------
        Batch
            The transformed batch.
        """
        batch = self.map_ints(batch)
        batch = self.map_floats(batch)
        return self.map_variables(batch)

    def map_variables(self, batch: Batch) -> Batch:
        """Rename variables in each group according to variable_map.

        Parameters
        ----------
        batch : Batch
            Dictionary mapping group names to structured numpy arrays.

        Returns
        -------
        Batch
            The batch with variables renamed where applicable.
        """
        assert self.variable_map is not None
        for group in self.variable_map:
            if group in batch:
                batch[group] = batch[group].astype(self.map_dtype(group, batch[group].dtype))
        return batch

    def map_ints(self, batch: Batch) -> Batch:
        """Remap integer values for specified variables inside each group.

        Parameters
        ----------
        batch : Batch
            Dictionary mapping group names to structured numpy arrays.

        Returns
        -------
        Batch
            The batch with integer values remapped.
        """
        assert self.ints_map is not None
        for group, map_dict in self.ints_map.items():
            if group not in batch:
                continue
            for variable, int_map in map_dict.items():
                if variable not in (batch[group].dtype.names or ()):
                    continue
                data = batch[group][variable]
                for old, new in int_map.items():
                    data[data == old] = new
        return batch

    def map_floats(self, batch: Batch) -> Batch:
        """Apply float transformations to selected variables.

        Parameters
        ----------
        batch : Batch
            Dictionary mapping group names to structured numpy arrays.

        Returns
        -------
        Batch
            The batch with float transformations applied.
        """
        assert self.floats_map is not None
        for group, map_dict in self.floats_map.items():
            if group not in batch:
                continue
            for variable, func in map_dict.items():
                assert callable(func)
                batch[group][variable] = func(batch[group][variable])
        return batch

    def map_dtype(self, name: str, dtype: np.dtype) -> np.dtype:
        """Compute a new dtype with renamed fields according to variable_map.

        Parameters
        ----------
        name : str
            Group name associated with the dtype.
        dtype : np.dtype
            Structured dtype whose field names may be modified.

        Returns
        -------
        np.dtype
            A dtype with renamed fields where required.

        Raises
        ------
        ValueError
            When the variables already exist in the dataset.
        """
        assert self.variable_map is not None
        if not (map_dict := self.variable_map.get(name.lstrip("/"))):
            return dtype

        names = list(dtype.names or ())
        for old, new in map_dict.items():
            if old in names and new in names:
                raise ValueError(f"Variables {old, new} already exist in {name}.")
        return np.dtype([(map_dict.get(field, field), dtype[field]) for field in names])

    def map_variable_names(
        self,
        name: str,
        variables: list[str],
        inverse: bool = False,
    ) -> list[str]:
        """Map a list of variable names using variable_map or variable_map_inv.

        Parameters
        ----------
        name : str
            Group name used to select the appropriate name-mapping dictionary.
        variables : list[str]
            List of variable names to be mapped.
        inverse : bool, optional
            If False (default), apply variable_map.
            If True, apply the inverse mapping variable_map_inv.

        Returns
        -------
        list[str]
            A new list of mapped variable names.
        """
        key = name.lstrip("/")

        # Define the map dict
        map_dict = self.variable_map_inv.get(key) if inverse else self.variable_map.get(key)

        # Return variables if map_dict is None
        if not map_dict:
            return variables

        return [map_dict.get(var, var) for var in variables]
