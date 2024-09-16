from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ftag import Flavours
from ftag.flavour import Flavour, FlavourContainer
from ftag.hdf5 import join_structured_arrays, structured_from_dict


@dataclass
class Labeller:
    """
    Defines a labelling scheme.

    Labels are [0, ..., n] and are assigned using pre-defined selections.

    Parameters
    ----------
    labels : FlavourContainer | list[str | Flavour]
        The labels to be use.
    require_labels : bool
        Whether to require that all objects are labelled.
    """

    labels: FlavourContainer | list[str | Flavour]
    require_labels: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.labels, FlavourContainer):
            self.labels = list(self.labels)
        self.labels = sorted([Flavours[label] for label in self.labels])

    def get_labels(self, array: np.ndarray) -> np.ndarray:
        """
        Returns the labels for the given array.

        Parameters
        ----------
        array : np.ndarray
            The array to label.

        Returns
        -------
        np.ndarray
            The labels for the given array.

        Raises
        ------
        ValueError
            If the `require_labels` attribute is set to `True` and some objects were not labelled.
        """
        labels = -1 * np.ones_like(array, dtype=int)
        for i, label in enumerate(self.labels):
            labels[label.cuts(array).idx] = i

        if self.require_labels and -1 in labels:
            raise ValueError("Some objects were not labelled")

        return labels[labels != -1]

    def add_labels(self, array: np.ndarray, label_name: str = "labels") -> np.ndarray:
        """
        Adds the labels to the given array.

        Parameters
        ----------
        array : np.ndarray
            The array to label.
        label_name : str
            The name of the label column.

        Returns
        -------
        np.ndarray
            The array with the labels added.

        Raises
        ------
        ValueError
            If the `require_labels` attribute is set to `False`.
        """
        if not self.require_labels:
            raise ValueError("Cannot add labels if require_labels is set to False")
        labels = self.get_labels(array)
        labels = structured_from_dict({label_name: labels})
        return join_structured_arrays([array, labels])
