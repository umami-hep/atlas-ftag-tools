from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ftag import Flavours
from ftag.hdf5 import join_structured_arrays, structured_from_dict
from ftag.labels import Label, LabelContainer


@dataclass
class Labeller:
    """
    Defines a labelling scheme.

    Classes are assigned integer labels in [0, ..., n] based on pre-defined selections.

    Parameters
    ----------
    labels : LabelContainer | list[str | Label]
        The labels to be use.
    require_labels : bool
        Whether to require that all objects are labelled.
    """

    labels: LabelContainer | list[str | Label]
    require_labels: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.labels, LabelContainer):
            self.labels = list(self.labels)
        self.labels = [Flavours[label] for label in self.labels]

    @property
    def variables(self) -> list[str]:
        """
        Returns the variables used for labelling.

        Returns
        -------
        list[str]
            The variables used for labelling.
        """
        return sum((label.cuts.variables for label in self.labels), [])  # type: ignore[union-attr]

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
