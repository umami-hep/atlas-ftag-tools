from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ftag.labels import Label, LabelContainer


def get_discriminant(
    jets: np.ndarray,
    tagger: str,
    signal: Label,
    flavours: LabelContainer,
    fraction_values: dict[str, float],
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Calculate the tagging discriminant for a given tagger.

    Calculated as the logarithm of the ratio of a specified signal probability
    to a weighted sum ofbackground probabilities.

    Parameters
    ----------
    jets : np.ndarray
        Structured array of jets containing tagger outputs
    tagger : str
        Name of the tagger
    signal : Label
        Signal flavour (bjets/cjets or hbb/hcc)
    fraction_values : dict
        Dict with the fraction values for the background classes for the given tagger
    epsilon : float, optional
        Small number to avoid division by zero, by default 1e-10

    Returns
    -------
    np.ndarray
        Array of discriminant values.

    Raises
    ------
    ValueError
        If the signal flavour is not recognised.
    """
    # Init the denominator
    denominator = 0.0

    # Loop over background flavours
    for flav in flavours:
        # Skip signal flavour for denominator
        if flav == signal:
            continue

        # Get the probability name of the tagger/flavour combo + fraction value
        prob_name = f"{tagger}_{flav.px}"
        fraction_value = fraction_values[flav.frac_str]

        # If fraction_value for the given flavour is zero, skip it
        if fraction_value == 0:
            continue

        # Check that the probability value for the flavour is available
        if fraction_value > 0 and prob_name not in jets.dtype.names:
            raise ValueError(
                f"Nonzero fraction value for {flav.name}, but '{prob_name}' "
                "not found in input array."
            )

        # Update denominator
        denominator += jets[prob_name] * fraction_value if prob_name in jets.dtype.names else 0

    # Calculate numerator
    signal_field = f"{tagger}_{signal.px}"

    # Check that the probability of the signal is available
    if signal_field not in jets.dtype.names:
        raise ValueError(
            f"No signal probability value(s) found for tagger {tagger}. "
            f"Missing variable: {signal_field}"
        )

    return np.log((jets[signal_field] + epsilon) / (denominator + epsilon))
