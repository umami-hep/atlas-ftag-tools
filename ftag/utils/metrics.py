"""Tools for metrics module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ftag.utils import logger

if TYPE_CHECKING:  # pragma: no cover
    from ftag.labels import Label, LabelContainer


def save_divide(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
    default: float = 1.0,
):
    """Save divide for denominator equal to 0.

    Division using numpy divide function returning default value in cases where
    denominator is 0.

    Parameters
    ----------
    numerator: np.ndarray | float,
        Numerator in the ratio calculation.
    denominator: np.ndarray | float,
        Denominator in the ratio calculation.
    default: float
        Default value which is returned if denominator is 0.

    Returns
    -------
    ratio: np.ndarray | float
        Result of the division
    """
    logger.debug("Calculating save division.")
    logger.debug("numerator: %s", numerator)
    logger.debug("denominator: %s", denominator)
    logger.debug("default: %s", default)

    if isinstance(numerator, (int, float, np.number)) and isinstance(
        denominator, (int, float, np.number)
    ):
        output_shape = 1
    else:
        try:
            output_shape = denominator.shape
        except AttributeError:
            output_shape = numerator.shape

    ratio = np.divide(
        numerator,
        denominator,
        out=np.ones(
            output_shape,
            dtype=float,
        )
        * default,
        where=(denominator != 0),
    )
    if output_shape == 1:
        return float(ratio)
    return ratio


def weighted_percentile(
    arr: np.ndarray,
    percentile: np.ndarray,
    weights: np.ndarray = None,
):
    """Calculate weighted percentile.

    Implementation according to https://stackoverflow.com/a/29677616/11509698
    (https://en.wikipedia.org/wiki/Percentile#The_weighted_percentile_method)

    Parameters
    ----------
    arr : np.ndarray
        Data array
    percentile : np.ndarray
        Percentile array
    weights : np.ndarray
        Weights array, by default None

    Returns
    -------
    np.ndarray
        Weighted percentile array
    """
    logger.debug("Calculating weighted percentile.")
    logger.debug("arr: %s", arr)
    logger.debug("percentile: %s", percentile)
    logger.debug("weights: %s", weights)

    # Set weights to one if no weights are given
    if weights is None:
        weights = np.ones_like(arr)

    # Set dtype to float64 if the weights are too large
    dtype = np.float64 if np.sum(weights) > 1000000 else np.float32

    # Get an array sorting and sort the array and the weights
    ix = np.argsort(arr)
    arr = arr[ix]
    weights = weights[ix]

    # Return the cumulative sum
    cdf = np.cumsum(weights, dtype=dtype) - 0.5 * weights
    cdf -= cdf[0]
    cdf /= cdf[-1]

    # Return the linear interpolation
    return np.interp(percentile, cdf, arr)


def calculate_efficiency(
    sig_disc: np.ndarray,
    bkg_disc: np.ndarray,
    target_eff: float | list | np.ndarray,
    return_cuts: bool = False,
    sig_weights: np.ndarray = None,
    bkg_weights: np.ndarray = None,
):
    """Calculate efficiency.

    Parameters
    ----------
    sig_disc : np.ndarray
        Signal discriminant
    bkg_disc : np.ndarray
        Background discriminant
    target_eff : float or list or np.ndarray
        Working point which is used for discriminant calculation
    return_cuts : bool
        Specifies if the cut values corresponding to the provided WPs are returned.
        If target_eff is a float, only one cut value will be returned. If target_eff
        is an array, target_eff is an array as well.
    sig_weights : np.ndarray
        Weights for signal events
    bkg_weights : np.ndarray
        Weights for background events

    Returns
    -------
    eff : float or np.ndarray
        Efficiency.
        Return float if target_eff is a float, else np.ndarray
    cutvalue : float or np.ndarray
        Cutvalue if return_cuts is True.
        Return float if target_eff is a float, else np.ndarray
    """
    logger.debug("Calculating efficiency.")
    logger.debug("sig_disc: %s", sig_disc)
    logger.debug("bkg_disc: %s", bkg_disc)
    logger.debug("target_eff: %s", target_eff)
    logger.debug("return_cuts: %s", return_cuts)
    logger.debug("sig_weights: %s", sig_weights)
    logger.debug("bkg_weights: %s", bkg_weights)

    # float | np.ndarray for both target_eff and the returned values
    return_float = False
    if isinstance(target_eff, float):
        return_float = True

    # Flatten the target efficiencies
    target_eff = np.asarray([target_eff]).flatten()

    # Get the cutvalue for the given target efficiency
    cutvalue = weighted_percentile(arr=sig_disc, percentile=1.0 - target_eff, weights=sig_weights)

    # Sort the cutvalues to get the correct order
    sorted_args = np.argsort(1 - target_eff)

    # Get the histogram for the backgrounds
    hist, _ = np.histogram(bkg_disc, (-np.inf, *cutvalue[sorted_args], np.inf), weights=bkg_weights)

    # Calculate the efficiencies for the calculated cut values
    eff = hist[::-1].cumsum()[-2::-1] / hist.sum()
    eff = eff[sorted_args]

    # Ensure that a float is returned if float was given
    if return_float:
        eff = eff[0]
        cutvalue = cutvalue[0]

    # Also return the cuts if wanted
    if return_cuts:
        return eff, cutvalue

    return eff


def calculate_rejection(
    sig_disc: np.ndarray,
    bkg_disc: np.ndarray,
    target_eff,
    return_cuts: bool = False,
    sig_weights: np.ndarray = None,
    bkg_weights: np.ndarray = None,
    smooth: bool = False,
):
    """Calculate rejection.

    Parameters
    ----------
    sig_disc : np.ndarray
        Signal discriminant
    bkg_disc : np.ndarray
        Background discriminant
    target_eff : float or list
        Working point which is used for discriminant calculation
    return_cuts : bool
        Specifies if the cut values corresponding to the provided WPs are returned.
        If target_eff is a float, only one cut value will be returned. If target_eff
        is an array, target_eff is an array as well.
    sig_weights : np.ndarray
        Weights for signal events, by default None
    bkg_weights : np.ndarray
        Weights for background events, by default None

    Returns
    -------
    rej : float or np.ndarray
        Rejection.
        If target_eff is a float, a float is returned if it's a list a np.ndarray
    cut_value : float or np.ndarray
        Cutvalue if return_cuts is True.
        If target_eff is a float, a float is returned if it's a list a np.ndarray
    """
    logger.debug("Calculating rejection.")
    logger.debug("sig_disc: %s", sig_disc)
    logger.debug("bkg_disc: %s", bkg_disc)
    logger.debug("target_eff: %s", target_eff)
    logger.debug("return_cuts: %s", return_cuts)
    logger.debug("sig_weights: %s", sig_weights)
    logger.debug("bkg_weights: %s", bkg_weights)
    logger.debug("smooth: %s", smooth)

    # Calculate the efficiency
    eff = calculate_efficiency(
        sig_disc=sig_disc,
        bkg_disc=bkg_disc,
        target_eff=target_eff,
        return_cuts=return_cuts,
        sig_weights=sig_weights,
        bkg_weights=bkg_weights,
    )

    # Invert the efficiency to get a rejection
    rej = save_divide(1, eff[0] if return_cuts else eff, np.inf)

    # Smooth out the rejection if wanted
    if smooth:
        rej = gaussian_filter1d(rej, sigma=1, radius=2, mode="nearest")

    # Return also the cut values if wanted
    if return_cuts:
        return rej, eff[1]

    return rej


def calculate_efficiency_error(
    arr: np.ndarray,
    n_counts: int,
    suppress_zero_divison_error: bool = False,
    norm: bool = False,
) -> np.ndarray:
    """Calculate statistical efficiency uncertainty.

    Parameters
    ----------
    arr : numpy.array
        Efficiency values
    n_counts : int
        Number of used statistics to calculate efficiency
    suppress_zero_divison_error : bool
        Not raising Error for zero division
    norm : bool, optional
        If True, normed (relative) error is being calculated, by default False

    Returns
    -------
    numpy.array
        Efficiency uncertainties

    Raises
    ------
    ValueError
        If n_counts <=0

    Notes
    -----
    This method uses binomial errors as described in section 2.2 of
    https://inspirehep.net/files/57287ac8e45a976ab423f3dd456af694
    """
    logger.debug("Calculating efficiency error.")
    logger.debug("arr: %s", arr)
    logger.debug("n_counts: %i", n_counts)
    logger.debug("suppress_zero_divison_error: %s", suppress_zero_divison_error)
    logger.debug("norm: %s", norm)
    if np.any(n_counts <= 0) and not suppress_zero_divison_error:
        raise ValueError(f"You passed as argument `N` {n_counts} but it has to be larger 0.")
    if norm:
        return np.sqrt(arr * (1 - arr) / n_counts) / arr
    return np.sqrt(arr * (1 - arr) / n_counts)


def calculate_rejection_error(
    arr: np.ndarray,
    n_counts: int,
    norm: bool = False,
) -> np.ndarray:
    """Calculate the rejection uncertainties.

    Parameters
    ----------
    arr : numpy.array
        Rejection values
    n_counts : int
        Number of used statistics to calculate rejection
    norm : bool, optional
        If True, normed (relative) error is being calculated, by default False

    Returns
    -------
    numpy.array
        Rejection uncertainties

    Raises
    ------
    ValueError
        If n_counts <=0
    ValueError
        If any rejection value is 0

    Notes
    -----
    Special case of `eff_err()`
    """
    logger.debug("Calculating rejection error.")
    logger.debug("arr: %s", arr)
    logger.debug("n_counts: %i", n_counts)
    logger.debug("norm: %s", norm)
    if np.any(n_counts <= 0):
        raise ValueError(f"You passed as argument `n_counts` {n_counts} but it has to be larger 0.")
    if np.any(arr == 0):
        raise ValueError("One rejection value is 0, cannot calculate error.")
    if norm:
        return np.power(arr, 2) * calculate_efficiency_error(1 / arr, n_counts) / arr
    return np.power(arr, 2) * calculate_efficiency_error(1 / arr, n_counts)


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
