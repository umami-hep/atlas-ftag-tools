from __future__ import annotations

from typing import Callable

import numpy as np

from ftag.flavour import Flavour, Flavours


def discriminant(
    jets: np.ndarray,
    tagger: str,
    signal: Flavour,
    fxs: dict[str, float],
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Get the tagging discriminant.

    Calculated as the logarithm of the ratio of a specified signal probability
    to a weighted sum ofbackground probabilities.

    Parameters
    ----------
    jets : np.ndarray
        Structed jet array containing tagger scores.
    tagger : str
        Name of the tagger, used to construct field names.
    signal : str
        Type of signal.
    fxs : dict[str, float]
        Dict of background probability names and their fractions.
        If a fraction is None, it is calculated as (1 - sum of provided fractions).
    epsilon : float, optional
        A small value added to probabilities to prevent division by zero, by default 1e-10.

    Returns
    -------
    np.ndarray
        The tagger discriminant values for the jets.

    Raises
    ------
    ValueError
        If a fraction is specified for a denominator that is not present in the input array.
    """
    denominator = 0.0
    for d, fx in fxs.items():
        name = f"{tagger}_{d}"
        if fx > 0 and name not in jets.dtype.names:
            raise ValueError(f"Nonzero fx for {d}, but '{name}' not found in input array.")
        denominator += jets[name] * fx if name in jets.dtype.names else 0
    return np.log((jets[f"{tagger}_{signal.px}"] + epsilon) / (denominator + epsilon))


def tautag_dicriminant(jets, tagger, fb, fc, epsilon=1e-10):
    fxs = {"pb": fb, "pc": fc, "pu": 1 - fb - fc}
    return discriminant(jets, tagger, Flavours.taujets, fxs, epsilon=epsilon)


def btag_discriminant(jets, tagger, fc, ftau=0, epsilon=1e-10):
    fxs = {"pc": fc, "ptau": ftau, "pu": 1 - fc - ftau}
    return discriminant(jets, tagger, Flavours.bjets, fxs, epsilon=epsilon)


def ctag_discriminant(jets, tagger, fb, ftau=0, epsilon=1e-10):
    fxs = {"pb": fb, "ptau": ftau, "pu": 1 - fb - ftau}
    return discriminant(jets, tagger, Flavours.cjets, fxs, epsilon=epsilon)


def hbb_discriminant(jets, tagger, ftop=0.25, fhcc=0.02, epsilon=1e-10):
    fxs = {"phcc": fhcc, "ptop": ftop, "pqcd": 1 - ftop - fhcc}
    return discriminant(jets, tagger, Flavours.hbb, fxs, epsilon=epsilon)


def hcc_discriminant(jets, tagger, ftop=0.25, fhbb=0.3, epsilon=1e-10):
    fxs = {"phbb": fhbb, "ptop": ftop, "pqcd": 1 - ftop - fhbb}
    return discriminant(jets, tagger, Flavours.hcc, fxs, epsilon=epsilon)


def get_discriminant(
    jets: np.ndarray, tagger: str, signal: Flavour | str, epsilon: float = 1e-10, **fxs
):
    """Calculate the b-tag or c-tag discriminant for a given tagger.

    Parameters
    ----------
    jets : np.ndarray
        Structured array of jets containing tagger outputs
    tagger : str
        Name of the tagger
    signal : Flavour
        Signal flavour (bjets/cjets or hbb/hcc)
    epsilon : float, optional
        Small number to avoid division by zero, by default 1e-10
    **fxs : dict
        Fractions for the different background flavours.

    Returns
    -------
    np.ndarray
        Array of discriminant values.

    Raises
    ------
    ValueError
        If the signal flavour is not recognised.
    """
    tagger_funcs: dict[str, Callable] = {
        "bjets": btag_discriminant,
        "cjets": ctag_discriminant,
        "taujets": tautag_dicriminant,
        "hbb": hbb_discriminant,
        "hcc": hcc_discriminant,
    }

    if str(signal) not in tagger_funcs:
        raise ValueError(f"Signal flavour must be one of {list(tagger_funcs.keys())}, not {signal}")

    func: Callable = tagger_funcs[str(Flavours[signal])]
    return func(jets, tagger, **fxs, epsilon=epsilon)
