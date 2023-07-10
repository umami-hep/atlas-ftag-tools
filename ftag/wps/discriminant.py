from __future__ import annotations

import numpy as np

from ftag.flavour import Flavour, Flavours


def btag_discriminant(pb, pc, pu, fc=0.1, epsilon=1e-10):
    return np.log((pb + epsilon) / ((1.0 - fc) * pu + fc * pc + epsilon))


def ctag_discriminant(pb, pc, pu, fb=0.2, epsilon=1e-10):
    return np.log((pc + epsilon) / ((1.0 - fb) * pu + fb * pb + epsilon))


def get_discriminant(
    jets: np.ndarray, tagger: str, signal: Flavour | str, fx: float, epsilon: float = 1e-10
):
    """Calculate the b-tag or c-tag discriminant for a given tagger.

    Parameters
    ----------
    jets : np.ndarray
        Structured array of jets containing tagger outputs
    tagger : str
        Name of the tagger
    signal : Flavour
        Signal flavour (bjets or cjets)
    fx : float, optional
        Value fb or fc
    epsilon : float, optional
        Small number to avoid division by zero, by default 1e-10

    Returns
    -------
    np.ndarray
        Array of discriminant values.
    """
    pb, pc, pu = (jets[f"{tagger}_pb"], jets[f"{tagger}_pc"], jets[f"{tagger}_pu"])
    if Flavours[signal] == Flavours.bjets:
        return btag_discriminant(pb, pc, pu, fx, epsilon)
    if Flavours[signal] == Flavours.cjets:
        return ctag_discriminant(pb, pc, pu, fx, epsilon)
    raise ValueError(f"Signal flavour must be bjets or cjets, not {signal}")
