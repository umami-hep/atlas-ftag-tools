from __future__ import annotations

import numpy as np

from ftag.flavour import Flavour, Flavours


def btag_discriminant(jets, tagger, fc=0.1, epsilon=1e-10):
    pb, pc, pu = (jets[f"{tagger}_pb"], jets[f"{tagger}_pc"], jets[f"{tagger}_pu"])
    return np.log((pb + epsilon) / ((1.0 - fc) * pu + fc * pc + epsilon))


def ctag_discriminant(jets, tagger, fb=0.2, epsilon=1e-10):
    pb, pc, pu = (jets[f"{tagger}_pb"], jets[f"{tagger}_pc"], jets[f"{tagger}_pu"])
    return np.log((pc + epsilon) / ((1.0 - fb) * pu + fb * pb + epsilon))


def hbb_discriminant(jets, tagger, ftop=0.25, fhcc=0.02, epsilon=1e-10):
    phbb = jets[f"{tagger}_phbb"]
    phcc = jets[f"{tagger}_phcc"]
    ptop = jets[f"{tagger}_ptop"]
    pqcd = jets[f"{tagger}_pqcd"]
    return np.log(phbb / (ftop * ptop + fhcc * phcc + (1 - ftop - fhcc) * pqcd + epsilon))


def hcc_discriminant(jets, tagger, ftop=0.25, fhbb=0.3, epsilon=1e-10):
    phbb = jets[f"{tagger}_phbb"]
    phcc = jets[f"{tagger}_phcc"]
    ptop = jets[f"{tagger}_ptop"]
    pqcd = jets[f"{tagger}_pqcd"]
    return np.log(phcc / (ftop * ptop + fhbb * phbb + (1 - ftop - fhbb) * pqcd + epsilon))


def get_discriminant(
    jets: np.ndarray,
    tagger: str,
    signal: Flavour | str,
    fx: float | tuple[float, ...],
    epsilon: float = 1e-10,
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
    fx : float, optional
        Value fb or fc (fhbb or fhcc and ftop for Xbb)
    epsilon : float, optional
        Small number to avoid division by zero, by default 1e-10

    Returns
    -------
    np.ndarray
        Array of discriminant values.
    """
    if not isinstance(fx, (tuple, list)):
        fx = (fx,)
    tagger_funcs = {
        "bjets": btag_discriminant,
        "cjets": ctag_discriminant,
        "hbb": hbb_discriminant,
        "hcc": hcc_discriminant,
    }

    func = tagger_funcs.get(str(Flavours[signal]), None)
    if func is None:
        raise ValueError(f"Signal flavour must be among {list(tagger_funcs.keys())}, not {signal}")

    return func(jets, tagger, *fx, epsilon)  # type: ignore
