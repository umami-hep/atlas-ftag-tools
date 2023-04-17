import numpy as np

from ftag.flavour import Flavours


def btag_discriminant(pb, pc, pu, fc=0.1, epsilon=1e-10):
    return np.log((pb + epsilon) / ((1.0 - fc) * pu + fc * pc + epsilon))


def ctag_discriminant(pb, pc, pu, fb=0.2, epsilon=1e-10):
    return np.log((pc + epsilon) / ((1.0 - fb) * pu + fb * pb + epsilon))


def get_discriminant(jets, tagger, signal, fx=0.1, epsilon=1e-10):
    pb, pc, pu = (jets[f"{tagger}_pb"], jets[f"{tagger}_pc"], jets[f"{tagger}_pu"])
    if Flavours[signal] == Flavours.bjets:
        return btag_discriminant(pb, pc, pu, fx, epsilon)
    elif Flavours[signal] == Flavours.cjets:
        return ctag_discriminant(pb, pc, pu, fx, epsilon)
    else:
        raise ValueError(f"Signal flavour must be bjets or cjets, not {signal}")
