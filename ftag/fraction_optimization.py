from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from ftag import Flavours
from ftag.utils import calculate_rejection, get_discriminant, logger

if TYPE_CHECKING:  # pragma: no cover
    from ftag.labels import Label, LabelContainer


def convert_dict(
    fraction_values: dict | np.ndarray,
    backgrounds: LabelContainer,
) -> np.ndarray | dict:
    if isinstance(fraction_values, dict):
        return np.array([fraction_values[iter_bkg.frac_str] for iter_bkg in backgrounds])

    if isinstance(fraction_values, np.ndarray):
        fraction_values = [
            float(frac_value / np.sum(fraction_values)) for frac_value in fraction_values
        ]

        return dict(zip([iter_bkg.frac_str for iter_bkg in backgrounds], fraction_values))

    raise ValueError(
        f"Only input of type `dict` or `np.ndarray` are accepted! You gave {type(fraction_values)}"
    )


def get_bkg_norm_dict(
    jets: np.ndarray,
    tagger: str,
    signal: Label,
    flavours: LabelContainer,
    working_point: float,
) -> dict:
    # Init a dict for the bkg rejection norm values
    bkg_rej_norm = {}

    # Get the background classes
    backgrounds = flavours.backgrounds(signal)

    # Define a bool array if the jet is signal
    is_signal = signal.cuts(jets).idx

    # Loop over backgrounds
    for bkg in backgrounds:
        # Get the fraction value dict to maximize rejection for given class
        frac_dict_bkg = {
            iter_bkg.frac_str: 1 - (0.01 * len(backgrounds)) if iter_bkg == bkg else 0.01
            for iter_bkg in backgrounds
        }

        # Calculate the disc value using the new fraction dict
        disc = get_discriminant(
            jets=jets,
            tagger=tagger,
            signal=signal,
            flavours=flavours,
            fraction_values=frac_dict_bkg,
        )

        # Calculate the discriminant
        bkg_rej_norm[bkg.name] = calculate_rejection(
            sig_disc=disc[is_signal],
            bkg_disc=disc[bkg.cuts(jets).idx],
            target_eff=working_point,
        )

    return bkg_rej_norm


def calculate_rejection_sum(
    fraction_dict: dict | np.ndarray,
    jets: np.ndarray,
    tagger: str,
    signal: Label,
    flavours: LabelContainer,
    working_point: float,
    bkg_norm_dict: dict,
    rejection_weights: dict,
) -> float:
    # Get the background classes
    backgrounds = flavours.backgrounds(signal)

    # Define a bool array if the jet is signal
    is_signal = signal.cuts(jets).idx

    # Check that the fraction dict is a dict
    if isinstance(fraction_dict, np.ndarray):
        fraction_dict = convert_dict(
            fraction_values=fraction_dict,
            backgrounds=backgrounds,
        )

    # Calculate discriminant
    disc = get_discriminant(
        jets=jets,
        tagger=tagger,
        signal=signal,
        flavours=flavours,
        fraction_values=fraction_dict,
    )

    # Init a dict to which the bkg rejs are added
    sum_bkg_rej = 0

    # Loop over the backgrounds and calculate the rejections
    for iter_bkg in backgrounds:
        sum_bkg_rej += (
            calculate_rejection(
                sig_disc=disc[is_signal],
                bkg_disc=disc[iter_bkg.cuts(jets).idx],
                target_eff=working_point,
            )
            / bkg_norm_dict[iter_bkg.name]
        ) * rejection_weights[iter_bkg.name]

    # Return the negative sum to enable minimizer
    return -1 * sum_bkg_rej


def calculate_best_fraction_values(
    jets: np.ndarray,
    tagger: str,
    signal: Label,
    flavours: LabelContainer,
    working_point: float,
    rejection_weights: dict | None = None,
    optimizer_method: str = "Powell",
) -> dict:
    logger.debug("Calculating best fraction values.")
    logger.debug(f"tagger: {tagger}")
    logger.debug(f"signal: {signal}")
    logger.debug(f"flavours: {flavours}")
    logger.debug(f"working_point: {working_point}")
    logger.debug(f"rejection_weights: {rejection_weights}")
    logger.debug(f"optimizer_method: {optimizer_method}")

    # Ensure Label instance
    if isinstance(signal, str):
        signal = Flavours[signal]

    # Get the background classes
    backgrounds = flavours.backgrounds(signal)

    # Define a default fraction dict
    def_frac_dict = {iter_bkg.frac_str: 1 / len(backgrounds) for iter_bkg in backgrounds}

    # Define rejection weights if not set
    if rejection_weights is None:
        rejection_weights = {iter_bkg.name: 1 for iter_bkg in backgrounds}

    # Get the normalisation for all bkg rejections
    bkg_norm_dict = get_bkg_norm_dict(
        jets=jets,
        tagger=tagger,
        signal=signal,
        flavours=flavours,
        working_point=working_point,
    )

    # Get the best fraction values combination
    result = minimize(
        fun=calculate_rejection_sum,
        x0=convert_dict(fraction_values=def_frac_dict, backgrounds=backgrounds),
        method=optimizer_method,
        bounds=[(0, 1)] * len(backgrounds),
        args=(jets, tagger, signal, flavours, working_point, bkg_norm_dict, rejection_weights),
    )

    # Get the final fraction dict
    final_frac_dict = convert_dict(fraction_values=result.x, backgrounds=backgrounds)

    logger.info(f"Minimization Success: {result.success}")
    logger.info("The following best fraction values were found:")
    for frac_str, frac_value in final_frac_dict.items():
        logger.info(f"{frac_str}: {round(frac_value, ndigits=3)}")

    return final_frac_dict
