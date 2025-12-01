from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from ftag.cli_utils import HelpFormatter
from ftag.cuts import Cuts
from ftag.hdf5 import H5Reader
from ftag.labels import LabelContainer
from ftag.utils import calculate_rejection, get_discriminant, logger, set_log_level

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from ftag.labels import Label


def convert_dict(
    fraction_values: dict | np.ndarray,
    backgrounds: LabelContainer,
) -> np.ndarray | dict:
    """Convert the fraction values from dict to array or vice versa.

    Parameters
    ----------
    fraction_values : dict | np.ndarray
        Dict of array with the fraction values
    backgrounds : LabelContainer
        LabelContainer with the background flavours

    Returns
    -------
    np.ndarray | dict
        Array or dict with the fraction values

    Raises
    ------
    TypeError
        If the type of the input was wrong
    """
    if isinstance(fraction_values, dict):
        return np.array([fraction_values[iter_bkg.frac_str] for iter_bkg in backgrounds])

    if isinstance(fraction_values, np.ndarray):
        fraction_values = [
            float(frac_value / np.sum(fraction_values)) for frac_value in fraction_values
        ]

        return dict(
            zip([iter_bkg.frac_str for iter_bkg in backgrounds], fraction_values, strict=False)
        )

    raise TypeError(
        f"Only input of type `dict` or `np.ndarray` are accepted! You gave {type(fraction_values)}"
    )


def get_bkg_norm_dict(
    jets: np.ndarray,
    tagger: str,
    signal: Label,
    flavours: LabelContainer,
    working_point: float,
) -> dict:
    """Get the normalisation dict for the background flavours.

    Parameters
    ----------
    jets : np.ndarray
        Loaded jets
    tagger : str
        Name of the tagger
    signal : Label
        Label instance of the signal
    flavours : LabelContainer
        LabelContainer instance with all flavours used
    working_point : float
        Working point that is to be used

    Returns
    -------
    dict
        Background normalisation dict
    """
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
    """Calculate the sum of the normalised rejections.

    Parameters
    ----------
    fraction_dict : dict | np.ndarray
        Dict/Array with the fraction values
    jets : np.ndarray
        Loaded jets
    tagger : str
        Name of the tagger
    signal : Label
        Label instance of the signal
    flavours : LabelContainer
        LabelContainer with all flavours
    working_point : float
        Working point that is used
    bkg_norm_dict : dict
        Backgroud normalisation dict
    rejection_weights : dict
        Weights for the rejections

    Returns
    -------
    float
        Sum of the normalised rejections times -1 (for minimize)
    """
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
    """Calculate the best fraction values for a given tagger and working point.

    Parameters
    ----------
    jets : np.ndarray
        Loaded jets
    tagger : str
        Name of the tagger
    signal : Label
        Label instance of the signal
    flavours : LabelContainer
        LabelContainer with all flavours
    working_point : float
        Working point that is used
    rejection_weights : dict | None, optional
        Rejection weights for the background classes, by default None
    optimizer_method : str, optional
        Optimizer method for the minimization, by default "Powell"

    Returns
    -------
    dict
        Dict with the best fraction values
    """
    logger.debug("Calculating best fraction values.")
    logger.debug(f"tagger: {tagger}")
    logger.debug(f"signal: {signal}")
    logger.debug(f"flavours: {flavours}")
    logger.debug(f"working_point: {working_point}")
    logger.debug(f"rejection_weights: {rejection_weights}")
    logger.debug(f"optimizer_method: {optimizer_method}")

    # Ensure Label instance
    if isinstance(signal, str):
        signal = flavours[signal]

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


def parse_args(args: Sequence[str] | None) -> argparse.Namespace:
    """Parse the input arguments into a Namespace.

    Parameters
    ----------
    args : Sequence[str] | None
        Sequence of string inputs to the script

    Returns
    -------
    argparse.Namespace
        Namespace with the parsed arguments
    """
    # Create the parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        type=Path,
        help="Path to the H5 file that will be used. Wildcard for multiple H5 files is supported",
    )
    parser.add_argument(
        "-t",
        "--tagger",
        required=True,
        type=str,
        help="Name of the tagger in the files.",
    )
    parser.add_argument(
        "-s",
        "--signal",
        default="bjets",
        type=str,
        help="Name of the signal class.",
    )
    parser.add_argument(
        "-w",
        "--working_point",
        required=True,
        type=float,
        help="Working point for which to optimize the fraction values. "
        "Values are given in sub-one values (e.g. 0.70 for 70%% WP).",
    )
    parser.add_argument(
        "-o",
        "--optimizer_method",
        default="Powell",
        type=str,
        help="Optimizer method for the minimization.",
    )
    parser.add_argument(
        "-n",
        "--num_jets",
        default=None,
        type=int,
        help="Number of jets to load from H5. By default None (all jets).",
    )
    parser.add_argument(
        "-c",
        "--cuts",
        default=[
            "pt_btagJes > 20e3",
            "pt_btagJes < 250e3",
            "absEta_btagJes < 2.5",
        ],
        type=list,
        help="Cuts that are to be applied as list. Default is the ttbar selection we use. ",
    )
    parser.add_argument(
        "--batch_size",
        default=100_000,
        type=int,
        help="Batch size used when loading the jets from H5.",
    )
    parser.add_argument(
        "--jets_name",
        default="jets",
        type=str,
        help="Name of the jet collection in the H5 file.",
    )
    parser.add_argument(
        "--flavour_file",
        default=None,
        type=str,
        help="File with custom flavour definition that differs from the flavours.yaml",
    )
    parser.add_argument(
        "--flavour_class",
        default="single-btag",
        type=str,
        help="Name of the category of the flavours that will be used.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity level",
    )

    # Final parse of all arguments
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    """Main function to run the fraction value optimization.

    Parameters
    ----------
    args : Sequence[str] | None
        Input arguments, by default None
    """
    # Parse the command line arguments
    parsed_args = parse_args(args=args)

    # Set the log level
    set_log_level(tools_logger=logger, log_level=parsed_args.verbose)

    # Get the flavour class we will use
    flavours = LabelContainer.from_yaml(
        yaml_path=parsed_args.flavour_file,
        include_categories=[parsed_args.flavour_class],
    )

    # Get the cuts that are to be applied
    cuts = Cuts.from_list(parsed_args.cuts)

    # Get the probability names which need to be loaded from H5
    vars_to_load = [f"{parsed_args.tagger}_{flav.px}" for flav in flavours]

    # Add the variables needed for the flavours to the list
    vars_to_load += flavours.cut_variables()

    # Ensure that each variable only is once in the list
    vars_to_load = list(set(vars_to_load))

    # Debug statement before loading the jets
    logger.debug(f"Flavours: {flavours}")
    logger.debug(f"Flavour File: {parsed_args.flavour_file}")
    logger.debug(f"Variables: {vars_to_load}")
    logger.debug(f"Cuts: {vars_to_load}")

    # Load the actual jets from the file
    jets = H5Reader(
        fname=parsed_args.input_file,
        batch_size=parsed_args.batch_size,
        jets_name=parsed_args.jets_name,
        shuffle=False,
    ).load(
        variables={parsed_args.jets_name: vars_to_load},
        num_jets=parsed_args.num_jets,
        cuts=cuts,
    )[parsed_args.jets_name]

    # Call the actual fraction value optimization function
    calculate_best_fraction_values(
        jets=jets,
        tagger=parsed_args.tagger,
        signal=parsed_args.signal,
        flavours=flavours,
        working_point=parsed_args.working_point,
        optimizer_method=parsed_args.optimizer_method,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
