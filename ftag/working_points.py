"""Calculate tagger working points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from ftag import Flavours
from ftag.cli_utils import HelpFormatter
from ftag.cuts import Cuts
from ftag.hdf5 import H5Reader
from ftag.utils import get_discriminant

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from ftag.labels import Label, LabelContainer


def parse_args(args: Sequence[str]) -> argparse.Namespace:
    """Parse the input arguments into a Namespace.

    Parameters
    ----------
    args : Sequence[str] | None
        Sequence of string inputs to the script

    Returns
    -------
    argparse.Namespace
        Namespace with the parsed arguments

    Raises
    ------
    ValueError
        When both --effs and --disc_cuts are provided
    ValueError
        When neither --effs nor --disc_cuts are provided
    ValueError
        When the number of fraction values is not conistent
    ValueError
        When the sum of fraction values for a tagger is not equal to one
    """
    # Define the pre-parser which checks the --category
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "-c",
        "--category",
        default="single-btag",
        type=str,
        help="Label category to use for the working point calculation",
    )

    pre_parser.add_argument(
        "-s",
        "--signal",
        default="bjets",
        type=str,
        help="Signal flavour which is to be used",
    )

    # Parse only --category/--signal and ignore for now all other args
    pre_args, remaining_argv = pre_parser.parse_known_args(args=args)

    # Create the "real" parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=HelpFormatter,
    )

    # Add --category/--signal so the help is correctly shown
    parser.add_argument(
        "-c",
        "--category",
        default="single-btag",
        type=str,
        help="Label category to use for the working point calculation",
    )
    parser.add_argument(
        "-s",
        "--signal",
        default="bjets",
        type=str,
        help="Signal flavour which is to be used",
    )

    # Check which label category was chosen and load the corresponding flavours
    flavours = Flavours.by_category(pre_args.category)

    # Build the fraction value arguments for all classes (besides signal)
    for flav in flavours:
        # Skip signal
        if flav.name == pre_args.signal:
            continue

        # Built fraction values for all background classes
        parser.add_argument(
            f"--{flav.frac_str}",
            nargs="+",
            required=True,
            type=float,
            help=f"{flav.frac_str} value(s) for each tagger",
        )

    # # Adding the other arguments
    parser.add_argument(
        "--ttbar",
        required=True,
        type=Path,
        help="Path to ttbar sample (supports globbing)",
    )
    parser.add_argument(
        "--zprime",
        required=False,
        type=Path,
        help="Path to zprime (supports globbing). WPs from ttbar will be reused for zprime",
    )
    parser.add_argument(
        "-t",
        "--tagger",
        nargs="+",
        required=True,
        type=str,
        help="tagger name(s)",
    )
    parser.add_argument(
        "-e",
        "--effs",
        nargs="+",
        type=float,
        help="Efficiency working point(s). If -r is specified, values should be 1/efficiency",
    )
    parser.add_argument(
        "-r",
        "--rejection",
        default=None,
        help="Use rejection of specified background class to determine working points",
    )
    parser.add_argument(
        "-d",
        "--disc_cuts",
        nargs="+",
        type=float,
        help="D_x value(s) to calculate efficiency at",
    )
    parser.add_argument(
        "-n",
        "--num_jets",
        default=1_000_000,
        type=int,
        help="Use this many jets (post selection)",
    )
    parser.add_argument(
        "--ttbar_cuts",
        nargs="+",
        default=["pt > 20e3"],
        type=list,
        help="Selection to apply to ttbar (|eta| < 2.5 is always applied)",
    )
    parser.add_argument(
        "--zprime_cuts",
        nargs="+",
        default=["pt > 250e3"],
        type=list,
        help="Selection to apply to zprime (|eta| < 2.5 is always applied)",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        help="Save results to yaml instead of printing",
    )

    # Final parse of all arguments
    parsed_args = parser.parse_args(remaining_argv)

    # Define the signal as an instance of Flavours
    parsed_args.signal = Flavours[parsed_args.signal]

    # Check that only --effs or --disc_cuts is given
    if parsed_args.effs and parsed_args.disc_cuts:
        raise ValueError("Cannot specify both --effs and --disc_cuts")
    if not parsed_args.effs and not parsed_args.disc_cuts:
        raise ValueError("Must specify either --effs or --disc_cuts")

    # Check that all fraction values have the same length
    for flav in flavours:
        if flav.name != parsed_args.signal.name and len(getattr(parsed_args, flav.frac_str)) != len(
            parsed_args.tagger
        ):
            raise ValueError(f"Number of {flav.frac_str} values must match number of taggers")

    # Check that all fraction value combinations add up to one
    for tagger_idx in range(len(parsed_args.tagger)):
        fraction_value_sum = 0
        for flav in flavours:
            if flav.name != parsed_args.signal.name:
                fraction_value_sum += getattr(parsed_args, flav.frac_str)[tagger_idx]

        # Round the value to take machine precision into account
        fraction_value_sum = np.round(fraction_value_sum, 8)

        # Check it's equal to one
        if fraction_value_sum != 1:
            raise ValueError(
                "Sum of the fraction values must be one! You gave "
                f"{fraction_value_sum} for tagger {parsed_args.tagger[tagger_idx]}"
            )
    return parsed_args


def get_fxs_from_args(args: argparse.Namespace, flavours: LabelContainer) -> list:
    """Get the fraction values for each tagger from the argparsed inputs.

    Parameters
    ----------
    args : argparse.Namespace
        Input arguments parsed by the argparser
    flavours : LabelContainer
        LabelContainer instance of the labels that are used

    Returns
    -------
    list
        List of dicts with the fraction values. Each dict is for one tagger.
    """
    # Init the fraction_dict dict
    fraction_dict = {}

    # Add the fraction values to the dict
    for flav in flavours:
        if flav.name != args.signal.name:
            fraction_dict[flav.frac_str] = vars(args)[flav.frac_str]

    return [{k: v[i] for k, v in fraction_dict.items()} for i in range(len(args.tagger))]


def get_eff_rej(
    jets: np.ndarray,
    disc: np.ndarray,
    wp: float,
    flavours: LabelContainer,
) -> dict:
    """Calculate the efficiency/rejection for each flavour.

    Parameters
    ----------
    jets : np.ndarray
        Loaded jets
    disc : np.ndarray
        Discriminant values of the jets
    wp : float
        Working point that is used
    flavours : LabelContainer
        LabelContainer instance of the flavours used

    Returns
    -------
    dict
        Dict with the efficiency/rejection values for each flavour
    """
    # Init an out dict
    out: dict[str, dict] = {"eff": {}, "rej": {}}

    # Loop over the flavours
    for flav in flavours:
        # Calculate discriminant values and efficiencies/rejections
        flav_disc = disc[flav.cuts(jets).idx]
        eff = sum(flav_disc > wp) / len(flav_disc)
        out["eff"][flav.name] = float(f"{eff:.3g}")
        out["rej"][flav.name] = float(f"{1 / eff:.3g}")

    return out


def get_rej_eff_at_disc(
    jets: np.ndarray,
    tagger: str,
    signal: Label,
    disc_cuts: list,
    flavours: LabelContainer,
    fraction_values: dict,
) -> dict:
    """Calculate the efficiency/rejection at a certain discriminant values.

    Parameters
    ----------
    jets : np.ndarray
        Loaded jets used
    tagger : str
        Name of the tagger
    signal : Label
        Label instance of the signal flavour
    disc_cuts : list
        List of discriminant cut values for which the efficiency/rejection is calculated
    flavours : LabelContainer
        LabelContainer instance of the flavours that are used

    Returns
    -------
    dict
        Dict with the discriminant cut values and their respective efficiencies/rejections
    """
    # Calculate discriminants
    disc = get_discriminant(
        jets=jets,
        tagger=tagger,
        signal=signal,
        flavours=flavours,
        fraction_values=fraction_values,
    )

    # Init out dict
    ref_eff_dict: dict[str, dict] = {}

    # Loop over the disc cut values
    for dcut in disc_cuts:
        ref_eff_dict[str(dcut)] = {"eff": {}, "rej": {}}

        # Loop over the flavours
        for flav in flavours:
            e_discs = disc[flav.cuts(jets).idx]
            eff = sum(e_discs > dcut) / len(e_discs)
            ref_eff_dict[str(dcut)]["eff"][str(flav)] = float(f"{eff:.3g}")
            ref_eff_dict[str(dcut)]["rej"][str(flav)] = 1 / float(f"{eff:.3g}")

    return ref_eff_dict


def setup_common_parts(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray | None, LabelContainer]:
    """Load the jets from the files and setup the taggers.

    Parameters
    ----------
    args : argparse.Namespace
        Input arguments from the argparser

    Returns
    -------
    tuple[dict, dict | None, list]
        Outputs the ttbar jets, the zprime jets (if wanted, else None), and the flavours used.
    """
    # Get the used flavours
    flavours = Flavours.by_category(args.category)

    # Get the cuts for the samples
    default_cuts = Cuts.from_list(["eta > -2.5", "eta < 2.5"])
    ttbar_cuts = Cuts.from_list(args.ttbar_cuts) + default_cuts
    zprime_cuts = Cuts.from_list(args.zprime_cuts) + default_cuts

    # Prepare the loading of the jets
    all_vars = list(set(sum((flav.cuts.variables for flav in flavours), [])))
    reader = H5Reader(args.ttbar)
    jet_vars = reader.dtypes()["jets"].names

    # Create for all taggers the fraction values
    for tagger in args.tagger:
        all_vars += [
            f"{tagger}_{flav.px}" for flav in flavours if (f"{tagger}_{flav.px}" in jet_vars)
        ]

    # Load ttbar jets
    ttbar_jets = reader.load({"jets": all_vars}, args.num_jets, cuts=ttbar_cuts)["jets"]
    zprime_jets = None

    # Load zprime jets if needed
    if args.zprime:
        zprime_reader = H5Reader(args.zprime)
        zprime_jets = zprime_reader.load({"jets": all_vars}, args.num_jets, cuts=zprime_cuts)[
            "jets"
        ]

    else:
        zprime_jets = None

    return ttbar_jets, zprime_jets, flavours


def get_working_points(args: argparse.Namespace) -> dict | None:
    """Calculate the working points.

    Parameters
    ----------
    args : argparse.Namespace
        Input arguments from the argparser

    Returns
    -------
    dict | None
        Dict with the working points. If args.outfile is given, the function returns None and
        stored the resulting dict in a yaml file in args.outfile.
    """
    # Load the jets and flavours and get the fraction values
    ttbar_jets, zprime_jets, flavours = setup_common_parts(args=args)
    fraction_values = get_fxs_from_args(args=args, flavours=flavours)

    # Init an out dict
    out = {}

    # Loop over taggers
    for i, tagger in enumerate(args.tagger):
        # Calculate discriminant
        out[tagger] = {"signal": str(args.signal), **fraction_values[i]}
        disc = get_discriminant(
            jets=ttbar_jets,
            tagger=tagger,
            signal=args.signal,
            flavours=flavours,
            fraction_values=fraction_values[i],
        )

        # Loop over efficiency working points
        for eff in args.effs:
            d = out[tagger][f"{eff:.0f}"] = {}

            # Set the working point
            wp_flavour = args.signal
            if args.rejection:
                eff = 100 / eff  # noqa: PLW2901
                wp_flavour = args.rejection

            # Calculate the discriminant value of the working point
            wp_disc = disc[flavours[wp_flavour].cuts(ttbar_jets).idx]
            wp = d["cut_value"] = round(float(np.percentile(wp_disc, 100 - eff)), 3)

            # Calculate efficiency and rejection for each flavour
            d["ttbar"] = get_eff_rej(
                jets=ttbar_jets,
                disc=disc,
                wp=wp,
                flavours=flavours,
            )

            # calculate for zprime
            if args.zprime:
                zprime_disc = get_discriminant(
                    jets=zprime_jets,
                    tagger=tagger,
                    signal=args.signal,
                    flavours=flavours,
                    fraction_values=fraction_values[i],
                )
                d["zprime"] = get_eff_rej(
                    jets=zprime_jets,
                    disc=zprime_disc,
                    wp=wp,
                    flavours=flavours,
                )

    if args.outfile:
        with open(args.outfile, "w") as f:
            yaml.dump(out, f, sort_keys=False)
            return None

    else:
        return out


def get_efficiencies(args: argparse.Namespace) -> dict | None:
    """Calculate the efficiencies for the given jets.

    Parameters
    ----------
    args : argparse.Namespace
        Input arguments from the argparser

    Returns
    -------
    dict | None
        Dict with the efficiencies. If args.outfile is given, the function returns None and
        stored the resulting dict in a yaml file in args.outfile.
    """
    # Load the jets and flavours and get the fraction values
    ttbar_jets, zprime_jets, flavours = setup_common_parts(args=args)
    fraction_values = get_fxs_from_args(args=args, flavours=flavours)

    # Init an out dict
    out = {}

    # Loop over the taggers
    for i, tagger in enumerate(args.tagger):
        out[tagger] = {"signal": str(args.signal), **fraction_values[i]}

        out[tagger]["ttbar"] = get_rej_eff_at_disc(
            jets=ttbar_jets,
            tagger=tagger,
            signal=args.signal,
            disc_cuts=args.disc_cuts,
            flavours=flavours,
            fraction_values=fraction_values[i],
        )
        if args.zprime:
            out[tagger]["zprime"] = get_rej_eff_at_disc(
                jets=zprime_jets,
                tagger=tagger,
                signal=args.signal,
                disc_cuts=args.disc_cuts,
                flavours=flavours,
                fraction_values=fraction_values[i],
            )

    if args.outfile:
        with open(args.outfile, "w") as f:
            yaml.dump(out, f, sort_keys=False)
            return None
    else:
        return out


def main(args: Sequence[str]) -> dict | None:
    """Main function to run working point calculation.

    Parameters
    ----------
    args : Sequence[str] | None, optional
        Input arguments, by default None

    Returns
    -------
    dict | None
        The output dict with the calculated values. When --outfile
        was given, the return value is None
    """
    parsed_args = parse_args(args=args)

    if parsed_args.effs:
        out = get_working_points(args=parsed_args)

    elif parsed_args.disc_cuts:
        out = get_efficiencies(args=parsed_args)

    if out:
        print(yaml.dump(out, sort_keys=False))
        return out

    return None


if __name__ == "__main__":  # pragma: no cover
    main(args=sys.argv[1:])
