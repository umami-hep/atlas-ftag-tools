from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from ftag.cuts import Cuts
from ftag.flavour import Flavours
from ftag.hdf5 import H5Reader
from ftag.wps.discriminant import get_discriminant


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Calculate tagger working points",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--ttbar",
        required=True,
        type=Path,
        help="path to ttbar sample (supports globbing)",
    )
    parser.add_argument(
        "--zprime",
        required=False,
        type=Path,
        help="path to zprime (supports globbing). WPs from ttbar will be reused for zprime",
    )
    parser.add_argument(
        "-e",
        "--effs",
        nargs="+",
        type=float,
        help="efficiency working point(s). If -r is specified, values should be 1/efficiency",
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
        "-f",
        "--fx",
        nargs="+",
        required=True,
        type=float,
        help="fb or fc value(s) for each tagger",
    )
    parser.add_argument(
        "-s",
        "--signal",
        default="bjets",
        choices=["bjets", "cjets", "hbb", "hcc"],
        type=str,
        help='signal flavour ("bjets" or "cjets" for b-tagging, "hbb" or "hcc" for Xbb)',
    )
    parser.add_argument(
        "-r",
        "--rejection",
        default=None,
        choices=["ujets", "cjets", "bjets", "hbb", "hcc", "top", "qcd"],
        help="use rejection of specified background class to determine working points",
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
        help="use this many jets (post selection)",
    )
    parser.add_argument(
        "--ttbar_cuts",
        nargs="+",
        default=["pt > 20e3"],
        type=list,
        help="selection to apply to ttbar (|eta| < 2.5 is always applied)",
    )
    parser.add_argument(
        "--zprime_cuts",
        nargs="+",
        default=["pt > 250e3"],
        type=list,
        help="selection to apply to zprime (|eta| < 2.5 is always applied)",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        help="save results to yaml instead of printing",
    )
    parser.add_argument(
        "--xbb",
        action="store_true",
        help="Enable Xbb tagging which expects two fx values ftop and fhcc/fhbb for each tagger",
    )

    return parser.parse_args(args)


def get_eff_rej(jets, disc, wp, flavs):
    out = {"eff": {}, "rej": {}}
    for bkg in list(flavs):
        bkg_disc = disc[bkg.cuts(jets).idx]
        eff = sum(bkg_disc > wp) / len(bkg_disc)
        out["eff"][str(bkg)] = float(f"{eff:.3g}")
        out["rej"][str(bkg)] = float(f"{1/eff:.3g}")
    return out


def get_working_points(args=None):

    if args.xbb:
        if len(args.fx) != 2 * len(args.tagger):
            raise ValueError("For Xbb tagging, each tagger must have two fx values")
        fx_values = list(zip(args.fx[::2], args.fx[1::2]))
    else:
        if len(args.fx) != len(args.tagger):
            raise ValueError("Number of taggers must match number of fx values")
        fx_values = [(fx,) for fx in args.fx]

    # setup cuts and variables
    flavs = Flavours.by_category("single-btag") if not args.xbb else Flavours.by_category("xbb")

    default_cuts = Cuts.from_list(["eta > -2.5", "eta < 2.5"])
    ttbar_cuts = Cuts.from_list(args.ttbar_cuts) + default_cuts
    zprime_cuts = Cuts.from_list(args.zprime_cuts) + default_cuts
    all_vars = next(iter(flavs)).cuts.variables
    for tagger in args.tagger:
        all_vars += [f"{tagger}_{f.px}" for f in flavs if "tau" not in f.px]

    # load jets
    reader = H5Reader(args.ttbar)
    jets = reader.load({"jets": all_vars}, args.num_jets, cuts=ttbar_cuts)["jets"]
    if args.zprime:
        zp_reader = H5Reader(args.zprime)
        zp_jets = zp_reader.load({"jets": all_vars}, args.num_jets, cuts=zprime_cuts)["jets"]

    # loop over taggers
    out = {}
    for tagger, fx in zip(args.tagger, fx_values):
        out[tagger] = {"signal": args.signal, "fx": fx}

        # calculate discriminant
        disc = get_discriminant(jets, tagger, args.signal, fx)

        # loop over efficiency working points
        for eff in args.effs:
            d = out[tagger][f"{eff:.0f}"] = {}

            wp_flavour = args.signal
            if args.rejection:
                eff = 100 / eff
                wp_flavour = args.rejection

            wp_disc = disc[flavs[wp_flavour].cuts(jets).idx]
            wp = d["cut_value"] = round(float(np.percentile(wp_disc, 100 - eff)), 3)

            # calculate eff and rej for each flavour
            d["ttbar"] = get_eff_rej(jets, disc, wp, flavs)

            # calculate for zprime
            if args.zprime:
                zp_disc = get_discriminant(zp_jets, tagger, Flavours[args.signal], fx)
                d["zprime"] = get_eff_rej(zp_jets, zp_disc, wp, flavs)

    if args.outfile:
        with open(args.outfile, "w") as f:
            yaml.dump(out, f, sort_keys=False)
            return None
    else:
        return out


def get_rej_eff_at_disc(jets, tagger, signal, fx, disc_cuts):
    disc = get_discriminant(jets, tagger, signal, fx)
    d = {}
    flavs = Flavours.by_category("single-btag")
    for dcut in disc_cuts:
        d[str(dcut)] = {"eff": {}, "rej": {}}
        for f in flavs:
            e_discs = disc[f.cuts(jets).idx]
            eff = sum(e_discs > dcut) / len(e_discs)
            d[str(dcut)]["eff"][str(f)] = float(f"{eff:.3g}")
            d[str(dcut)]["rej"][str(f)] = 1 / float(f"{eff:.3g}")
    return d


def get_efficiencies(args=None):
    

    if len(args.tagger) != len(args.fx):
        raise ValueError("Must provide fb/fc for each tagger")

    fx_values = [(fx,) for fx in args.fx]
    # setup cuts and variables
    flavs = Flavours.by_category("single-btag")
    default_cuts = Cuts.from_list(["eta > -2.5", "eta < 2.5"])
    ttbar_cuts = Cuts.from_list(args.ttbar_cuts) + default_cuts
    zprime_cuts = Cuts.from_list(args.zprime_cuts) + default_cuts
    all_vars = next(iter(flavs)).cuts.variables
    for tagger in args.tagger:
        all_vars += [f"{tagger}_{f.px}" for f in flavs if "tau" not in f.px]

    # load jets
    reader = H5Reader(args.ttbar)
    jets = reader.load({"jets": all_vars}, args.num_jets, cuts=ttbar_cuts)["jets"]
    if args.zprime:
        zp_reader = H5Reader(args.zprime)
        zp_jets = zp_reader.load({"jets": all_vars}, args.num_jets, cuts=zprime_cuts)["jets"]

    # loop over taggers
    out = {}
    for tagger, fx in zip(args.tagger, fx_values):
        out[tagger] = {"signal": args.signal, "fx": fx}

        out[tagger]["ttbar"] = get_rej_eff_at_disc(jets, tagger, args.signal, fx, args.disc_cuts)
        if args.zprime:
            out[tagger]["zprime"] = get_rej_eff_at_disc(
                zp_jets, tagger, args.signal, fx, args.disc_cuts
            )

    if args.outfile:
        with open(args.outfile, "w") as f:
            yaml.dump(out, f, sort_keys=False)
            return None
    else:
        return out

def main(args=None):
    args = parse_args(args)

    if args.effs and args.disc_cuts:
        raise ValueError("Cannot specify both --effs and --disc_cuts")

    if args.effs:
        out = get_working_points(args)
    elif args.disc_cuts:
        out = get_efficiencies(args)
    else:
        raise ValueError("Must specify either --effs or --disc_cuts")
    if out:
        print(yaml.dump(out, sort_keys=False))
        return out


if __name__ == "__main__":
    main()
