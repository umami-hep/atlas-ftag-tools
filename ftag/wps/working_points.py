"""Calculate tagger working points."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from ftag.cli_utils import HelpFormatter
from ftag.cuts import Cuts
from ftag.flavour import Flavours
from ftag.hdf5 import H5Reader
from ftag.wps.discriminant import get_discriminant


def parse_args(args):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=HelpFormatter,
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
    parser.add_argument(
        "--fb",
        nargs="+",
        type=float,
        help="fb value(s) for each tagger",
    )
    parser.add_argument(
        "--fc",
        nargs="+",
        type=float,
        help="fc value(s) for each tagger",
    )
    parser.add_argument(
        "--ftau",
        nargs="+",
        type=float,
        help="ftau value(s) for each tagger",
    )
    parser.add_argument(
        "--ftop",
        nargs="+",
        type=float,
        help="ftop value(s) for each tagger",
    )
    parser.add_argument(
        "--fhbb",
        nargs="+",
        type=float,
        help="fhbb value(s) for each tagger",
    )
    parser.add_argument(
        "--fhcc",
        nargs="+",
        type=float,
        help="fhcc value(s) for each tagger",
    )
    args = parser.parse_args(args)

    args.signal = Flavours[args.signal]

    if args.effs and args.disc_cuts:
        raise ValueError("Cannot specify both --effs and --disc_cuts")
    if not args.effs and not args.disc_cuts:
        raise ValueError("Must specify either --effs or --disc_cuts")

    if args.xbb:
        if args.signal not in {Flavours.hbb, Flavours.hcc}:
            raise ValueError("Xbb tagging only supports hbb or hcc signal flavours")
        if args.fb or args.fc or args.ftau:
            raise ValueError("For Xbb tagging, fb, fc and ftau should not be specified")
        if not args.ftop:
            raise ValueError("For Xbb tagging, ftop should be specified")
        if args.signal == "hbb" and not args.fhcc:
            raise ValueError("For hbb tagging, fhcc should be specified")
        if args.signal == "hcc" and not args.fhbb:
            raise ValueError("For hcc tagging, fhbb should be specified")
    else:
        if args.ftop or args.fhbb or args.fhcc:
            raise ValueError("For single-b tagging, ftop, fhbb and fhcc should not be specified")
        if args.signal == "bjets" and not args.fc:
            raise ValueError("For bjets tagging, fc should be specified")
        if args.signal == "cjets" and not args.fb:
            raise ValueError("For cjets tagging, fb should be specified")
        if args.ftau is None:
            args.ftau = [0.0] * len(args.tagger)

    for fx in ["fb", "fc", "ftau", "ftop", "fhbb", "fhcc"]:
        if getattr(args, fx) and len(getattr(args, fx)) != len(args.tagger):
            raise ValueError(f"Number of {fx} values must match number of taggers")

    return args


def get_fxs_from_args(args):
    if args.signal == Flavours.bjets:
        fxs = {"fc": args.fc, "ftau": args.ftau}
    elif args.signal == Flavours.cjets:
        fxs = {"fb": args.fb, "ftau": args.ftau}
    elif args.signal == Flavours.hbb:
        fxs = {"ftop": args.ftop, "fhcc": args.fhcc}
    elif args.signal == Flavours.hcc:
        fxs = {"ftop": args.ftop, "fhbb": args.fhbb}
    assert fxs is not None
    return [{k: v[i] for k, v in fxs.items()} for i in range(len(args.tagger))]


def get_eff_rej(jets, disc, wp, flavs):
    out = {"eff": {}, "rej": {}}
    for bkg in list(flavs):
        bkg_disc = disc[bkg.cuts(jets).idx]
        eff = sum(bkg_disc > wp) / len(bkg_disc)
        out["eff"][str(bkg)] = float(f"{eff:.3g}")
        out["rej"][str(bkg)] = float(f"{1 / eff:.3g}")
    return out


def get_rej_eff_at_disc(jets, tagger, signal, disc_cuts, **fxs):
    disc = get_discriminant(jets, tagger, signal, **fxs)
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


def setup_common_parts(args):
    flavs = Flavours.by_category("single-btag") if not args.xbb else Flavours.by_category("xbb")
    default_cuts = Cuts.from_list(["eta > -2.5", "eta < 2.5"])
    ttbar_cuts = Cuts.from_list(args.ttbar_cuts) + default_cuts
    zprime_cuts = Cuts.from_list(args.zprime_cuts) + default_cuts

    # prepare to load jets
    all_vars = next(iter(flavs)).cuts.variables
    reader = H5Reader(args.ttbar)
    jet_vars = reader.dtypes()["jets"].names
    for tagger in args.tagger:
        all_vars += [f"{tagger}_{f.px}" for f in flavs if (f"{tagger}_{f.px}" in jet_vars)]

    # load jets
    jets = reader.load({"jets": all_vars}, args.num_jets, cuts=ttbar_cuts)["jets"]
    zp_jets = None
    if args.zprime:
        zp_reader = H5Reader(args.zprime)
        zp_jets = zp_reader.load({"jets": all_vars}, args.num_jets, cuts=zprime_cuts)["jets"]

    return jets, zp_jets, flavs


def get_working_points(args=None):
    jets, zp_jets, flavs = setup_common_parts(args)
    fxs = get_fxs_from_args(args)

    # loop over taggers
    out = {}
    for i, tagger in enumerate(args.tagger):
        # calculate discriminant
        out[tagger] = {"signal": str(args.signal), **fxs[i]}
        disc = get_discriminant(jets, tagger, args.signal, **fxs[i])

        # loop over efficiency working points
        for eff in args.effs:
            d = out[tagger][f"{eff:.0f}"] = {}

            wp_flavour = args.signal
            if args.rejection:
                eff = 100 / eff  # noqa: PLW2901
                wp_flavour = args.rejection

            wp_disc = disc[flavs[wp_flavour].cuts(jets).idx]
            wp = d["cut_value"] = round(float(np.percentile(wp_disc, 100 - eff)), 3)

            # calculate eff and rej for each flavour
            d["ttbar"] = get_eff_rej(jets, disc, wp, flavs)

            # calculate for zprime
            if args.zprime:
                zp_disc = get_discriminant(zp_jets, tagger, Flavours[args.signal], **fxs[i])
                d["zprime"] = get_eff_rej(zp_jets, zp_disc, wp, flavs)

    if args.outfile:
        with open(args.outfile, "w") as f:
            yaml.dump(out, f, sort_keys=False)
            return None
    else:
        return out


def get_efficiencies(args=None):
    jets, zp_jets, _ = setup_common_parts(args)
    fxs = get_fxs_from_args(args)

    out = {}
    for i, tagger in enumerate(args.tagger):
        out[tagger] = {"signal": str(args.signal), **fxs[i]}

        out[tagger]["ttbar"] = get_rej_eff_at_disc(
            jets, tagger, args.signal, args.disc_cuts, **fxs[i]
        )
        if args.zprime:
            out[tagger]["zprime"] = get_rej_eff_at_disc(
                zp_jets, tagger, args.signal, args.disc_cuts, **fxs[i]
            )

    if args.outfile:
        with open(args.outfile, "w") as f:
            yaml.dump(out, f, sort_keys=False)
            return None
    else:
        return out


def main(args=None):
    args = parse_args(args)

    if args.effs:
        out = get_working_points(args)
    elif args.disc_cuts:
        out = get_efficiencies(args)

    if out:
        print(yaml.dump(out, sort_keys=False))
        return out

    return None


if __name__ == "__main__":
    main()
