from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from types import SimpleNamespace

import pytest

from ftag import Flavours
from ftag.hdf5 import H5Reader
from ftag.mock import get_mock_file
from ftag.wps.working_points import (
    get_discriminant,
    get_eff_rej,
    get_efficiencies,
    get_fxs_from_args,
    get_rej_eff_at_disc,
    get_working_points,
    main,
    parse_args,
    setup_common_parts,
)


@pytest.fixture
def ttbar_file():
    return get_mock_file(100_000)[0]


@pytest.fixture
def zprime_file():
    return get_mock_file(100_000)[0]


def test_get_fxs_from_args():
    """Test minimal scenario for get_fxs_from_args."""
    flavours = Flavours.by_category("single-btag")

    # Prepare an argparse-like object
    # signal is bjets
    args = SimpleNamespace(
        signal=flavours["bjets"],
        fc=[0.2],
        fu=[0.7],
        ftau=[0.1],
        tagger=["mytagger"],
    )

    # Call the function
    result = get_fxs_from_args(args=args, flavours=flavours)

    # We expect a list with length = number of taggers (1).
    # Each item is a dict with key 'fc' => 0.7
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == {"fc": 0.2, "fu": 0.7, "ftau": 0.1}


def test_get_eff_rej(ttbar_file):
    """Test minimal scenario for get_eff_rej."""
    flavours = Flavours.by_category("single-btag")
    fraction_values = {
        "fc": 0.2,
        "fu": 0.7,
        "ftau": 0.1,
    }

    # Setup the reading
    reader = H5Reader(ttbar_file)
    jets = reader.load()["jets"]

    # Get the discriminant values
    disc = get_discriminant(
        jets=jets,
        tagger="MockTagger",
        signal=flavours["bjets"],
        flavours=flavours,
        fraction_values=fraction_values,
    )

    # Calculate efficiency/rejection
    out = get_eff_rej(
        jets=jets,
        disc=disc,
        wp=0,
        flavours=flavours,
    )

    assert "eff" in out
    assert "rej" in out
    for iter_flav in flavours:
        assert iter_flav.name in out["eff"]
        assert iter_flav.name in out["rej"]


def test_get_rej_eff_at_disc(ttbar_file):
    """Test minimal scenario for get_rej_eff_at_disc."""
    flavours = Flavours.by_category("single-btag")
    fraction_values = {
        "fc": 0.2,
        "fu": 0.7,
        "ftau": 0.1,
    }

    # Setup the reading
    reader = H5Reader(ttbar_file)
    jets = reader.load()["jets"]

    out = get_rej_eff_at_disc(
        jets=jets,
        tagger="MockTagger",
        signal=flavours["bjets"],
        disc_cuts=[0, 1, 2],
        flavours=flavours,
        fraction_values=fraction_values,
    )

    assert "0" in out
    assert "1" in out
    assert "2" in out
    for iter_disc in ["0", "1", "2"]:
        assert "eff" in out[iter_disc]
        assert "rej" in out[iter_disc]
        for iter_flav in flavours:
            assert iter_flav.name in out[iter_disc]["eff"]
            assert iter_flav.name in out[iter_disc]["rej"]


def test_setup_common_parts_with_zprime(ttbar_file, zprime_file):
    """Test minimal scenario for setup_common_parts."""
    # Prepare an argparse-like object
    args = SimpleNamespace(
        tagger=["MockTagger"],
        category="single-btag",
        ttbar=ttbar_file,
        zprime=zprime_file,
        num_jets=100_000,
        ttbar_cuts=["pt > 20e3"],
        zprime_cuts=["pt > 250e3"],
    )

    ttbar_jets, zprime_jets, flavours = setup_common_parts(args=args)

    assert flavours == Flavours.by_category("single-btag")
    assert ttbar_jets is not None
    assert zprime_jets is not None


def test_setup_common_parts_no_zprime(ttbar_file):
    """Test minimal scenario for setup_common_parts."""
    # Prepare an argparse-like object
    args = SimpleNamespace(
        tagger=["MockTagger"],
        category="single-btag",
        ttbar=ttbar_file,
        zprime=None,
        num_jets=100_000,
        ttbar_cuts=["pt > 20e3"],
        zprime_cuts=["pt > 250e3"],
    )

    ttbar_jets, zprime_jets, flavours = setup_common_parts(args=args)

    assert flavours == Flavours.by_category("single-btag")
    assert ttbar_jets is not None
    assert zprime_jets is None


def test_get_working_points_no_outfile_no_rejection(ttbar_file, zprime_file):
    """Test minimal scenario for get_working_points."""
    # Get Flavours for the single-btag category
    flavours = Flavours.by_category("single-btag")

    # Prepare an argparse-like object
    args = SimpleNamespace(
        fc=[0.2],
        fu=[0.8],
        ftau=[0.1],
        tagger=["MockTagger"],
        category="single-btag",
        ttbar=ttbar_file,
        zprime=zprime_file,
        num_jets=100_000,
        ttbar_cuts=["pt > 20e3"],
        zprime_cuts=["pt > 250e3"],
        signal=flavours["bjets"],
        effs=[60, 70],
        rejection=None,
        outfile=None,
    )

    out = get_working_points(args=args)

    assert "MockTagger" in out
    for iter_var in ["signal", "fc", "fu", "ftau", "60", "70"]:
        assert iter_var in out["MockTagger"]
    assert out["MockTagger"]["signal"] == "bjets"
    for wp_iter in ["60", "70"]:
        for iter_var in ["cut_value", "ttbar", "zprime"]:
            assert iter_var in out["MockTagger"][wp_iter]

            if iter_var in {"ttbar", "zprime"}:
                for eff_rej_iter in ["eff", "rej"]:
                    assert eff_rej_iter in out["MockTagger"][wp_iter][iter_var]


def test_get_working_points_no_outfile_with_rejection(ttbar_file, zprime_file):
    """Test minimal scenario for get_working_points."""
    # Get Flavours for the single-btag category
    flavours = Flavours.by_category("single-btag")

    # Prepare an argparse-like object
    args = SimpleNamespace(
        fc=[0.2],
        fu=[0.8],
        ftau=[0.1],
        tagger=["MockTagger"],
        category="single-btag",
        ttbar=ttbar_file,
        zprime=zprime_file,
        num_jets=100_000,
        ttbar_cuts=["pt > 20e3"],
        zprime_cuts=["pt > 250e3"],
        signal=flavours["bjets"],
        effs=[60, 70],
        rejection="cjets",
        outfile=None,
    )

    out = get_working_points(args=args)

    assert "MockTagger" in out
    for iter_var in ["signal", "fc", "fu", "ftau", "60", "70"]:
        assert iter_var in out["MockTagger"]
    assert out["MockTagger"]["signal"] == "bjets"
    for wp_iter in ["60", "70"]:
        for iter_var in ["cut_value", "ttbar", "zprime"]:
            assert iter_var in out["MockTagger"][wp_iter]

            if iter_var in {"ttbar", "zprime"}:
                for eff_rej_iter in ["eff", "rej"]:
                    assert eff_rej_iter in out["MockTagger"][wp_iter][iter_var]


def test_get_working_points_with_outfile_no_rejection(ttbar_file, zprime_file):
    """Test minimal scenario for get_working_points."""
    # Get Flavours for the single-btag category
    flavours = Flavours.by_category("single-btag")
    outfile = NamedTemporaryFile(suffix=".yaml", dir=mkdtemp()).name
    outfile_path = Path(outfile)

    # Prepare an argparse-like object
    args = SimpleNamespace(
        fc=[0.2],
        fu=[0.8],
        ftau=[0.1],
        tagger=["MockTagger"],
        category="single-btag",
        ttbar=ttbar_file,
        zprime=zprime_file,
        num_jets=100_000,
        ttbar_cuts=["pt > 20e3"],
        zprime_cuts=["pt > 250e3"],
        signal=flavours["bjets"],
        effs=[60, 70],
        rejection=None,
        outfile=outfile,
    )

    out = get_working_points(args=args)

    assert out is None
    assert outfile_path.exists()


def test_get_efficiencies_no_outfile_no_rejection(ttbar_file, zprime_file):
    """Test minimal scenario for get_efficiencies."""
    # Get Flavours for the single-btag category
    flavours = Flavours.by_category("single-btag")

    # Prepare an argparse-like object
    args = SimpleNamespace(
        fc=[0.2],
        fu=[0.8],
        ftau=[0.1],
        tagger=["MockTagger"],
        category="single-btag",
        ttbar=ttbar_file,
        zprime=zprime_file,
        num_jets=100_000,
        ttbar_cuts=["pt > 20e3"],
        zprime_cuts=["pt > 250e3"],
        signal=flavours["bjets"],
        effs=[60, 70],
        disc_cuts=[0, 1, 2],
        rejection=None,
        outfile=None,
    )

    out = get_efficiencies(args=args)

    assert "MockTagger" in out
    for iter_var in ["signal", "fc", "fu", "ftau", "ttbar", "zprime"]:
        assert iter_var in out["MockTagger"]
    assert out["MockTagger"]["signal"] == "bjets"
    for sample_iter in ["ttbar", "zprime"]:
        for disc_iter in ["0", "1", "2"]:
            for iter_var in ["eff", "rej"]:
                assert iter_var in out["MockTagger"][sample_iter][disc_iter]


def test_get_efficiencies_no_outfile_with_rejection(ttbar_file, zprime_file):
    """Test minimal scenario for get_efficiencies."""
    # Get Flavours for the single-btag category
    flavours = Flavours.by_category("single-btag")

    # Prepare an argparse-like object
    args = SimpleNamespace(
        fc=[0.2],
        fu=[0.8],
        ftau=[0.1],
        tagger=["MockTagger"],
        category="single-btag",
        ttbar=ttbar_file,
        zprime=zprime_file,
        num_jets=100_000,
        ttbar_cuts=["pt > 20e3"],
        zprime_cuts=["pt > 250e3"],
        signal=flavours["bjets"],
        effs=[60, 70],
        disc_cuts=[0, 1, 2],
        rejection="cjets",
        outfile=None,
    )

    out = get_efficiencies(args=args)

    assert "MockTagger" in out
    for iter_var in ["signal", "fc", "fu", "ftau", "ttbar", "zprime"]:
        assert iter_var in out["MockTagger"]
    assert out["MockTagger"]["signal"] == "bjets"
    for sample_iter in ["ttbar", "zprime"]:
        for disc_iter in ["0", "1", "2"]:
            for iter_var in ["eff", "rej"]:
                assert iter_var in out["MockTagger"][sample_iter][disc_iter]


def test_get_efficiencies_with_outfile_no_rejection(ttbar_file, zprime_file):
    """Test minimal scenario for get_efficiencies."""
    # Get Flavours for the single-btag category
    flavours = Flavours.by_category("single-btag")
    outfile = NamedTemporaryFile(suffix=".yaml", dir=mkdtemp()).name
    outfile_path = Path(outfile)

    # Prepare an argparse-like object
    args = SimpleNamespace(
        fc=[0.2],
        fu=[0.8],
        ftau=[0.1],
        tagger=["MockTagger"],
        category="single-btag",
        ttbar=ttbar_file,
        zprime=zprime_file,
        num_jets=100_000,
        ttbar_cuts=["pt > 20e3"],
        zprime_cuts=["pt > 250e3"],
        signal=flavours["bjets"],
        effs=[60, 70],
        disc_cuts=[0, 1, 2],
        rejection=None,
        outfile=outfile,
    )

    out = get_efficiencies(args=args)

    assert out is None
    assert outfile_path.exists()


def test_parse_args_single_btag(ttbar_file):
    """Minimal test for standard behaviour."""
    args = [
        "--category",
        "single-btag",
        "--signal",
        "bjets",
        "--fc",
        "0.2",
        "--fu",
        "0.7",
        "--ftau",
        "0.1",
        "--tagger",
        "MockTagger",
        "--ttbar",
        ttbar_file,
        "--effs",
        "70",
    ]

    parsed_args = parse_args(args=args)

    assert parsed_args is not None
    assert parsed_args.signal == Flavours["bjets"]
    assert parsed_args.category == "single-btag"


def test_parse_args_effs_and_disc_cuts(ttbar_file):
    """Minimal test for ValueError of effs/disc_cuts."""
    args = [
        "--category",
        "single-btag",
        "--signal",
        "bjets",
        "--fc",
        "0.2",
        "--fu",
        "0.7",
        "--ftau",
        "0.1",
        "--tagger",
        "MockTagger",
        "--ttbar",
        ttbar_file,
        "--effs",
        "70",
        "--disc_cuts",
        "50",
    ]

    with pytest.raises(ValueError, match="Cannot specify both --effs and --disc_cuts"):
        parse_args(args=args)


def test_parse_args_neither_effs_nor_disc_cuts(ttbar_file):
    """Minimal test for ValueError of effs/disc_cuts."""
    args = [
        "--category",
        "single-btag",
        "--signal",
        "bjets",
        "--fc",
        "0.2",
        "--fu",
        "0.7",
        "--ftau",
        "0.1",
        "--tagger",
        "MockTagger",
        "--ttbar",
        ttbar_file,
    ]

    with pytest.raises(ValueError, match="Must specify either --effs or --disc_cuts"):
        parse_args(args=args)


def test_parse_args_unequal_number_of_fraction_values(ttbar_file):
    """Minimal test for ValueError for unequal number of fraction values."""
    args = [
        "--category",
        "single-btag",
        "--signal",
        "bjets",
        "--fc",
        "0.2",
        "0.2",
        "--fu",
        "0.7",
        "--ftau",
        "0.1",
        "--tagger",
        "MockTagger",
        "--ttbar",
        ttbar_file,
        "--effs",
        "70",
    ]

    with pytest.raises(ValueError, match="Number of fc values must match number of taggers"):
        parse_args(args=args)


def test_parse_args_fraction_value_sum_unequal_to_one(ttbar_file):
    """Minimal test for ValueError of the fraction value sum."""
    args = [
        "--category",
        "single-btag",
        "--signal",
        "bjets",
        "--fc",
        "0.3",
        "--fu",
        "0.7",
        "--ftau",
        "0.1",
        "--tagger",
        "MockTagger",
        "--ttbar",
        ttbar_file,
        "--effs",
        "70",
    ]

    with pytest.raises(
        ValueError,
        match=("Sum of the fraction values must be one! You gave " "1.1 for tagger MockTagger"),
    ):
        parse_args(args=args)


def test_main_no_output_file(ttbar_file, zprime_file):
    """Minimal test for standard behaviour."""
    args = [
        "--category",
        "single-btag",
        "--signal",
        "bjets",
        "--fc",
        "0.2",
        "--fu",
        "0.7",
        "--ftau",
        "0.1",
        "--tagger",
        "MockTagger",
        "--ttbar",
        ttbar_file,
        "--zprime",
        zprime_file,
        "--effs",
        "70",
    ]

    out = main(args=args)

    assert out is not None


def test_main_with_output_file(ttbar_file, zprime_file):
    """Minimal test for standard behaviour with outfile."""
    outfile = NamedTemporaryFile(suffix=".yaml", dir=mkdtemp()).name
    outfile_path = Path(outfile)

    args = [
        "--category",
        "single-btag",
        "--signal",
        "bjets",
        "--fc",
        "0.2",
        "--fu",
        "0.7",
        "--ftau",
        "0.1",
        "--tagger",
        "MockTagger",
        "--ttbar",
        ttbar_file,
        "--zprime",
        zprime_file,
        "--effs",
        "70",
        "--outfile",
        outfile,
    ]

    out = main(args=args)

    assert out is None
    assert outfile_path.exists()
