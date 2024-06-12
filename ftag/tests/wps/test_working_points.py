from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ftag.mock import get_mock_file
from ftag.wps.working_points import main


@pytest.fixture
def test_file():
    return get_mock_file(10_000)[0]


@pytest.fixture
def zprime_file():
    return get_mock_file(10_000)[0]


def test_get_working_points(test_file, eff_val="60"):
    args = [
        "--ttbar",
        str(test_file),
        "-t",
        "MockTagger",
        "--fc",
        "0.01",
        "-e",
        eff_val,
        "-n",
        "10_000",
    ]
    output = main(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fc"] == (0.01)
    assert output["MockTagger"]["ftau"] == (0.00)
    assert eff_val in output["MockTagger"]
    assert "cut_value" in output["MockTagger"][eff_val]
    assert "ttbar" in output["MockTagger"][eff_val]
    assert "eff" in output["MockTagger"][eff_val]["ttbar"]
    assert "rej" in output["MockTagger"][eff_val]["ttbar"]
    assert output["MockTagger"][eff_val]["ttbar"]["eff"]["bjets"] == pytest.approx(
        float(eff_val) / 100, rel=1e-2
    )


def test_get_working_points_rejection(test_file, rej_val="100"):
    args = [
        "--ttbar",
        str(test_file),
        "-t",
        "MockTagger",
        "--fc",
        "0.01",
        "-e",
        rej_val,
        "-n",
        "10_000",
        "-r",
        "ujets",
    ]
    output = main(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fc"] == (0.01)
    assert output["MockTagger"]["ftau"] == (0.00)
    assert rej_val in output["MockTagger"]
    assert "cut_value" in output["MockTagger"][rej_val]
    assert "ttbar" in output["MockTagger"][rej_val]
    assert "eff" in output["MockTagger"][rej_val]["ttbar"]
    assert "rej" in output["MockTagger"][rej_val]["ttbar"]
    assert output["MockTagger"][rej_val]["ttbar"]["eff"]["ujets"] == pytest.approx(
        1 / float(rej_val), rel=1e-1
    )


def test_get_working_points_cjets(test_file, eff_val="60"):
    args = [
        "--ttbar",
        str(test_file),
        "-t",
        "MockTagger",
        "-s",
        "cjets",
        "--fb",
        "0.01",
        "-e",
        eff_val,
        "-n",
        "10_000",
    ]
    output = main(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "cjets"
    assert output["MockTagger"]["fb"] == 0.01
    assert output["MockTagger"]["ftau"] == 0.00
    assert eff_val in output["MockTagger"]
    assert "cut_value" in output["MockTagger"][eff_val]
    assert "ttbar" in output["MockTagger"][eff_val]
    assert "eff" in output["MockTagger"][eff_val]["ttbar"]
    assert "rej" in output["MockTagger"][eff_val]["ttbar"]
    assert output["MockTagger"][eff_val]["ttbar"]["eff"]["cjets"] == pytest.approx(
        float(eff_val) / 100, rel=1e-2
    )


def test_get_working_points_zprime(test_file, zprime_file, eff_val="60"):
    args = [
        "--ttbar",
        str(test_file),
        "--zprime",
        str(zprime_file),
        "-t",
        "MockTagger",
        "--fc",
        "0.15",
        "-e",
        eff_val,
        "-n",
        "10_000",
    ]
    output = main(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fc"] == (0.15)
    assert output["MockTagger"]["ftau"] == (0.00)
    assert eff_val in output["MockTagger"]
    assert "cut_value" in output["MockTagger"][eff_val]
    assert "ttbar" in output["MockTagger"][eff_val]
    assert "eff" in output["MockTagger"][eff_val]["ttbar"]
    assert "rej" in output["MockTagger"][eff_val]["ttbar"]
    assert "zprime" in output["MockTagger"][eff_val]
    assert "eff" in output["MockTagger"][eff_val]["zprime"]
    assert "rej" in output["MockTagger"][eff_val]["zprime"]
    assert output["MockTagger"][eff_val]["ttbar"]["eff"]["bjets"] == pytest.approx(
        float(eff_val) / 100, rel=1e-2
    )


def test_get_working_points_inc_tau(test_file, eff_val="60"):
    args = [
        "--ttbar",
        str(test_file),
        "-t",
        "MockTagger",
        "--fc",
        "0.01",
        "--ftau",
        "0.02",
        "-e",
        eff_val,
        "-n",
        "10_000",
    ]
    output = main(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fc"] == (0.01)
    assert output["MockTagger"]["ftau"] == (0.02)
    assert eff_val in output["MockTagger"]
    assert "cut_value" in output["MockTagger"][eff_val]
    assert "ttbar" in output["MockTagger"][eff_val]
    assert "eff" in output["MockTagger"][eff_val]["ttbar"]
    assert "rej" in output["MockTagger"][eff_val]["ttbar"]
    assert output["MockTagger"][eff_val]["ttbar"]["eff"]["bjets"] == pytest.approx(
        float(eff_val) / 100, rel=1e-2
    )


def test_get_working_points_xbb(test_file, eff_val="60"):
    # Assuming you're testing with two fx values for each tagger as required for Xbb
    ftop_value = "0.25"
    fhcc_value = "0.02"

    args = [
        "--ttbar",
        str(test_file),
        "-t",
        "MockXbbTagger",
        "--ftop",
        ftop_value,
        "--fhcc",
        fhcc_value,
        "-e",
        eff_val,
        "-n",
        "10_000",
        "--xbb",  # Enable Xbb tagging
        "-s",
        "hbb",  # Test for hbb signal
    ]

    output = main(args)

    assert "MockXbbTagger" in output
    assert output["MockXbbTagger"]["signal"] == "hbb"
    assert output["MockXbbTagger"]["ftop"] == float(ftop_value)
    assert output["MockXbbTagger"]["fhcc"] == float(fhcc_value)
    assert eff_val in output["MockXbbTagger"]
    assert "cut_value" in output["MockXbbTagger"][eff_val]
    assert "ttbar" in output["MockXbbTagger"][eff_val]
    assert "eff" in output["MockXbbTagger"][eff_val]["ttbar"]
    assert "rej" in output["MockXbbTagger"][eff_val]["ttbar"]
    assert output["MockXbbTagger"][eff_val]["ttbar"]["eff"]["hbb"] == pytest.approx(
        float(eff_val) / 100, rel=1e-2
    )


def test_get_working_points_fx_length_check():
    # test with incorrect length of fx values for regular b-tagging
    with pytest.raises(ValueError):
        main(["--ttbar", "path", "-t", "MockTagger", "--fc", "0.1", "0.2", "0.3"])

    # test with incorrect length of fx values for Xbb tagging
    with pytest.raises(ValueError):
        main(["--ttbar", "path", "--xbb", "-t", "MockXbbTagger", "--fc", "0.25"])

    with pytest.raises(ValueError):
        main(["--ttbar", "path", "-t", "MockTagger", "--fc", "0.1", "0.2", "0.3", "-d", "1.0"])


def test_get_rej_eff_at_disc_ttbar(test_file, disc_vals=None):
    if disc_vals is None:
        disc_vals = [1.0, 1.5]
    args = [
        "--ttbar",
        str(test_file),
        "-t",
        "MockTagger",
        "--fc",
        "0.01",
        "-n",
        "10_000",
    ]
    args.extend(["-d"] + [str(x) for x in disc_vals])

    output = main(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fc"] == 0.01
    assert output["MockTagger"]["ftau"] == 0.00
    assert "ttbar" in output["MockTagger"]
    ttbar = output["MockTagger"]["ttbar"]
    for dval in disc_vals:
        assert str(dval) in ttbar
        assert "eff" in ttbar[str(dval)]
        assert "rej" in ttbar[str(dval)]

    assert "zprime" not in output["MockTagger"]


def test_get_rej_eff_at_disc_zprime(test_file, zprime_file, disc_vals=None):
    if disc_vals is None:
        disc_vals = [1.0, 1.5]
    args = [
        "--ttbar",
        str(test_file),
        "--zprime",
        str(zprime_file),
        "-t",
        "MockTagger",
        "--fc",
        "0.01",
        "-n",
        "10_000",
    ]
    args.extend(["-d"] + [str(x) for x in disc_vals])

    output = main(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fc"] == 0.01
    assert output["MockTagger"]["ftau"] == 0.00
    assert "ttbar" in output["MockTagger"]
    assert "zprime" in output["MockTagger"]

    ttbar = output["MockTagger"]["ttbar"]
    zprime = output["MockTagger"]["zprime"]

    for dval in disc_vals:
        for out in [ttbar, zprime]:
            assert str(dval) in out
            assert "eff" in out[str(dval)]
            assert "rej" in out[str(dval)]


def test_output_file(test_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        output = str(tmpdir) + "/output.yaml"
        args = [
            "--ttbar",
            str(test_file),
            "-t",
            "MockTagger",
            "--fc",
            "0.01",
            "-n",
            "10_000",
            "-d",
            "1.0",
            "-o",
            str(output),
        ]

        main(args)
        assert Path(output).exists()


def test_wps_args_check(test_file):
    base_args = ["--ttbar", str(test_file), "-t", "MockTagger", "--effs", "0.1"]
    args = [*base_args, "--disc_cuts", "0.2"]
    with pytest.raises(ValueError, match="both --effs and --disc_cuts"):
        main(args)

    args = [*base_args, "--fhcc", "0.2"]
    with pytest.raises(ValueError, match="For single-b tagging, ftop, fhbb and fhcc should not"):
        main(args)

    base_args += ["--xbb"]
    args = base_args
    with pytest.raises(ValueError, match="Xbb tagging only supports hbb or hcc signal flavours"):
        main(args)

    args = [*base_args, "-s", "hbb", "--fc", "0.1"]
    with pytest.raises(
        ValueError, match="For Xbb tagging, fb, fc and ftau should not be specified"
    ):
        main(args)

    args = [*base_args, "-s", "hcc", "--fhcc", "0.25"]
    with pytest.raises(ValueError, match="For Xbb tagging, ftop should be specified"):
        main(args)
