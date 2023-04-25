import pytest

from ftag.mock import get_mock_file
from ftag.wps.working_points import get_working_points


@pytest.fixture
def ttbar_file():
    yield get_mock_file(10_000)[0]


@pytest.fixture
def zprime_file():
    yield get_mock_file(10_000)[0]


def test_get_working_points(ttbar_file, eff_val="60"):
    args = [
        "--ttbar",
        str(ttbar_file),
        "-t",
        "MockTagger",
        "-f",
        "0.01",
        "-e",
        eff_val,
        "-n",
        "10_000",
    ]
    output = get_working_points(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fx"] == 0.01
    assert eff_val in output["MockTagger"]
    assert "cut_value" in output["MockTagger"][eff_val]
    assert "ttbar" in output["MockTagger"][eff_val]
    assert "eff" in output["MockTagger"][eff_val]["ttbar"]
    assert "rej" in output["MockTagger"][eff_val]["ttbar"]
    assert output["MockTagger"][eff_val]["ttbar"]["eff"]["bjets"] == pytest.approx(
        float(eff_val) / 100, rel=1e-2
    )


def test_get_working_points_rejection(ttbar_file, rej_val="100"):
    args = [
        "--ttbar",
        str(ttbar_file),
        "-t",
        "MockTagger",
        "-f",
        "0.01",
        "-e",
        rej_val,
        "-n",
        "10_000",
        "-r",
        "ujets",
    ]
    output = get_working_points(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fx"] == 0.01
    assert rej_val in output["MockTagger"]
    assert "cut_value" in output["MockTagger"][rej_val]
    assert "ttbar" in output["MockTagger"][rej_val]
    assert "eff" in output["MockTagger"][rej_val]["ttbar"]
    assert "rej" in output["MockTagger"][rej_val]["ttbar"]
    assert output["MockTagger"][rej_val]["ttbar"]["eff"]["ujets"] == pytest.approx(
        1 / float(rej_val), rel=1e-1
    )


def test_get_working_points_cjets(ttbar_file, eff_val="60"):
    args = [
        "--ttbar",
        str(ttbar_file),
        "-t",
        "MockTagger",
        "-s",
        "cjets",
        "-f",
        "0.01",
        "-e",
        eff_val,
        "-n",
        "10_000",
    ]
    output = get_working_points(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "cjets"
    assert output["MockTagger"]["fx"] == 0.01
    assert eff_val in output["MockTagger"]
    assert "cut_value" in output["MockTagger"][eff_val]
    assert "ttbar" in output["MockTagger"][eff_val]
    assert "eff" in output["MockTagger"][eff_val]["ttbar"]
    assert "rej" in output["MockTagger"][eff_val]["ttbar"]
    assert output["MockTagger"][eff_val]["ttbar"]["eff"]["cjets"] == pytest.approx(
        float(eff_val) / 100, rel=1e-2
    )


def test_get_working_points_zprime(ttbar_file, zprime_file, eff_val="60"):
    args = [
        "--ttbar",
        str(ttbar_file),
        "--zprime",
        str(zprime_file),
        "-t",
        "MockTagger",
        "-f",
        "0.15",
        "-e",
        eff_val,
        "-n",
        "10_000",
    ]
    output = get_working_points(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fx"] == 0.15
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
