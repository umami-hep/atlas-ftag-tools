import pytest

from ftag.mock import get_mock_file
from ftag.wps.working_points import get_working_points


@pytest.fixture
def ttbar_file():
    yield get_mock_file(10_000)[0]


@pytest.fixture
def zprime_file():
    yield get_mock_file(10_000)[0]


def test_get_working_points(ttbar_file):
    args = [
        "--ttbar",
        str(ttbar_file),
        "-t",
        "MockTagger",
        "-f",
        "0.01",
        "-e",
        "60",
        "-n",
        "10_000",
    ]
    output = get_working_points(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fx"] == 0.01
    assert "60" in output["MockTagger"]
    assert "cut_value" in output["MockTagger"]["60"]
    assert "ttbar" in output["MockTagger"]["60"]
    assert "eff" in output["MockTagger"]["60"]["ttbar"]
    assert "rej" in output["MockTagger"]["60"]["ttbar"]


def test_get_working_points_cjets(ttbar_file):
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
        "60",
        "-n",
        "10_000",
    ]
    output = get_working_points(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "cjets"
    assert output["MockTagger"]["fx"] == 0.01
    assert "60" in output["MockTagger"]
    assert "cut_value" in output["MockTagger"]["60"]
    assert "ttbar" in output["MockTagger"]["60"]
    assert "eff" in output["MockTagger"]["60"]["ttbar"]
    assert "rej" in output["MockTagger"]["60"]["ttbar"]


def test_get_working_points_zprime(ttbar_file, zprime_file):
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
        "60",
        "-n",
        "10_000",
    ]
    output = get_working_points(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fx"] == 0.15
    assert "60" in output["MockTagger"]
    assert "cut_value" in output["MockTagger"]["60"]
    assert "ttbar" in output["MockTagger"]["60"]
    assert "eff" in output["MockTagger"]["60"]["ttbar"]
    assert "rej" in output["MockTagger"]["60"]["ttbar"]
    assert "zprime" in output["MockTagger"]["60"]
    assert "eff" in output["MockTagger"]["60"]["zprime"]
    assert "rej" in output["MockTagger"]["60"]["zprime"]
