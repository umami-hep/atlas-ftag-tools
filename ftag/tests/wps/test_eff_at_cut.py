from __future__ import annotations
import tempfile
from pathlib import Path

import pytest
from ftag.mock import get_mock_file
from ftag.wps.eff_at_disc_cut import get_efficiencies


@pytest.fixture
def ttbar_file():
    yield get_mock_file(10_000)[0]


@pytest.fixture
def zprime_file():
    yield get_mock_file(10_000)[0]

def test_get_rej_eff_at_disc_ttbar(ttbar_file, disc_vals=[1.0, 1.5]):
    args = [
        "--ttbar",
        str(ttbar_file),
        "-t",
        "MockTagger",
        "-f",
        "0.01",
        "-n",
        "10_000",
    ]
    args.extend(["-d"] + [str(x) for x in disc_vals])

    output = get_efficiencies(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fx"] == (0.01,)
    assert "ttbar" in output["MockTagger"]
    ttbar = output["MockTagger"]["ttbar"]
    for dval in disc_vals:
        assert str(dval) in ttbar
        assert "eff" in ttbar[str(dval)]
        assert "rej" in ttbar[str(dval)]

    assert "zprime" not in output["MockTagger"]

def test_get_rej_eff_at_disc_zprime(ttbar_file, zprime_file, disc_vals=[1.0, 1.5]):
    args = [
        "--ttbar",
        str(ttbar_file),
        "--zprime",
        str(zprime_file),
        "-t",
        "MockTagger",
        "-f",
        "0.01",
        "-n",
        "10_000",
    ]
    args.extend(["-d"] + [str(x) for x in disc_vals])

    output = get_efficiencies(args)

    assert "MockTagger" in output
    assert output["MockTagger"]["signal"] == "bjets"
    assert output["MockTagger"]["fx"] == (0.01,)
    assert "ttbar" in output["MockTagger"]
    assert "zprime" in output["MockTagger"]

    ttbar = output["MockTagger"]["ttbar"]
    zprime = output["MockTagger"]["zprime"]

    for dval in disc_vals:
        for out in [ttbar, zprime]:
            assert str(dval) in out
            assert "eff" in out[str(dval)]
            assert "rej" in out[str(dval)]

def test_output_file(ttbar_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        print(str(tmpdir))
        output = str(tmpdir) + "/output.yaml"
        args = [
            "--ttbar",
            str(ttbar_file),
            "-t",
            "MockTagger",
            "-f",
            "0.01",
            "-n",
            "10_000",
            "-d",
            "1.0",
            "-o",
            str(output),
        ]
        
        get_efficiencies(args)
        assert Path(output).exists()
    
def test_get_working_points_fx_length_check():
    # test with incorrect length of fx values for regular b-tagging
    with pytest.raises(ValueError):
        get_efficiencies(["--ttbar", "path", "-t", "MockTagger", "-f", "0.1", "0.2", "-d", "1.0"])