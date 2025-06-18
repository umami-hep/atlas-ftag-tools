from __future__ import annotations

import h5py
import pytest

from ftag.find_metadata import parse_line, query_xsecdb, write_metadata_to_h5
from ftag.mock import get_mock_file

# parse_line() tests


def test_parse_line_valid():
    container = "mc16_13TeV.301200.Sherpa_ttbar.e5270_"
    raw_campaign, campaign, dsid, etag = parse_line(container)
    assert raw_campaign == "mc16"
    assert campaign == "mc16"
    assert dsid == 301200
    assert etag == "e5270"


def test_parse_line_invalid():
    container = "invalid_container_name"
    raw_campaign, campaign, dsid, etag = parse_line(container)
    assert raw_campaign is None
    assert campaign is None
    assert dsid is None
    assert etag is None


# query_xsecdb() tests


@pytest.mark.parametrize("campaign", ["mc15", "mc16", "mc21", "mc23"])
def test_query_xsecdb_missing_file(campaign):
    # Ensure no file exists
    result = query_xsecdb(campaign, 301200, "e5270")
    assert result is None


# write_metadata_to_h5() test using ftag.mock


def test_write_metadata_to_h5_real_file():
    mock_path, h5file = get_mock_file(num_jets=10)
    h5file.close()  # Close for writing

    fake_metadata = {
        "301200": {
            "cross_section_pb": 1.23,
            "genFiltEff": 0.95,
            "kfactor": 1.1,
        }
    }

    write_metadata_to_h5(mock_path, fake_metadata)

    with h5py.File(mock_path, "r") as f:
        assert "metadata" in f
        assert "301200" in f["metadata"]
        dsid_group = f["metadata"]["301200"]
        assert dsid_group["cross_section_pb"][()] == pytest.approx(1.23)
        assert dsid_group["genFiltEff"][()] == pytest.approx(0.95)
        assert dsid_group["kfactor"][()] == pytest.approx(1.1)
