from __future__ import annotations

import json
import sys
import urllib.error
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from ftag import find_metadata, mock
from ftag.find_metadata import (
    download_xsecdb_files,
    main,
    parse_line,
    process_container_list,
    query_xsecdb,
    write_metadata_to_h5,
)
from ftag.mock import get_mock_file

# ------------------------
# parse_line() tests
# ------------------------


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


# ------------------------
# query_xsecdb() tests
# ------------------------


def test_query_xsecdb_missing_file():
    with patch("ftag.find_metadata.Path.exists", return_value=False):
        result = query_xsecdb("mc16", 301200, "e5270")
        assert result is None


def test_query_xsecdb_with_match(tmp_path):
    fake_file = tmp_path / "PMGxsecDB_mc16.txt"
    content = "301200 dummy 1.23 0.95 1.1 x y z e5270\n"
    fake_file.write_text(content)

    find_metadata.XSECDB_MAP["mc16"] = str(fake_file)

    result = query_xsecdb("mc16", 301200, "e5270")
    assert result is not None
    assert result["cross_section_pb"] == pytest.approx(1.23)
    assert result["genFiltEff"] == pytest.approx(0.95)
    assert result["kfactor"] == pytest.approx(1.1)


def test_query_xsecdb_no_match(tmp_path):
    fake_file = tmp_path / "PMGxsecDB_mc16.txt"
    fake_file.write_text("999999 dummy 1.0 1.0 1.0 x x x e1234\n")
    find_metadata.XSECDB_MAP["mc16"] = str(fake_file)
    result = query_xsecdb("mc16", 301200, "e5270")
    assert result is None


def test_query_xsecdb_bad_format(tmp_path):
    fake_file = tmp_path / "PMGxsecDB_mc16.txt"
    fake_file.write_text("bad line\n")
    find_metadata.XSECDB_MAP["mc16"] = str(fake_file)
    result = query_xsecdb("mc16", 301200, "e5270")
    assert result is None


# ------------------------
# write_metadata_to_h5() test using mock file
# ------------------------


@patch("ftag.mock.mock_jets")
def test_write_metadata_to_h5_real_file(mock_mock_jets):
    dummy_jets = u2s(
        np.array([[1.0] * len(mock.JET_VARS)], dtype=np.float32).repeat(10, axis=0),
        dtype=np.dtype(mock.JET_VARS),
    )
    mock_mock_jets.return_value = dummy_jets

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


def test_write_metadata_to_h5_error():
    with patch("ftag.find_metadata.h5py.File", side_effect=OSError("fail")), pytest.raises(
        OSError, match="fail"
    ):
        write_metadata_to_h5(
            "bad.h5",
            {"123": {"cross_section_pb": 1.0, "genFiltEff": 1.0, "kfactor": 1.0}},
        )


# ------------------------
# process_container_list() integration test
# ------------------------


def test_process_container_list_integration(tmp_path):
    input_file = tmp_path / "containers.txt"
    output_json = tmp_path / "output.json"
    output_h5 = tmp_path / "output.h5"

    input_file.write_text("mc16_13TeV.301200.Sherpa_ttbar.e5270_\n")

    db_file = tmp_path / "PMGxsecDB_mc16.txt"
    db_file.write_text("301200 dummy 1.23 0.95 1.1 x y z e5270\n")

    find_metadata.XSECDB_MAP["mc16"] = str(db_file)

    process_container_list(str(input_file), str(output_json), str(output_h5))

    assert output_json.exists()
    assert "301200" in output_json.read_text()

    with h5py.File(output_h5, "r") as f:
        assert "301200" in f["metadata"]


def test_process_container_list_parse_fail(tmp_path):
    input_file = tmp_path / "bad.txt"
    output_json = tmp_path / "output.json"
    input_file.write_text("invalid_container_name\n")
    process_container_list(str(input_file), str(output_json))
    assert output_json.exists()


def test_process_container_list_query_fail(tmp_path):
    input_file = tmp_path / "queryfail.txt"
    output_json = tmp_path / "output.json"
    input_file.write_text("mc16_13TeV.301200.Sherpa_ttbar.e5270_\n")
    db_file = tmp_path / "PMGxsecDB_mc16.txt"
    db_file.write_text("999999 dummy 1.0 1.0 1.0 x x x e1234\n")
    find_metadata.XSECDB_MAP["mc16"] = str(db_file)
    process_container_list(str(input_file), str(output_json))
    assert output_json.exists()


def test_process_container_list_error_log(tmp_path):
    input_file = tmp_path / "bad2.txt"
    output_json = tmp_path / "metadata.json"
    input_file.write_text("bad_container\n" "mc16_13TeV.301200.Sherpa_ttbar.e5270_")
    db_file = tmp_path / "PMGxsecDB_mc16.txt"
    db_file.write_text("999999 dummy 1.0 1.0 1.0 x x x e0000\n")
    find_metadata.XSECDB_MAP["mc16"] = str(db_file)
    process_container_list(str(input_file), str(output_json))
    assert (tmp_path / "metadata_errors.log").exists()


# ------------------------
# download_xsecdb_files() test
# ------------------------


def test_download_xsecdb_files():
    with patch("ftag.find_metadata.Path.exists", return_value=False), patch(
        "ftag.find_metadata.urllib.request.urlretrieve"
    ) as mock_urlretrieve:
        download_xsecdb_files()
        assert mock_urlretrieve.call_count == 4


def test_download_xsecdb_files_fail():
    with patch("ftag.find_metadata.Path.exists", return_value=False), patch(
        "ftag.find_metadata.urllib.request.urlretrieve",
        side_effect=urllib.error.URLError("fail"),
    ) as mock_urlretrieve:
        download_xsecdb_files()
        assert mock_urlretrieve.call_count == 4


# ------------------------
# CLI main() test
# ------------------------


def test_main_invokes_all(tmp_path, monkeypatch):
    containers = tmp_path / "containers.txt"
    containers.write_text("mc16_13TeV.301200.Sherpa_ttbar.e5270_\n")
    db_file = tmp_path / "PMGxsecDB_mc16.txt"
    db_file.write_text("301200 dummy 1.23 0.95 1.1 x y z e5270\n")
    find_metadata.XSECDB_MAP["mc16"] = str(db_file)

    output_json = tmp_path / "result.json"
    h5file = tmp_path / "merged.h5"

    monkeypatch.setattr(
        sys, "argv", ["script", "-i", str(containers), "-o", str(output_json), "-m", str(h5file)]
    )
    main()

    assert output_json.exists()
    with h5py.File(h5file) as f:
        assert "metadata" in f
        assert "301200" in f["metadata"]


def test_query_xsecdb_print_not_found(tmp_path, capsys):
    fake_file = tmp_path / "PMGxsecDB_mc16.txt"
    fake_file.write_text("301200 dummy 1.0 1.0 1.0 x x x e1234\n")
    find_metadata.XSECDB_MAP["mc16"] = str(fake_file)
    result = query_xsecdb("mc16", 301200, "e9999")
    captured = capsys.readouterr()
    assert "not found" in captured.out
    assert result is None


def test_write_metadata_to_h5_key_skip(tmp_path):
    h5file = tmp_path / "file.h5"
    with h5py.File(h5file, "w") as f:
        g = f.create_group("metadata/123")
        g.create_dataset("cross_section_pb", data=0.1)
    meta = {"123": {"cross_section_pb": 0.2, "genFiltEff": 0.3, "kfactor": 0.4}}
    write_metadata_to_h5(h5file, meta)
    with h5py.File(h5file, "r") as f:
        assert f["metadata/123/genFiltEff"][()] == pytest.approx(0.3)
        assert f["metadata/123/kfactor"][()] == pytest.approx(0.4)


def test_main_merge_fail(tmp_path, monkeypatch):
    container = tmp_path / "c.txt"
    db_file = tmp_path / "PMGxsecDB_mc16.txt"
    container.write_text("mc16_13TeV.301200.Sherpa_ttbar.e5270_\n")
    db_file.write_text("301200 dummy 1.0 1.0 1.0 x x x e5270\n")
    find_metadata.XSECDB_MAP["mc16"] = str(db_file)

    out_json = tmp_path / "x.json"
    out_h5 = tmp_path / "fail.h5"

    monkeypatch.setattr(
        sys,
        "argv",
        ["script", "-i", str(container), "-o", str(out_json), "-m", str(out_h5)],
    )

    with patch("ftag.find_metadata.h5py.File", side_effect=RuntimeError("merge fail")):
        main()


def test_query_xsecdb_partial_match_but_etag_fail(tmp_path, capsys):
    file = tmp_path / "PMGxsecDB_mc16.txt"
    file.write_text("301200 dummy 1.0 1.0 1.0 x x x e1111\n")
    find_metadata.XSECDB_MAP["mc16"] = str(file)
    query_xsecdb("mc16", 301200, "e9999")
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_process_container_list_missing_metadata_log(tmp_path, capsys):
    container_file = tmp_path / "missing.txt"
    container_file.write_text("mc16_13TeV.301200.Sherpa_ttbar.e0000_\n")
    db_file = tmp_path / "PMGxsecDB_mc16.txt"
    db_file.write_text("999999 dummy 1.0 1.0 1.0 x x x e1234\n")
    find_metadata.XSECDB_MAP["mc16"] = str(db_file)
    output_file = tmp_path / "meta.json"
    process_container_list(str(container_file), str(output_file))
    captured = capsys.readouterr()
    assert "lookup failed" in captured.out


def test_main_no_merge(monkeypatch, tmp_path):
    container_file = tmp_path / "c.txt"
    out_json = tmp_path / "o.json"
    db_file = tmp_path / "PMGxsecDB_mc16.txt"
    container_file.write_text("mc16_13TeV.301200.Sherpa_ttbar.e5270_\n")
    db_file.write_text("301200 dummy 1.23 0.95 1.1 x y z e5270\n")
    find_metadata.XSECDB_MAP["mc16"] = str(db_file)
    monkeypatch.setattr(sys, "argv", ["script", "-i", str(container_file), "-o", str(out_json)])
    main()
    assert out_json.exists()


def test_query_xsecdb_skip_comment_line(tmp_path):
    # Create a database file with a comment line
    fake_file = tmp_path / "PMGxsecDB_mc16.txt"
    fake_file.write_text("# this is a comment line\n301200 dummy 1.2 0.9 1.1 x x x e5270\n")

    # Override the db path
    find_metadata.XSECDB_MAP["mc16"] = str(fake_file)

    # Query existing data (should skip the comment and match)
    result = query_xsecdb("mc16", 301200, "e5270")
    assert result is not None
    assert result["cross_section_pb"] == pytest.approx(1.2)


def test_process_container_list_skips_empty_line(tmp_path):
    input_file = tmp_path / "containers.txt"
    output_file = tmp_path / "meta.json"
    input_file.write_text("\nmc16_13TeV.301200.Sherpa_ttbar.e5270_\n")

    db_file = tmp_path / "PMGxsecDB_mc16.txt"
    db_file.write_text("301200 dummy 1.0 1.0 1.0 x x x e5270\n")
    find_metadata.XSECDB_MAP["mc16"] = str(db_file)

    process_container_list(str(input_file), str(output_file))
    assert output_file.exists()
    content = json.loads(output_file.read_text())
    assert "301200" in content
