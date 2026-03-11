# ruff: noqa: SLF001
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import pytest

from ftag.find_metadata import MetadataFinder

# === 1. Mock Data Setup ===

# Ensure etag is in the 9th column (index 8) to satisfy len(parts) >= 9
# Format: DSID  Name  Xsec  Eff  k  ?  ?  ?  etag
MOCK_XSEC_LINE = "601229  Pythia8_jetjet  0.015  1.0  1.1  0.0  x  x  e8514\n"

MOCK_PANDA_INFO = [
    {
        "jeditaskid": 12345678,
        "taskname": "mc23_13TeV.601229.PhPy8.e8514_s4162/",
    }
]


@pytest.fixture
def mock_h5(tmp_path: Path) -> Path:
    p = tmp_path / "user.test.12345678.output.h5"
    with h5py.File(p, "w") as f:
        f.create_group("metadata")
    return p


# === 2. Test Class ===


class TestMetadataFinder:
    # --- Basic Parsing Tests ---
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("user.test.12345678._000001.h5", "12345678"),
            ("mc23.11040412.h5", "11040412"),
            ("no_id_here.h5", None),
        ],
    )
    def test_extract_taskid(self, tmp_path, filename, expected):
        p = tmp_path / filename
        finder = MetadataFinder(p)
        assert finder._extract_taskid() == expected

    @pytest.mark.parametrize(
        ("container", "expected"),
        [
            ("mc20_13TeV.123456.e7890", (123456, "e7890", "mc16")),
            ("mc23_13TeV.601229.e8514", (601229, "e8514", "mc23")),
            ("bad.name", None),
        ],
    )
    def test_parse_info(self, mock_h5, container, expected):
        finder = MetadataFinder(mock_h5)
        result = finder._parse_info(container)
        assert result == expected

    # --- Network and Database Logic Tests ---
    @patch("requests.get")
    def test_fetch_taskinfo_success(self, mock_get, mock_h5):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = MOCK_PANDA_INFO
        finder = MetadataFinder(mock_h5)
        info = finder._fetch_taskinfo("12345678")
        assert info["jeditaskid"] == 12345678

    @patch("requests.get")
    def test_fetch_taskinfo_empty(self, mock_get, mock_h5):
        """Cover the branch where _fetch_taskinfo returns None."""
        mock_get.return_value.json.return_value = []
        finder = MetadataFinder(mock_h5)
        assert finder._fetch_taskinfo("12345") is None

    def test_query_xsecdb_logic(self, mock_h5, tmp_path):
        """Test the normal query logic (fixes the 9-column data issue)."""
        db_file = tmp_path / "PMGxsecDB_mc23.txt"
        db_file.write_text("# comment\n" + MOCK_XSEC_LINE)

        finder = MetadataFinder(mock_h5)
        with patch.object(finder, "_download_db", return_value=str(db_file)):
            meta = finder._query_xsecdb("mc23", 601229, "e8514")
            assert meta["cross_section_pb"] == 0.015
            assert meta["genFiltEff"] == 1.0

    def test_query_xsecdb_not_found(self, mock_h5, tmp_path):
        """Cover branch where _query_xsecdb returns None when no data is found."""
        db_file = tmp_path / "PMGxsecDB_mc23.txt"
        db_file.write_text(MOCK_XSEC_LINE)
        finder = MetadataFinder(mock_h5)
        with patch.object(finder, "_download_db", return_value=str(db_file)):
            # DSID matches but etag does not
            assert finder._query_xsecdb("mc23", 601229, "wrong_etag") is None

    # --- Core Injection Flow Tests ---
    @patch("requests.get")
    def test_inject_metadata_full_chain(self, mock_get, mock_h5, tmp_path):
        """Test complete successful injection chain, ensuring HDF5 write is covered."""
        # Mock BigPanDA response
        mock_panda_resp = MagicMock()
        mock_panda_resp.json.return_value = MOCK_PANDA_INFO

        # Mock DB download response
        mock_db_resp = MagicMock()
        mock_db_resp.content = MOCK_XSEC_LINE.encode()

        mock_get.side_effect = [mock_panda_resp, mock_db_resp]

        # Let the file actually be written to pytest's auto-cleaned temporary directory
        db_file = tmp_path / "PMGxsecDB_mc23.txt"

        with patch.dict("ftag.find_metadata.XSECDB_MAP", {"mc23": str(db_file)}):
            finder = MetadataFinder(mock_h5)
            finder.inject_metadata()

        # Verify results
        with h5py.File(mock_h5, "r") as f:
            assert "metadata/601229/cross_section_pb" in f
            assert f["metadata/601229/cross_section_pb"][()] == 0.015

    def test_inject_metadata_overwrite(self, mock_h5):
        """Ensure the overwrite logic 'if k in g: del g[k]' is executed."""
        dsid = 601229
        with h5py.File(mock_h5, "a") as f:
            g = f.require_group(f"metadata/{dsid}")
            g.create_dataset("cross_section_pb", data=999.0)

        finder = MetadataFinder(mock_h5)
        mock_meta = {"cross_section_pb": 0.015}

        # Mock prerequisite steps to directly test the injection part
        with (
            patch.object(finder, "_extract_taskid", return_value="12"),
            patch.object(finder, "_fetch_taskinfo", return_value={"t": "m"}),
            patch.object(finder, "_extract_container", return_value="mc23_13TeV.601229.e8514"),
            patch.object(finder, "_query_xsecdb", return_value=mock_meta),
        ):
            finder.inject_metadata()

        with h5py.File(mock_h5, "r") as f:
            assert f[f"metadata/{dsid}/cross_section_pb"][()] == 0.015

    # --- Exception Branch Tests ---
    def test_inject_metadata_fail_branches(self, mock_h5, capsys):
        finder = MetadataFinder(mock_h5)

        # 1. No TaskID
        with patch.object(finder, "_extract_taskid", return_value=None):
            finder.inject_metadata()
            assert "No Task ID found" in capsys.readouterr().out

        # 2. No task info
        with (
            patch.object(finder, "_extract_taskid", return_value="12345678"),
            patch.object(finder, "_fetch_taskinfo", return_value=None),
        ):
            finder.inject_metadata()
            assert "No BigPanDA info found" in capsys.readouterr().out

        # 3. Container parsing failed
        with (
            patch.object(finder, "_extract_taskid", return_value="12"),
            patch.object(finder, "_fetch_taskinfo", return_value={"t": "m"}),
            patch.object(finder, "_extract_container", return_value="invalid"),
            patch.object(finder, "_parse_info", return_value=None),
        ):
            finder.inject_metadata()
            assert "Failed to parse DSID" in capsys.readouterr().out

        # 4. No result in database
        with (
            patch.object(finder, "_extract_taskid", return_value="12"),
            patch.object(finder, "_fetch_taskinfo", return_value={"t": "m"}),
            patch.object(finder, "_extract_container", return_value="mc23_13TeV.601229.e8514"),
            patch.object(finder, "_query_xsecdb", return_value=None),
        ):
            finder.inject_metadata()
            assert "No metadata found in PMG DB" in capsys.readouterr().out
