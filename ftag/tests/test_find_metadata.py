from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import requests
import yaml

from ftag import find_metadata


class TestFindMetadata(unittest.TestCase):
    #
    # === 1. TaskID/Container/Campaign Extraction Functions ===
    #
    def test_extract_taskid_from_filename(self):
        self.assertEqual(
            find_metadata.extract_taskid_from_filename(Path("user.test.12345678._000001.h5")),
            "12345678",
        )

    def test_extract_taskid_fail(self):
        self.assertIsNone(find_metadata.extract_taskid_from_filename(Path("badname.h5")))

    def test_parse_line_from_taskname(self):
        dsid, etag = find_metadata.parse_line_from_taskname("user.123456.e1234_tid123")
        self.assertEqual((dsid, etag), (123456, "e1234"))

    def test_parse_line_from_taskname_fail(self):
        self.assertEqual(find_metadata.parse_line_from_taskname("not.match"), (None, None))

    def test_parse_campaign(self):
        self.assertEqual(find_metadata.parse_campaign_from_taskname("mc20_13TeV"), "mc16")
        self.assertEqual(find_metadata.parse_campaign_from_taskname("mc21_13TeV"), "mc21")
        self.assertIsNone(find_metadata.parse_campaign_from_taskname("data18"))

    def test_extract_info_from_container(self):
        self.assertEqual(
            find_metadata.extract_info_from_container("mc20_13TeV.123456.e7890_s1234_r5678"),
            (123456, "e7890", "mc16"),
        )
        self.assertIsNone(find_metadata.extract_info_from_container("bad.container"))

    def test_extract_mc_container_from_json(self):
        data = {"key": "mc21_13TeV.123456.e3456_s2345"}
        self.assertIn("mc21_13TeV", find_metadata.extract_mc_container_from_json(data))
        self.assertIsNone(find_metadata.extract_mc_container_from_json({"no": "match"}))

    #
    # === 2. URL Validation ===
    #
    def test_validate_url_scheme(self):
        self.assertIsNotNone(find_metadata.validate_url_scheme("https://abc.com"))
        with self.assertRaises(ValueError):
            find_metadata.validate_url_scheme("ftp://bad")

    #
    # === 3. Xsec Database Download Logic ===
    #
    def test_download_xsecdb_files_invalid_scheme(self):
        with (
            patch("ftag.find_metadata.Path.exists", return_value=False),
            patch("ftag.find_metadata.validate_url_scheme", side_effect=ValueError("bad")),
            patch("ftag.find_metadata.requests.get"),
        ):
            find_metadata.download_xsecdb_files()

    def test_download_xsecdb_files_network_fail(self):
        with (
            patch("ftag.find_metadata.Path.exists", return_value=False),
            patch("ftag.find_metadata.requests.get", side_effect=requests.RequestException("fail")),
        ):
            find_metadata.download_xsecdb_files()

    def test_download_xsecdb_files_success(self):
        with (
            patch("ftag.find_metadata.Path.exists", return_value=False),
            patch("ftag.find_metadata.validate_url_scheme") as mock_validate,
            patch("ftag.find_metadata.requests.get") as mock_get,
            patch("ftag.find_metadata.Path.write_bytes") as mock_write,
            patch("builtins.print") as mock_print,
        ):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b"mock file content"
            mock_get.return_value = mock_response

            find_metadata.download_xsecdb_files()

            self.assertTrue(mock_validate.called)
            self.assertTrue(mock_get.called)
            self.assertTrue(mock_response.raise_for_status.called)
            self.assertTrue(mock_write.called)
            mock_print.assert_any_call("Downloaded: PMGxsecDB_mc15.txt")

    #
    # === 4. BigPanDA Query ===
    #
    def test_fetch_taskinfo_success(self):
        mock_data = [
            {"taskname": "user.123456.e7890_tid123", "inputdataset": "mc20_13TeV.123456.e7890"}
        ]
        with patch("ftag.find_metadata.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_data
            out = find_metadata.fetch_taskinfo_from_bigpanda("12345678")
            self.assertEqual(out, mock_data[0])

    def test_fetch_taskinfo_empty(self):
        with patch("ftag.find_metadata.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = []
            self.assertIsNone(find_metadata.fetch_taskinfo_from_bigpanda("123"))

    def test_fetch_taskinfo_jsonfail(self):
        with patch("ftag.find_metadata.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.side_effect = json.JSONDecodeError("fail", "doc", 0)
            self.assertIsNone(find_metadata.fetch_taskinfo_from_bigpanda("123"))

    #
    # === 5. Xsec Database Query ===
    #
    def test_query_xsecdb(self):
        line = "123456 x 2.2 0.9 1.1 x x x e9999\n"
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write(line)
            f.flush()
            with patch("ftag.find_metadata.XSECDB_MAP", {"mc16": f.name}):
                r = find_metadata.query_xsecdb("mc16", 123456, "e9999")
                self.assertEqual(r["cross_section_pb"], 2.2)
                self.assertEqual(r["genFiltEff"], 0.9)
                self.assertEqual(r["kfactor"], 1.1)

    def test_query_xsecdb_invalid_line(self):
        line = "123456 a b\n"
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write(line)
            f.flush()
            with patch("ftag.find_metadata.XSECDB_MAP", {"mc16": f.name}):
                r = find_metadata.query_xsecdb("mc16", 123456, "e0000")
                self.assertIsNone(r)

    def test_query_xsecdb_dbfile_missing(self):
        with (
            patch("ftag.find_metadata.Path.exists", return_value=False),
            patch("builtins.print") as mock_print,
        ):
            r1 = find_metadata.query_xsecdb("unknown_campaign", 123456, "e1234")
            self.assertIsNone(r1)
            with patch("ftag.find_metadata.XSECDB_MAP", {"mc16": "nonexistent_file.txt"}):
                r2 = find_metadata.query_xsecdb("mc16", 123456, "e1234")
                self.assertIsNone(r2)
            mock_print.assert_any_call(
                "ERROR: Database file for campaign 'unknown_campaign' not found."
            )
            mock_print.assert_any_call("ERROR: Database file for campaign 'mc16' not found.")

    def test_query_xsecdb_skips_comment_and_blank_lines(self):
        content = "\n# comment line\n123456 x 2.2 0.9 1.1 x x x e1234\n"
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write(content)
            f.flush()
            with patch("ftag.find_metadata.XSECDB_MAP", {"mc16": f.name}):
                result = find_metadata.query_xsecdb("mc16", 123456, "e1234")
                self.assertIsNotNone(result)
                self.assertEqual(result["cross_section_pb"], 2.2)
                self.assertEqual(result["genFiltEff"], 0.9)
                self.assertEqual(result["kfactor"], 1.1)

    #
    # === 6. Metadata Writing ===
    #
    def test_write_metadata_to_h5(self):
        meta = {"cross_section_pb": 2.1, "genFiltEff": 0.5, "kfactor": 1.3}
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            find_metadata.write_metadata_to_h5(f.name, 123456, meta)
            with h5py.File(f.name, "r") as h5:
                g = h5["metadata/123456"]
                self.assertEqual(g["cross_section_pb"][()], 2.1)

    def test_write_metadata_overwrites_existing_keys(self):
        meta1 = {"cross_section_pb": 1.0, "genFiltEff": 0.5, "kfactor": 1.1}
        meta2 = {"cross_section_pb": 2.0, "genFiltEff": 0.6, "kfactor": 1.2}

        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            find_metadata.write_metadata_to_h5(f.name, 123456, meta1)
            find_metadata.write_metadata_to_h5(f.name, 123456, meta2)

            with h5py.File(f.name, "r") as h5:
                g = h5["metadata/123456"]
                self.assertEqual(g["cross_section_pb"][()], 2.0)
                self.assertEqual(g["genFiltEff"][()], 0.6)
                self.assertEqual(g["kfactor"][()], 1.2)

    #
    # === 7. YAML Fallback Path ===
    #
    def test_handle_yaml_fallback_manual(self):
        data = {"123456": {"cross_section_pb": 1, "genFiltEff": 1, "kfactor": 1}}
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            find_metadata.handle_yaml_fallback(Path(f.name), data)
            with h5py.File(f.name, "r") as h5:
                self.assertIn("123456", h5["metadata"])

    def test_handle_yaml_fallback_container_parse_fail(self):
        data = {"container name": "bad.container.name"}
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with self.assertRaises(ValueError) as ctx:
                find_metadata.handle_yaml_fallback(Path(f.name), data)
            self.assertIn("Failed to parse valid DSID", str(ctx.exception))

    def test_handle_yaml_fallback_container_not_in_db(self):
        container = "mc21_13TeV.123456.e7890_s1234_r5678"
        data = {"container name": container}
        with (
            tempfile.NamedTemporaryFile(suffix=".h5") as f,
            patch(
                "ftag.find_metadata.extract_info_from_container",
                return_value=(123456, "e7890", "mc21"),
            ),
            patch("ftag.find_metadata.query_xsecdb", return_value=None),
        ):
            with self.assertRaises(ValueError) as ctx:
                find_metadata.handle_yaml_fallback(Path(f.name), data)
            self.assertIn("Container not found", str(ctx.exception))

    def test_handle_yaml_fallback_multiple_entries_raises(self):
        bad_yaml = {
            "123456": {"cross_section_pb": 1, "genFiltEff": 1, "kfactor": 1},
            "123457": {"cross_section_pb": 2, "genFiltEff": 2, "kfactor": 2},
        }
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with self.assertRaises(ValueError) as ctx:
                find_metadata.handle_yaml_fallback(Path(f.name), bad_yaml)
            self.assertIn("YAML fallback file must contain exactly one entry", str(ctx.exception))

    def test_handle_yaml_fallback_empty_yaml_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with self.assertRaises(ValueError) as ctx:
                find_metadata.handle_yaml_fallback(Path(f.name), {})
            self.assertIn("YAML fallback file must contain exactly one entry", str(ctx.exception))

    def test_handle_yaml_fallback_container_not_string(self):
        bad_yaml = {"container name": ["not", "a", "string"]}
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with self.assertRaises(ValueError) as ctx:
                find_metadata.handle_yaml_fallback(Path(f.name), bad_yaml)
            self.assertIn("'container name' must be a string", str(ctx.exception))

    def test_handle_yaml_fallback_invalid_dsid_key_raises(self):
        bad_yaml = {"not_a_dsid": {"cross_section_pb": 1, "genFiltEff": 1, "kfactor": 1}}
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with self.assertRaises(ValueError) as ctx:
                find_metadata.handle_yaml_fallback(Path(f.name), bad_yaml)
            self.assertIn("YAML key must be a valid DSID or 'container name'", str(ctx.exception))

    def test_handle_yaml_fallback_manual_missing_keys(self):
        bad_yaml = {"123456": {"cross_section_pb": 0.1, "genFiltEff": 0.9}}
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with self.assertRaises(ValueError) as ctx:
                find_metadata.handle_yaml_fallback(Path(f.name), bad_yaml)
            self.assertIn("Manual metadata must include", str(ctx.exception))

    def test_handle_yaml_fallback_manual_not_dict(self):
        bad_yaml = {"123456": "this is not a dict"}
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with self.assertRaises(ValueError) as ctx:
                find_metadata.handle_yaml_fallback(Path(f.name), bad_yaml)
            self.assertIn("Manual metadata must include", str(ctx.exception))

    #
    # === 8. CLI Argument Parsing ===
    #
    def test_parse_args_and_yaml_no_fallback(self):
        test_args = ["find_metadata.py", "a.h5", "b.h5"]
        with patch("sys.argv", test_args):
            h5_files, yaml_data = find_metadata.parse_args_and_yaml()
            self.assertEqual(h5_files, ["a.h5", "b.h5"])
            self.assertEqual(yaml_data, {})

    def test_parse_args_and_yaml_with_fallback(self):
        test_yaml = {"123456": {"cross_section_pb": 1.1, "genFiltEff": 0.9, "kfactor": 1.0}}
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp_yaml:
            yaml.dump(test_yaml, tmp_yaml)
            tmp_yaml_path = tmp_yaml.name
        test_args = ["find_metadata.py", "file1.h5", "-m", tmp_yaml_path]
        with patch("sys.argv", test_args):
            h5_files, yaml_data = find_metadata.parse_args_and_yaml()
            self.assertEqual(h5_files, ["file1.h5"])
            self.assertEqual(yaml_data, test_yaml)

    #
    # === 9. Main Flow/Integration Tests ===
    #
    def test_main_cli_path(self):
        with (
            patch("ftag.find_metadata.parse_args_and_yaml", return_value=(["a.h5"], {})),
            patch("ftag.find_metadata.download_xsecdb_files"),
            patch("ftag.find_metadata.process_single_file"),
            patch("ftag.find_metadata.Path.exists", return_value=True),
            patch("ftag.find_metadata.Path.unlink", side_effect=OSError("fail")),
        ):
            find_metadata.main()

    def test_process_single_file_auto_success(self):
        mock_taskinfo = {
            "taskname": "user.123456.e7890_tid123",
            "inputdataset": "mc20_13TeV.123456.e7890_s1234_r5678",
        }
        meta = {"cross_section_pb": 3.0, "genFiltEff": 0.5, "kfactor": 1.2, "etag": "e7890"}
        with (
            tempfile.NamedTemporaryFile(suffix=".h5") as f,
            patch("ftag.find_metadata.Path.exists", return_value=True),
            patch("ftag.find_metadata.extract_taskid_from_filename", return_value="12345678"),
            patch("ftag.find_metadata.fetch_taskinfo_from_bigpanda", return_value=mock_taskinfo),
            patch("ftag.find_metadata.parse_line_from_taskname", return_value=(123456, "e7890")),
            patch(
                "ftag.find_metadata.extract_mc_container_from_json",
                return_value=mock_taskinfo["inputdataset"],
            ),
            patch("ftag.find_metadata.parse_campaign_from_taskname", return_value="mc16"),
            patch("ftag.find_metadata.query_xsecdb", return_value=meta.copy()),
            patch("ftag.find_metadata.write_metadata_to_h5") as mock_write,
            patch("builtins.print") as mock_print,
        ):
            find_metadata.process_single_file(Path(f.name), yaml_data={})
            mock_write.assert_called_once_with(str(f.name), 123456, {**meta, "campaign": "mc16"})
            mock_print.assert_any_call("Extracted Task ID: 12345678")

    def test_process_single_file_yaml_fallback_fails(self):
        path = Path("some.12345678._000001.h5")
        yaml_data = {"container name": "bad.container"}
        with (
            patch("ftag.find_metadata.Path.exists", return_value=True),
            patch("ftag.find_metadata.extract_taskid_from_filename", return_value=None),
            patch("builtins.print") as mock_print,
        ):
            find_metadata.process_single_file(path, yaml_data)
            mock_print.assert_any_call("Using YAML fallback path")
            self.assertTrue(
                any("YAML fallback failed:" in c[0][0] for c in mock_print.call_args_list)
            )

    def test_process_single_file_no_taskid_no_fallback(self):
        path = Path("noid.h5")
        with (
            patch("ftag.find_metadata.Path.exists", return_value=True),
            patch("ftag.find_metadata.extract_taskid_from_filename", return_value=None),
            patch("builtins.print") as mock_print,
        ):
            find_metadata.process_single_file(path, yaml_data={})
            mock_print.assert_any_call("Failed to retrieve metadata and no YAML fallback provided")

    def test_process_single_file_path_not_exists(self):
        fake_path = Path("nonexistent_file.h5")
        with (
            patch("ftag.find_metadata.Path.exists", return_value=False),
            patch("builtins.print") as mock_print,
            patch("ftag.find_metadata.extract_taskid_from_filename") as mock_extract,
            patch("ftag.find_metadata.fetch_taskinfo_from_bigpanda") as mock_fetch,
        ):
            find_metadata.process_single_file(fake_path, yaml_data={})
            mock_print.assert_any_call(f"File not found: {fake_path}")
            mock_extract.assert_not_called()
            mock_fetch.assert_not_called()

    def test_process_single_file_write_metadata_fails(self):
        mock_taskinfo = {
            "taskname": "user.123456.e7890_tid123",
            "inputdataset": "mc20_13TeV.123456.e7890_s1234_r5678",
        }
        meta = {"cross_section_pb": 1.0, "genFiltEff": 1.0, "kfactor": 1.0, "etag": "e7890"}

        with (
            tempfile.NamedTemporaryFile(suffix=".h5") as f,
            patch("ftag.find_metadata.Path.exists", return_value=True),
            patch("ftag.find_metadata.extract_taskid_from_filename", return_value="12345678"),
            patch("ftag.find_metadata.fetch_taskinfo_from_bigpanda", return_value=mock_taskinfo),
            patch("ftag.find_metadata.parse_line_from_taskname", return_value=(123456, "e7890")),
            patch(
                "ftag.find_metadata.extract_mc_container_from_json",
                return_value=mock_taskinfo["inputdataset"],
            ),
            patch("ftag.find_metadata.parse_campaign_from_taskname", return_value="mc16"),
            patch("ftag.find_metadata.query_xsecdb", return_value=meta),
            patch(
                "ftag.find_metadata.write_metadata_to_h5",
                side_effect=OSError("simulated write failure"),
            ),
            patch("builtins.print") as mock_print,
        ):
            find_metadata.process_single_file(Path(f.name), yaml_data={})
            self.assertTrue(
                any(
                    "Failed to write metadata: simulated write failure" in call.args[0]
                    for call in mock_print.call_args_list
                )
            )

    def test_main_deletes_tempfile_success(self):
        with (
            patch("ftag.find_metadata.parse_args_and_yaml", return_value=(["a.h5"], {})),
            patch("ftag.find_metadata.download_xsecdb_files"),
            patch("ftag.find_metadata.process_single_file"),
            patch("ftag.find_metadata.Path.exists", return_value=True),
            patch("ftag.find_metadata.Path.unlink"),
            patch("builtins.print") as mock_print,
        ):
            find_metadata.main()
            calls = [
                f"Deleted temporary file: {Path(name)}"
                for name in find_metadata.XSECDB_MAP.values()
            ]
            for c in calls:
                assert any(c in p.args[0] for p in mock_print.call_args_list)


if __name__ == "__main__":
    unittest.main()
