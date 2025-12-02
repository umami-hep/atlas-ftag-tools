from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, urlparse

import h5py
import requests
import yaml

"""
find_metadata.py

Injects metadata (cross_section_pb, genFiltEff, kfactor) into .h5 files produced by TDD.

Usage:
    python find_metadata.py <h5_file1> [<h5_file2> ...] [-m fallback.yaml]
"""

XSECDB_MAP: dict[str, str] = {
    "mc15": "PMGxsecDB_mc15.txt",
    "mc16": "PMGxsecDB_mc16.txt",
    "mc21": "PMGxsecDB_mc21.txt",
    "mc23": "PMGxsecDB_mc23.txt",
}

XSECDB_URL_BASE: str = "https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/PMGTools/"


def validate_url_scheme(url: str) -> ParseResult:
    """
    Validate the scheme of a given URL, ensuring it is http or https.

    Parameters
    ----------
    url : str
        URL string to validate.

    Returns
    -------
    ParseResult
        Parsed URL object.

    Raises
    ------
    ValueError
        If the URL scheme is not supported.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    return parsed


def download_xsecdb_files() -> None:
    """Download the PMG xsecDB files from CERN if they are not present locally."""
    for filename in XSECDB_MAP.values():
        if not Path(filename).exists():
            url = XSECDB_URL_BASE + filename
            try:
                validate_url_scheme(url)
                print(f"Downloading {filename} from {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                Path(filename).write_bytes(response.content)
                print(f"Downloaded: {filename}")
            except (requests.RequestException, ValueError) as e:
                print(f"ERROR: Failed to download {filename} - {e}")


def extract_taskid_from_filename(h5_path: Path) -> str | None:
    """Extract the BigPanDA Task ID (8-digit) from an HDF5 filename.

    Parameters
    ----------
    h5_path : Path
        Path object pointing to the .h5 file.

    Returns
    -------
    str | None
        The Task ID as a string if found, otherwise None.
    """
    m = re.search(r"\.(\d{8})\.", h5_path.name)
    return m.group(1) if m else None


def fetch_taskinfo_from_bigpanda(taskid: str) -> dict[str, Any] | None:
    """Fetch task information from BigPanDA for a given Task ID.

    Parameters
    ----------
    taskid : str
        BigPanDA task ID.

    Returns
    -------
    dict[str, Any] | None
        Task info as a dictionary if found, otherwise None.
    """
    url = f"https://bigpanda.cern.ch/tasks/?jeditaskid={taskid}&json"
    print(f"Fetching task info from: {url}")
    try:
        validate_url_scheme(url)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return data[0]
        print("No task info found.")
    except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: Failed to fetch task info for {taskid} - {e}")
    return None


def extract_mc_container_from_json(data: dict[str, Any]) -> str | None:
    """Extract the MC container name (e.g., mc16_13TeV.<something>) from a task JSON.

    Parameters
    ----------
    data : dict[str, Any]
        Task info dictionary from BigPanDA.

    Returns
    -------
    str | None
        The container string if found, otherwise None.
    """
    text = json.dumps(data)
    match = re.search(r"(mc\d+_13TeV\.[\w\.]+)", text)
    return match.group(1) if match else None


def parse_line_from_taskname(taskname: str) -> tuple[int | None, str | None]:
    """Extract DSID and etag from a task name string.

    Parameters
    ----------
    taskname : str
        Full task name.

    Returns
    -------
    tuple[int | None, str | None]
        A tuple of (DSID as int, etag as string), or (None, None) if not found.
    """
    match = re.match(r".*\.(\d{6})\.e(\d+)_", taskname)
    if match:
        dsid, etag = match.groups()
        return int(dsid), f"e{etag}"
    return None, None


def parse_campaign_from_taskname(taskname: str) -> str | None:
    """Derive campaign (mc15/mc16/etc.) from a task or container name.

    Parameters
    ----------
    taskname : str
        The name string.

    Returns
    -------
    str | None
        Campaign string, or None if not found.
    """
    m = re.match(r"(mc\d+)_13TeV", taskname)
    if m:
        raw_campaign = m.group(1)
        return "mc16" if raw_campaign == "mc20" else raw_campaign
    return None


def extract_info_from_container(container: str) -> tuple[int, str, str] | None:
    """Extract DSID, etag, and campaign name from a container string.

    Parameters
    ----------
    container : str
        The MC container string.

    Returns
    -------
    tuple[int, str, str] | None
        A tuple of (DSID, etag, campaign), or None if parsing fails.
    """
    m_campaign = re.search(r"\b(mc\d+)_13TeV", container)
    m_dsid = re.search(r"\.(\d{6})\.", container)
    m_etag = re.search(r"\.e(\d+)(?:[_.]|$)", container)
    if m_campaign and m_dsid and m_etag:
        raw_campaign = m_campaign.group(1)
        campaign = "mc16" if raw_campaign == "mc20" else raw_campaign
        dsid = int(m_dsid.group(1))
        etag = f"e{m_etag.group(1)}"
        return dsid, etag, campaign
    return None


def query_xsecdb(campaign: str, dsid: int, etag: str) -> dict[str, Any] | None:
    """Look up cross-section metadata in the PMG xsecDB.

    Parameters
    ----------
    campaign : str
        Campaign name (e.g., mc16).
    dsid : int
        Dataset ID.
    etag : str
        Event tag.

    Returns
    -------
    dict[str, Any] | None
        Dictionary with cross_section_pb, genFiltEff, kfactor, and etag if found, otherwise None.
    """
    db_path = XSECDB_MAP.get(campaign)
    print(f"Searching in {db_path} for DSID={dsid}, etag={etag}")
    if not db_path or not Path(db_path).exists():
        print(f"ERROR: Database file for campaign '{campaign}' not found.")
        return None

    # Iterate through lines in the DB to find a match
    with open(db_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 9:
                continue
            if str(dsid) == fields[0] and etag == fields[8]:
                print(f"Match found in {db_path}")
                return {
                    "cross_section_pb": float(fields[2]),
                    "genFiltEff": float(fields[3]),
                    "kfactor": float(fields[4]),
                    "etag": fields[8],
                }
    print(f"No match found for DSID={dsid}, etag={etag} in {db_path}")
    return None


def write_metadata_to_h5(h5_filename: str, dsid: int, metadata_dict: dict[str, Any]) -> None:
    """Write metadata values into an HDF5 file under metadata/<DSID>.

    Parameters
    ----------
    h5_filename : str
        Target HDF5 file.
    dsid : int
        Dataset ID to write metadata for.
    metadata_dict : dict[str, Any]
        Dictionary of metadata to inject.
    """
    with h5py.File(h5_filename, "a") as f:
        meta_group = f.require_group("metadata")
        dsid_group = meta_group.require_group(str(dsid))

        # Overwrite existing keys if present
        for key in ["cross_section_pb", "genFiltEff", "kfactor"]:
            if key in dsid_group:
                del dsid_group[key]
            dsid_group.create_dataset(key, data=metadata_dict[key])
            print(f"Wrote {key} = {metadata_dict[key]} to metadata/{dsid}/{key}")


def handle_yaml_fallback(h5_path: Path, yaml_data: dict[str, Any]) -> None:
    """Use fallback metadata from YAML if automatic lookup fails.

    Parameters
    ----------
    h5_path : Path
        Path to the HDF5 file.
    yaml_data : dict[str, Any]
        Metadata dictionary loaded from YAML.

    Raises
    ------
    ValueError
        If YAML is invalid, empty, or missing required fields.
    """
    if len(yaml_data) != 1:
        raise ValueError("YAML fallback file must contain exactly one entry")

    key, value = next(iter(yaml_data.items()))

    # YAML may specify container or DSID
    if key == "container name":
        if not isinstance(value, str):
            raise ValueError("'container name' must be a string")
        container = value
        print(f"YAML container fallback: {container}")
        info = extract_info_from_container(container)
        if not info:
            raise ValueError(
                f"Failed to parse valid DSID / etag / campaign from container: {container}"
            )
        dsid, etag, campaign = info
        meta = query_xsecdb(campaign, dsid, etag)
        if not meta:
            raise ValueError(f"Container not found in PMG database: {container}")
    else:
        try:
            dsid = int(key)
        except ValueError as e:
            raise ValueError("YAML key must be a valid DSID or 'container name'") from e
        meta = value
        required_keys = {"cross_section_pb", "genFiltEff", "kfactor"}
        if not isinstance(meta, dict) or not required_keys.issubset(meta):
            raise ValueError(
                "Manual metadata must include: cross_section_pb / genFiltEff / kfactor"
            )

    write_metadata_to_h5(str(h5_path), dsid, meta)
    print(f"YAML metadata written to {h5_path.name} for DSID {dsid}")


def parse_args_and_yaml() -> tuple[list[str], dict[str, Any]]:
    """Parse CLI arguments and load YAML metadata if provided.

    Returns
    -------
    tuple[list[str], dict[str, Any]]
        A tuple of (list of HDF5 file paths, YAML metadata dict).
    """
    parser = argparse.ArgumentParser(
        description="Inject metadata into .h5 files via BigPanDA or YAML fallback."
    )
    parser.add_argument("h5_files", nargs="+", help="List of .h5 files")
    parser.add_argument("-m", "--manual-yaml", help="Manual metadata YAML fallback file")
    args = parser.parse_args()

    yaml_data: dict[str, Any] = {}
    if args.manual_yaml:
        with open(args.manual_yaml) as f:
            yaml_data = yaml.safe_load(f)

    return args.h5_files, yaml_data


def process_single_file(path: Path, yaml_data: dict[str, Any]) -> None:
    """Process a single .h5 file by attempting BigPanDA lookup, then fallback to YAML.

    Parameters
    ----------
    path : Path
        Path to the HDF5 file.
    yaml_data : dict[str, Any]
        Optional fallback metadata.
    """
    if not path.exists():
        print(f"File not found: {path}")
        return

    taskid = extract_taskid_from_filename(path)
    if taskid:
        print(f"Extracted Task ID: {taskid}")
        taskinfo = fetch_taskinfo_from_bigpanda(taskid)
        if taskinfo:
            taskname = taskinfo.get("taskname", "")
            dsid, etag = parse_line_from_taskname(taskname)
            container = extract_mc_container_from_json(taskinfo)
            campaign = parse_campaign_from_taskname(container or "")

            # Only query DB if all three are found
            if dsid and etag and campaign:
                print(
                    f"Matched info: DSID={dsid}, etag={etag}, "
                    f"campaign={campaign}, container={container}"
                )
                meta = query_xsecdb(campaign, dsid, etag)
                if meta:
                    meta["campaign"] = campaign
                    print(
                        f"Got metadata: cross_section = {meta['cross_section_pb']} pb, "
                        f"eff = {meta['genFiltEff']}, k = {meta['kfactor']}"
                    )
                    try:
                        write_metadata_to_h5(str(path), dsid, meta)
                        print(f"Metadata written to {path.name}")
                    except (OSError, ValueError) as e:
                        print(f"Failed to write metadata: {e}")
                    else:
                        return
    else:
        print("Failed to extract Task ID from filename")

    # Fallback to YAML if automatic mode fails
    if yaml_data:
        try:
            print("Using YAML fallback path")
            handle_yaml_fallback(path, yaml_data)
        except (ValueError, KeyError, OSError) as e:
            print(f"YAML fallback failed: {e}")
    else:
        print("Failed to retrieve metadata and no YAML fallback provided")


def main() -> None:
    """Entry point: parse arguments, download xsecDBs, process each file, and clean up."""
    h5_files, yaml_data = parse_args_and_yaml()
    download_xsecdb_files()

    for h5_file in h5_files:
        print(f"\n==================== Processing {h5_file} ====================\n")
        process_single_file(Path(h5_file), yaml_data)

    # Remove temporary database files after use
    for filename in XSECDB_MAP.values():
        path = Path(filename)
        if path.exists():
            try:
                path.unlink()
                print(f"Deleted temporary file: {filename}")
            except OSError as e:
                print(f"Failed to delete {filename}: {e}")


if __name__ == "__main__":
    main()
