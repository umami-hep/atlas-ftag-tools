from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import h5py
import requests
import yaml

"""
find_metadata.py

Injects metadata (cross_section_pb, genFiltEff, kfactor) into .h5 files produced by TDD.

Usage:
    python find_metadata.py <h5_file1> [<h5_file2> ...] [-m fallback.yaml]

Modes:
    1. Auto mode (default):
        - Extract ATLAS BigPanDA Task ID from filename
        - Query BigPanDA → extract task info → parse DSID/etag/campaign
        - Query PMGxsecDB and write metadata into /metadata/<DSID>/ in the .h5 file

    2. Fallback mode (-m):
        - If auto lookup fails, use a YAML file to inject metadata manually
        - YAML must contain exactly one entry (see example_metadata.yaml)

Example:
    python find_metadata.py user.username.taskid._000001.h5
    python find_metadata.py localdump.h5 -m fallback.yaml
"""

XSECDB_MAP = {
    "mc15": "PMGxsecDB_mc15.txt",
    "mc16": "PMGxsecDB_mc16.txt",
    "mc21": "PMGxsecDB_mc21.txt",
    "mc23": "PMGxsecDB_mc23.txt",
}

XSECDB_URL_BASE = "https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/PMGTools/"


def validate_url_scheme(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    return parsed


def download_xsecdb_files():
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
    m = re.search(r"\.(\d{8})\.", h5_path.name)
    return m.group(1) if m else None


def fetch_taskinfo_from_bigpanda(taskid: str) -> dict | None:
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


def extract_mc_container_from_json(data: dict) -> str | None:
    text = json.dumps(data)
    match = re.search(r"(mc\d+_13TeV\.[\w\.]+)", text)
    return match.group(1) if match else None


def parse_line_from_taskname(taskname: str):
    match = re.match(r".*\.(\d{6})\.e(\d+)_", taskname)
    if match:
        dsid, etag = match.groups()
        return int(dsid), f"e{etag}"
    return None, None


def parse_campaign_from_taskname(taskname: str) -> str | None:
    m = re.match(r"(mc\d+)_13TeV", taskname)
    if m:
        raw_campaign = m.group(1)
        return "mc16" if raw_campaign == "mc20" else raw_campaign
    return None


def extract_info_from_container(container: str) -> tuple[int, str, str] | None:
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


def query_xsecdb(campaign, dsid, etag):
    db_path = XSECDB_MAP.get(campaign)
    print(f"Searching in {db_path} for DSID={dsid}, etag={etag}")
    if not db_path or not Path(db_path).exists():
        print(f"ERROR: Database file for campaign '{campaign}' not found.")
        return None
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


def write_metadata_to_h5(h5_filename, dsid, metadata_dict):
    with h5py.File(h5_filename, "a") as f:
        meta_group = f.require_group("metadata")
        dsid_group = meta_group.require_group(str(dsid))
        for key in ["cross_section_pb", "genFiltEff", "kfactor"]:
            if key in dsid_group:
                del dsid_group[key]
            dsid_group.create_dataset(key, data=metadata_dict[key])
            print(f"Wrote {key} = {metadata_dict[key]} to metadata/{dsid}/{key}")


def handle_yaml_fallback(h5_path: Path, yaml_data: dict):
    if len(yaml_data) != 1:
        raise ValueError("YAML fallback file must contain exactly one entry")

    key, value = next(iter(yaml_data.items()))
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


def parse_args_and_yaml():
    parser = argparse.ArgumentParser(
        description="Inject metadata into .h5 files via BigPanDA or YAML fallback."
    )
    parser.add_argument("h5_files", nargs="+", help="List of .h5 files")
    parser.add_argument("-m", "--manual-yaml", help="Manual metadata YAML fallback file")
    args = parser.parse_args()

    yaml_data = {}
    if args.manual_yaml:
        with open(args.manual_yaml) as f:
            yaml_data = yaml.safe_load(f)

    return args.h5_files, yaml_data


def process_single_file(path: Path, yaml_data: dict):
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
                        f"eff = {meta['genFiltEff']}, "
                        f"k = {meta['kfactor']}"
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

    if yaml_data:
        try:
            print("Using YAML fallback path")
            handle_yaml_fallback(path, yaml_data)
        except (ValueError, KeyError, OSError) as e:
            print(f"YAML fallback failed: {e}")
    else:
        print("Failed to retrieve metadata and no YAML fallback provided")


def main():
    h5_files, yaml_data = parse_args_and_yaml()
    download_xsecdb_files()

    for h5_file in h5_files:
        print(f"\n==================== Processing {h5_file} ====================\n")
        process_single_file(Path(h5_file), yaml_data)

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
