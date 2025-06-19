from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import h5py

# Mapping of campaign to local database file name
XSECDB_MAP = {
    "mc15": "PMGxsecDB_mc15.txt",
    "mc16": "PMGxsecDB_mc16.txt",
    "mc21": "PMGxsecDB_mc21.txt",
    "mc23": "PMGxsecDB_mc23.txt",
}

XSECDB_URL_BASE = "https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/PMGTools/"


def download_xsecdb_files():
    for filename in XSECDB_MAP.values():
        if not Path(filename).exists():
            url = XSECDB_URL_BASE + filename
            try:
                print(f"Downloading {filename} from {url}")
                urllib.request.urlretrieve(url, filename)  # noqa: S310
                print(f"Downloaded: {filename}")
            except urllib.error.URLError as e:
                print(f"ERROR: Failed to download {filename} - {e}")


def parse_line(container_name):
    match = re.match(r"(mc\d+)_13TeV\.(\d{6})\..*\.e(\d+)_", container_name)
    if match:
        raw_campaign, dsid, etag = match.groups()
        campaign = "mc16" if raw_campaign == "mc20" else raw_campaign
        print(
            f"Parsed container '{container_name}' -> "
            f"Campaign: {raw_campaign}, PMG Campaign: {campaign}, "
            f"DSID: {dsid}, etag: e{etag}"
        )
        return raw_campaign, campaign, int(dsid), f"e{etag}"
    print(f"ERROR: Failed to parse container '{container_name}'")
    return None, None, None, None


def query_xsecdb(campaign, dsid, etag):
    db_path = XSECDB_MAP.get(campaign)
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
                print(f"Found metadata for DSID {dsid} and etag {etag} in campaign '{campaign}'")
                return {
                    "cross_section_pb": float(fields[2]),
                    "genFiltEff": float(fields[3]),
                    "kfactor": float(fields[4]),
                    "etag": fields[8],
                }
    print(f"ERROR: Metadata for DSID {dsid} and etag {etag} not found in campaign '{campaign}'")
    return None


def write_metadata_to_h5(h5_filename, metadata_dict):
    with h5py.File(h5_filename, "a") as f:
        meta_group = f.require_group("metadata")
        for dsid_str, meta in metadata_dict.items():
            dsid_group = meta_group.require_group(dsid_str)
            for key in ["cross_section_pb", "genFiltEff", "kfactor"]:
                if key in dsid_group:
                    del dsid_group[key]
                dsid_group.create_dataset(key, data=meta[key])


def process_container_list(input_file, output_file, h5_merge_file=None):
    result = {}
    error_log = output_file.replace(".json", "_errors.log")
    error_lines = []

    print(f"Processing container list from file '{input_file}'")
    with open(input_file) as f:
        for line in f:
            container = line.strip()
            if not container:
                continue
            raw_campaign, campaign, dsid, etag = parse_line(container)
            if not campaign or not dsid or not etag:
                msg = f"[Skipped] Unable to parse container: {container}"
                print(msg)
                error_lines.append(msg)
                continue
            metadata = query_xsecdb(campaign, dsid, etag)
            if metadata:
                metadata["campaign"] = (
                    f"{raw_campaign} (fallback to {campaign})"
                    if raw_campaign != campaign
                    else campaign
                )
                result[str(dsid)] = metadata
            else:
                msg = (
                    f"ERROR: DSID {dsid} ({raw_campaign}) lookup failed, " f"container: {container}"
                )
                print(msg)
                error_lines.append(msg)

    print(f"Writing successful results to file '{output_file}'")
    with open(output_file, "w") as out_f:
        json.dump(result, out_f, indent=2)

    if error_lines:
        print(f"Writing error logs to file '{error_log}'")
        with open(error_log, "w") as err_f:
            for err in error_lines:
                err_f.write(err + "\n")

    if h5_merge_file:
        print(f"Merging metadata into HDF5 file: {h5_merge_file}")
        try:
            write_metadata_to_h5(h5_merge_file, result)
            print(f"Successfully merged metadata into {h5_merge_file}")
        except (OSError, RuntimeError) as e:
            print(f"ERROR during HDF5 merge: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Lookup cross-section metadata for containers " "and optionally merge into HDF5"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input txt file containing container names"
    )
    parser.add_argument("-o", "--output", default="metadata.json", help="Output JSON file name")
    parser.add_argument(
        "-m",
        "--merge",
        metavar="H5FILE",
        help="If provided, merge the result into the specified HDF5 file",
    )
    args = parser.parse_args()

    download_xsecdb_files()
    process_container_list(args.input, args.output, args.merge)


if __name__ == "__main__":  # pragma: no cover
    main()
