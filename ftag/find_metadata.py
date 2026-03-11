from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import requests

# Mapping of campaign names to their respective PMG database files
XSECDB_MAP = {
    "mc15": "PMGxsecDB_mc15.txt",
    "mc16": "PMGxsecDB_mc16.txt",
    "mc21": "PMGxsecDB_mc21.txt",
    "mc23": "PMGxsecDB_mc23.txt",
}
# Base URL for the PMG Tools group data at CERN
XSECDB_URL_BASE = "https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/PMGTools/"


class MetadataFinder:
    """Fetch and inject metadata into .h5 files.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file where metadata should be injected.
    """

    def __init__(self, h5_path: str):
        self.h5_path = Path(h5_path)

    def _extract_taskid(self) -> str | None:
        """Extract the 8-digit BigPanDA Task ID from the filename.

        Returns
        -------
        str | None
            The extracted Task ID, or ``None`` if no match is found.
        """
        m = re.search(r"\.(\d{8})\.", self.h5_path.name)
        return m.group(1) if m else None

    def _fetch_taskinfo(self, taskid: str) -> dict | None:
        """Fetch task details from the BigPanDA JSON API.

        Parameters
        ----------
        taskid : str
            The BigPanDA Task ID.

        Returns
        -------
        dict | None
            The first task record from the API response, or ``None`` if unavailable.
        """
        url = f"https://bigpanda.cern.ch/tasks/?jeditaskid={taskid}&json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data[0] if isinstance(data, list) and data else None

    def _extract_container(self, info: dict) -> str | None:
        """Find the MC container name within the task information.

        Parameters
        ----------
        info : dict
            The task information dictionary from BigPanDA.

        Returns
        -------
        str | None
            The container name (e.g., 'mc16_13TeV...'), or ``None`` if not found.
        """
        text = json.dumps(info)
        m = re.search(r"(mc\d+_13TeV\.[\w\.]+)", text)
        return m.group(1) if m else None

    def _parse_info(self, container: str) -> tuple[int, str, str] | None:
        """Parse DSID, etag, and campaign from a container name.

        Parameters
        ----------
        container : str
            The full MC container name.

        Returns
        -------
        tuple[int, str, str] | None
            A tuple of (DSID, etag, campaign), or ``None`` if parsing fails.
            Note: 'mc20' is automatically mapped to 'mc16'.
        """
        mc = re.search(r"\b(mc\d+)_13TeV", container)
        dsid = re.search(r"\.(\d{6})\.", container)
        etag = re.search(r"\.e(\d+)(?:[_.]|$)", container)
        if mc and dsid and etag:
            campaign = "mc16" if mc.group(1) == "mc20" else mc.group(1)
            return int(dsid.group(1)), f"e{etag.group(1)}", campaign
        return None

    def _download_db(self, campaign: str) -> str:
        """Download the PMG database for a specific campaign.

        Parameters
        ----------
        campaign : str
            The MC campaign name (e.g., 'mc16').

        Returns
        -------
        str
            The local path to the downloaded database file.
        """
        fn = XSECDB_MAP[campaign]
        url = XSECDB_URL_BASE + fn
        if not Path(fn).exists():
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            Path(fn).write_bytes(r.content)
        return fn

    def _query_xsecdb(self, campaign: str, dsid: int, etag: str) -> dict | None:
        """Search the PMG database for metadata matching a DSID and etag.

        Parameters
        ----------
        campaign : str
            The MC campaign name.
        dsid : int
            Dataset ID.
        etag : str
            The AMI tag (e.g., 'e1234').

        Returns
        -------
        dict | None
            Dictionary containing 'cross_section_pb', 'genFiltEff', and 'kfactor',
            or ``None`` if no matching record is found.
        """
        db_path = self._download_db(campaign)
        with open(db_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 9 and parts[0] == str(dsid) and parts[8] == etag:
                    return {
                        "cross_section_pb": float(parts[2]),
                        "genFiltEff": float(parts[3]),
                        "kfactor": float(parts[4]),
                    }
        return None

    def inject_metadata(self) -> None:
        """Execute the full workflow to inject metadata into the HDF5 file.

        This method coordinates Task ID extraction, API fetching, DB querying,
        and final HDF5 attribute writing. Metadata is stored in a group
        named ``metadata/{dsid}``.
        """
        taskid = self._extract_taskid()
        if not taskid:
            print(f"No Task ID found in {self.h5_path.name}")
            return
        info = self._fetch_taskinfo(taskid)
        if not info:
            print("No BigPanDA info found.")
            return
        container = self._extract_container(info)

        if not container:
            print("Failed to extract container name from BigPanDA info.")
            return

        parsed = self._parse_info(container)
        if not parsed:
            print("Failed to parse DSID/etag/campaign.")
            return
        dsid, etag, campaign = parsed
        meta = self._query_xsecdb(campaign, dsid, etag)
        if not meta:
            print("No metadata found in PMG DB.")
            return
        with h5py.File(self.h5_path, "a") as f:
            g = f.require_group(f"metadata/{dsid}")
            for k, v in meta.items():
                if k in g:
                    del g[k]
                g.create_dataset(k, data=v)
        print(f"Metadata injected for {self.h5_path.name}")
