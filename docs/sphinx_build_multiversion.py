"""Script to loop over a number of git branches/tags, check them out and build sphinx docs.

The branches/tags for which the docs are generated are defined in the file
"docs/source/_static/switcher.json" (note that the docs do not use the local version
by default, but instead the version from the latest commit on GH-Pages. This is due to
the fact that this button prefers urls instead of local filenames).

This script has to be executed in the root of the repository!
"""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copy
from subprocess import run


def build_docs_version(version: str) -> None:
    """Build the docs for a specific version.

    The latest conf.py is used no matter if it differs from the version from back then.
    This function expects the file "conf_latest.py" to exist in the current working
    directory.

    Parameters
    ----------
    version : str
        Branch or tag name for which the docs are built
    """
    # checkout the version/tag and obtain latest conf.py
    run(f"git checkout {version}", shell=True, check=True)
    if Path("docs/source/conf.py").is_file():
        # removing the old conf.py file to make room for the latest one
        Path("docs/source/conf.py").unlink()
    copy("conf_latest.py", "docs/source/conf.py")

    # run librep on markdown files (render placeholders with sytax §§§filename§§§)
    run(
        "librep --ref_dir $PWD --input 'docs/**/*.md' --no_backup",
        shell=True,
        check=True,
    )

    # build the docs for this version
    run(
        f"sphinx-build -b html docs/source docs/_build/html/{version}",
        shell=True,
        check=True,
    )
    run("git stash", shell=True, check=True)


def main() -> None:
    """Run sphinx docs creation."""
    with open("docs/source/_static/switcher.json") as f:  # pylint: disable=W1514
        versions = json.load(f)

    # remember which branch we started on
    current_branch = (
        run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, check=True)
        .stdout.decode()
        .strip()
    )

    # copy the latest conf.py, since we want to use that for *all* versions
    copy("docs/source/conf.py", "./conf_latest.py")

    # always build the main branch first
    build_docs_version("main")

    # then every tag/branch listed in switcher.json that is *not* “main”
    for entry in versions:
        ver = entry["version"]
        if ver != "main":
            build_docs_version(ver)

    # go back to where we started and clean up
    run(f"git checkout {current_branch}", shell=True, check=True)
    Path("./conf_latest.py").unlink()


if __name__ == "__main__":
    main()
