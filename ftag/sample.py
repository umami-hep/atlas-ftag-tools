from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

from ftag.labels import remove_suffix
from ftag.vds import create_virtual_file


@dataclass(frozen=True)
class Sample:
    pattern: Path | str | tuple[Path | str, ...]
    ntuple_dir: Path | str | None = None
    name: str | None = None
    weights: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("Sample pattern cannot be empty")
        if "*" in str(self.pattern) and not self.files:
            raise FileNotFoundError(f"No files matched pattern {self.pattern}")
        if missing := [file for file in self.files if not Path(file).is_file()]:
            raise FileNotFoundError(f"The following files do not exist: {missing}")

    @property
    def path(self) -> tuple[Path, ...]:
        pattern_tuple = self.pattern if isinstance(self.pattern, (list, tuple)) else (self.pattern,)
        if self.ntuple_dir is not None:
            return tuple(Path(self.ntuple_dir, p) for p in pattern_tuple)
        return tuple(Path(p) for p in pattern_tuple)

    @property
    def files(self) -> list[str]:
        files = []
        for p in self.path:
            files += glob.glob(str(p)) if "*" in str(p) else [str(p)]
        return files

    @property
    def num_files(self) -> int:
        return len(self.files)

    @property
    def dsid(self) -> list[str]:
        return list({Path(fname).parent.name for fname in self.files})

    @property
    def sample_id(self) -> list[str]:
        return list({dsid.split(".")[2] for dsid in self.dsid})

    @property
    def tags(self) -> list[str]:
        return list({dsid.split(".")[3] for dsid in self.dsid})

    @property
    def ptag(self) -> list[str]:
        return list({tag for tags in self.tags for tag in tags.split("_") if "p" in tag})

    @property
    def rtag(self) -> list[str]:
        return list({tag for tags in self.tags for tag in tags.split("_") if "r" in tag})

    @property
    def dumper_tag(self) -> list[str]:
        hashes = [remove_suffix(dsid.split(".")[7], "_output") for dsid in self.dsid]
        return list(set(hashes))

    def virtual_file(self, **kwargs) -> list[Path | str]:
        return [create_virtual_file(p, **kwargs) if "*" in str(p) else p for p in self.path]

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name
