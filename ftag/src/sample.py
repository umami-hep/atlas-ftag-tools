from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    name: str
    ntuple_dir: Path
    pattern: str | list[str]

    @property
    def path(self) -> Path | list[Path]:
        if isinstance(self.pattern, str):
            return self.ntuple_dir / self.pattern
        return [self.ntuple_dir / p for p in self.pattern]

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name
