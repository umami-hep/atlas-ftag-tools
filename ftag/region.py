from __future__ import annotations

from dataclasses import dataclass

from ftag.cuts import Cuts


@dataclass(frozen=True)
class Region:
    name: str
    cuts: Cuts

    def __str__(self) -> str:
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name
