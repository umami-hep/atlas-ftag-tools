from __future__ import annotations

from dataclasses import dataclass

from ftag.cuts import Cuts


@dataclass(frozen=True)
class Region:
    name: str
    cuts: Cuts

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.cuts[0].value < other.cuts[0].value

    def __eq__(self, other):
        return self.name == other.name
