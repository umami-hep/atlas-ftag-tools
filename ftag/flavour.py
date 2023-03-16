from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

from ftag.cuts import Cuts


@dataclass(frozen=True)
class Flavour:
    name: str
    label: str
    cuts: Cuts
    colour: str
    category: str

    @property
    def px(self) -> str:
        return f"p{self.name.removesuffix('jets')}"

    def __str__(self) -> str:
        return self.name


@dataclass
class FlavourContainer:
    flavours: dict[str, Flavour]

    def __iter__(self) -> Generator:
        yield from self.flavours.values()

    def __getitem__(self, key) -> Flavour:
        try:
            return self.flavours[key]
        except KeyError as e:
            raise KeyError(f"Flavour '{key}' not found") from e

    def __getattr__(self, name) -> Flavour:
        return self[name]

    def __repr__(self) -> str:
        return f"FlavourContainer({', '.join(list(f.name for f in self))})"

    @property
    def categories(self) -> list[str]:
        return list(dict.fromkeys(f.category for f in self))

    def by_category(self, category: str) -> FlavourContainer:
        return FlavourContainer({k: v for k, v in self.flavours.items() if v.category == category})
