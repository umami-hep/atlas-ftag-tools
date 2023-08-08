from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import yaml

from ftag.cuts import Cuts


def remove_suffix(string: str, suffix: str) -> str:
    if string.endswith(suffix):
        return string[: -len(suffix)]
    return string


@dataclass(frozen=True)
class Flavour:
    name: str
    label: str
    cuts: Cuts
    colour: str
    category: str

    @property
    def px(self) -> str:
        return f"p{remove_suffix(self.name, 'jets')}"

    @property
    def eff_str(self) -> str:
        return self.label.replace("jets", "jet") + " efficiency"

    @property
    def rej_str(self) -> str:
        return self.label.replace("jets", "jet") + " rejection"

    def __str__(self) -> str:
        return self.name


@dataclass
class FlavourContainer:
    flavours: dict[str, Flavour]

    def __iter__(self) -> Generator:
        yield from self.flavours.values()

    def __getitem__(self, key) -> Flavour:
        if isinstance(key, Flavour):
            key = key.name
        try:
            return self.flavours[key]
        except KeyError as e:
            raise KeyError(f"Flavour '{key}' not found") from e

    def __getattr__(self, name) -> Flavour:
        return self[name]

    def __contains__(self, flavour: str | Flavour) -> bool:
        if isinstance(flavour, Flavour):
            flavour = flavour.name
        return flavour in self.flavours

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f.name for f in self])})"

    @property
    def categories(self) -> list[str]:
        return list(dict.fromkeys(f.category for f in self))

    def by_category(self, category: str) -> FlavourContainer:
        return FlavourContainer({k: v for k, v in self.flavours.items() if v.category == category})

    def from_cuts(self, cuts: list | Cuts) -> Flavour:
        if isinstance(cuts, list):
            cuts = Cuts.from_list(cuts)
        for flavour in self:
            if flavour.cuts == cuts:
                return flavour
        raise KeyError(f"Flavour with {cuts} not found")


with open(Path(__file__).parent / "flavours.yaml") as f:
    flavours_yaml = yaml.safe_load(f)
flavours_dict = {f["name"]: Flavour(cuts=Cuts.from_list(f.pop("cuts")), **f) for f in flavours_yaml}
assert len(flavours_dict) == len(flavours_yaml), "Duplicate flavour names detected"
Flavours = FlavourContainer(flavours_dict)
