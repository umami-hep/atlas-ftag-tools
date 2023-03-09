from dataclasses import dataclass

from ftag.src.cuts import Cuts


@dataclass(frozen=True)
class Flavour:
    name: str
    label: str
    cuts: Cuts
    colour: str

    @property
    def px(self) -> str:
        return f"p{self.name}"

    def __str__(self) -> str:
        return self.name


@dataclass
class FlavourContainer:
    flavours: dict[str, Flavour]

    def __iter__(self):
        yield from self.flavours.values()

    def __getitem__(self, key) -> Flavour:
        return self.flavours[key]

    def __getattr__(self, name) -> Flavour:
        return self[name]
