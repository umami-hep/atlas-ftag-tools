from dataclasses import dataclass

from ftag.cuts import Cuts


@dataclass(frozen=True)
class Flavour:
    name: str
    label: str
    cuts: Cuts
    colour: str

    @property
    def px(self) -> str:
        return f"p{self.name.rstrip('jets')}"

    def __str__(self) -> str:
        return self.name


@dataclass
class FlavourContainer:
    flavours: dict[str, Flavour]

    def __iter__(self):
        yield from self.flavours.values()

    def __getitem__(self, key) -> Flavour:
        try:
            return self.flavours[key]
        except KeyError as e:
            raise KeyError(f"Flavour '{key}' not found") from e

    def __getattr__(self, name) -> Flavour:
        return self[name]
