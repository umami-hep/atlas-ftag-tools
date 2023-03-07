from dataclasses import dataclass

from ftag.src.cuts import Cuts


@dataclass(frozen=True)
class Flavour:
    name: str
    label: str
    cuts: Cuts
    colour: str = None

    @property
    def prob(self):
        return f"p{self.name}"

    def __str__(self):
        return self.name


class DefaultFlavours:
    bjets = Flavour(
        name="b",
        label="$b$-jets",
        cuts=Cuts.from_list(["HadronConeExclTruthLabelID == 5"]),
        colour="#1f77b4",
    )
    cjets = Flavour(
        name="c",
        label="$c$-jets",
        cuts=Cuts.from_list(["HadronConeExclTruthLabelID == 4"]),
        colour="#ff7f0e",
    )
    ujets = Flavour(
        name="u",
        label="Light-jets",
        cuts=["HadronConeExclTruthLabelID == 0"],
        colour="#2ca02c",
    )
    taujets = Flavour(
        name="tau",
        label="$\\tau$-jets",
        cuts=["HadronConeExclTruthLabelID == 15"],
        colour="#7c5295",
    )

    Hbb = Flavour(
        name="hbb",
        label="Hbb",
        cuts=Cuts.from_list(["R10TruthLabel_R22v1 == 11"]),
        colour="#1f77b4",
    )
    Hcc = Flavour(
        name="hcc",
        label="Hcc",
        cuts=Cuts.from_list(["R10TruthLabel_R22v1 == 12"]),
        colour="#B45F06",
    )
    top = Flavour(
        name="top",
        label="Top",
        cuts=Cuts.from_list(["R10TruthLabel_R22v1 == 1"]),
        colour="#A300A3",
    )
    qcd = Flavour(
        name="qcd",
        label="QCD",
        cuts=Cuts.from_list(["R10TruthLabel_R22v1 == 10"]),
        colour="#38761D",
    )

    def __getitem__(self, key):
        return getattr(self, key)
