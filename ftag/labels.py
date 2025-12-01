from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

from ftag.cuts import Cuts


def remove_suffix(string: str, suffix: str) -> str:
    """Remove the suffix from a string.

    Parameters
    ----------
    string : str
        String from which the suffix is to be removed
    suffix : str
        Suffix to remove

    Returns
    -------
    str
        String with the suffix removed
    """
    if string.endswith(suffix):
        return string[: -len(suffix)]
    return string


@dataclass(frozen=True)
class Label:
    """Dataclass to hold info about one flavour/label.

    Attributes
    ----------
    name : str
        Name of the flavour/label
    label : str
        Plot label for this flavour/label
    cuts : Cuts
        Cuts of this flavour/label
    colour : str
        Colour of this flavour/label
    category : str
        Flavour category, to which this flavour/label belongs
    """

    name: str
    label: str
    cuts: Cuts
    colour: str
    category: str
    _px: str | None = None

    @property
    def px(self) -> str:
        return self._px or f"p{remove_suffix(self.name, 'jets')}"

    @property
    def eff_str(self) -> str:
        return self.label.replace("jets", "jet") + " efficiency"

    @property
    def rej_str(self) -> str:
        return self.label.replace("jets", "jet") + " rejection"

    @property
    def frac_str(self) -> str:
        return "f" + remove_suffix(self.name, "jets")

    def __str__(self) -> str:
        return self.name

    def __lt__(self, other) -> bool:
        return self.name < other.name


@dataclass
class LabelContainer:
    """Label container that holds multiple labels.

    Attributes
    ----------
    labels : dict[str, Label]
        Dict with the labels this container will hold
    """

    labels: dict[str, Label]

    def __iter__(self) -> Iterator:
        yield from self.labels.values()

    def __getitem__(self, key) -> Label:
        if isinstance(key, Label):
            key = key.name
        try:
            return self.labels[key]
        except KeyError as e:
            raise KeyError(f"Label '{key}' not found") from e

    def __len__(self) -> int:
        return len(self.labels.keys())

    def __getattr__(self, name) -> Label:
        return self[name]

    def __contains__(self, label: str | Label) -> bool:
        if isinstance(label, Label):
            label = label.name
        return label in self.labels

    def __eq__(self, other) -> bool:
        if isinstance(other, LabelContainer):
            return self.labels == other.labels
        if isinstance(other, list) and all(isinstance(f, str) for f in other):
            return {f.name for f in self} == set(other)
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f.name for f in self])})"

    @property
    def categories(self) -> list[str]:
        return list(dict.fromkeys(f.category for f in self))

    def by_category(self, category: str) -> LabelContainer:
        f = LabelContainer({k: v for k, v in self.labels.items() if v.category == category})
        if not f.labels:
            raise KeyError(f"No labels with category '{category}' found")
        return f

    def from_cuts(self, cuts: list | Cuts) -> Label:
        if isinstance(cuts, list):
            cuts = Cuts.from_list(cuts)
        for label in self:
            if label.cuts == cuts:
                return label
        raise KeyError(f"Label with {cuts} not found")

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Path | None = None,
        include_categories: Iterable[str] | None = None,
        exclude_categories: Iterable[str] | None = None,
    ) -> LabelContainer:
        if yaml_path is None:
            yaml_path = Path(__file__).parent / "flavours.yaml"
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Filter for categories to include
        if include_categories is not None:
            include_categories = set(include_categories)
            config = [f for f in config if f.get("category") in include_categories]

        # Filter for categories to exclude
        if exclude_categories is not None:
            exclude_categories = set(exclude_categories)
            config = [f for f in config if f.get("category") not in exclude_categories]

        if not config:
            raise KeyError("No labels left after category filtering.")

        # sanity checks
        cuts = [Cuts.from_list(f["cuts"]) for f in config]
        if duplicates := [c for c in cuts if cuts.count(c) > 1]:
            raise ValueError(f"Duplicate label definitions detected: {duplicates}")
        names = [f["name"] for f in config]
        if duplicates := [n for n in names if names.count(n) > 1]:
            raise ValueError(f"Duplicate label names detected: {duplicates}")

        labels = {f["name"]: Label(cuts=Cuts.from_list(f.pop("cuts")), **f) for f in config}
        return cls(labels)

    @classmethod
    def from_list(cls, labels: list[Label]) -> LabelContainer:
        return cls({f.name: f for f in labels})

    def backgrounds(self, signal: Label, only_signals: bool = True) -> LabelContainer:
        bkg = [f for f in self if f.category == signal.category and f != signal]
        if not only_signals:
            bkg = [f for f in bkg if f.name not in {"ujets", "qcd"}]
        if len(bkg) == 0:
            raise TypeError(
                "No background flavour could be found in the flavours for signal "
                f"flavour {signal.name}"
            )
        return LabelContainer.from_list(bkg)

    def cut_variables(self) -> list[str]:
        """Return all variable names appearing in any Label cuts.

        Returns
        -------
        list[str]
            Unique variable names used across all cuts in all labels.
        """
        vars_found: set[str] = set()
        var_regex = re.compile(r"[A-Za-z_]\w*")

        for label in self:
            for cut in label.cuts:
                tokens = var_regex.findall(str(cut))
                # filter out boolean keywords
                vars_found.update(tokens)

        return list(vars_found)
