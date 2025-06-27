from __future__ import annotations

import functools
import operator
from ast import literal_eval
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import starmap
from typing import NamedTuple

import numpy as np

OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "in": np.isin,
    "notin": lambda x, y: ~np.isin(x, y),
}

for i in range(2, 101):
    OPERATORS[f"%{i}=="] = functools.partial(lambda x, y, i: (x % i) == y, i=i)
    OPERATORS[f"%{i}!="] = functools.partial(lambda x, y, i: (x % i) != y, i=i)
    OPERATORS[f"%{i}<="] = functools.partial(lambda x, y, i: (x % i) <= y, i=i)
    OPERATORS[f"%{i}>="] = functools.partial(lambda x, y, i: (x % i) >= y, i=i)


class CutsResult(NamedTuple):
    idx: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class Cut:
    variable: str
    operator: str
    _value: str | int | float

    @property
    def value(self) -> int | float:
        if isinstance(self._value, str):
            txt = self._value.strip().lower()
            if txt == "nan":
                return float("nan")
            if txt in {"+inf", "inf"}:
                return float("inf")
            if txt == "-inf":
                return float("-inf")
            return literal_eval(self._value)
        return self._value

    def __call__(self, array):
        col = array[self.variable]
        val = self.value

        if isinstance(val, float) and np.isnan(val):
            if self.operator == "!=":
                return ~np.isnan(col)
            if self.operator == "==":
                return np.isnan(col)
            raise ValueError("'nan' only makes sense with '==' or '!=' operators")

        return OPERATORS[self.operator](col, val)

    def __str__(self) -> str:
        return f"{self.variable} {self.operator} {self.value}"


@dataclass(frozen=True)
class Cuts:
    cuts: tuple[Cut, ...]

    @classmethod
    def from_list(cls, cuts: list) -> Cuts:
        if cuts and isinstance(cuts[0], str):
            cuts = [cut.split(" ") for cut in cuts]
        if cuts and isinstance(cuts[0], list):
            cuts = list(map(tuple, cuts))
        return cls(tuple(starmap(Cut, dict.fromkeys(cuts))))

    @classmethod
    def empty(cls) -> Cuts:
        return cls(())

    def __post_init__(self):
        assert isinstance(self.cuts, tuple)
        assert all(isinstance(c, Cut) for c in self.cuts)

    @property
    def variables(self) -> list[str]:
        return list(dict.fromkeys(c.variable for c in self))

    def ignore(self, variables: list[str]):
        return Cuts(tuple(c for c in self if c.variable not in variables))

    def __call__(self, array: np.ndarray) -> CutsResult:
        if array.ndim == 2:
            raise ValueError("This interface only supports jet selections")
        keep = np.arange(len(array))
        for cut in self.cuts:
            idx = cut(array)
            array, keep = array[idx], keep[idx]
        return CutsResult(keep, array)

    def __add__(self, other: Cuts):
        return Cuts(tuple(dict.fromkeys(self.cuts + other.cuts)))

    def __len__(self) -> int:
        return len(self.cuts)

    def __iter__(self) -> Iterator:
        yield from self.cuts

    def __getitem__(self, variable):
        return Cuts(tuple(cut for cut in self.cuts if cut.variable == variable))

    def __repr__(self) -> str:
        return str([f"{c}" for c in self.cuts])
