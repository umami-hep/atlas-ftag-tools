import operator
from dataclasses import dataclass

OPERATORS = {
    "==": operator.__eq__,
    ">=": operator.__ge__,
    "<=": operator.__le__,
    ">": operator.__gt__,
    "<": operator.__lt__,
}

for i in range(2, 20):
    OPERATORS[f"%{i}=="] = lambda x, y: (x % i) == y
    OPERATORS[f"%{i}<="] = lambda x, y: (x % i) == y
    OPERATORS[f"%{i}>="] = lambda x, y: (x % i) == y


@dataclass(frozen=True)
class Cut:
    variable: str
    operator: str
    value: str

    def __call__(self, array):
        return OPERATORS[self.operator](array[self.variable], self.value)

    def __str__(self):
        return f"{self.variable} {self.operator} {self.value}"


@dataclass(frozen=True)
class Cuts:
    cuts: tuple[Cut]

    @classmethod
    def from_list(cls, cuts: list):
        if cuts and isinstance(cuts[0], str):
            cuts = list(map(lambda cut: cut.split(" "), cuts))
        if cuts and isinstance(cuts[0], list):
            cuts = map(tuple, cuts)
        return cls(tuple(Cut(*cut) for cut in dict.fromkeys(cuts)))

    @classmethod
    def empty(cls):
        return cls(tuple())

    def __post_init__(self):
        assert isinstance(self.cuts, tuple)
        assert all(isinstance(c, Cut) for c in self.cuts)

    @property
    def variables(self):
        return list(dict.fromkeys(c.variable for c in self))

    def ignore(self, variables):
        return Cuts(tuple(c for c in self if c.variable not in variables))

    def __call__(self, array):
        keep = list(range(len(array)))
        for cut in self.cuts:
            idx = cut(array)
            array, keep = array[idx], keep[idx]
        return idx, array

    def __add__(self, other):
        return Cuts(tuple(dict.fromkeys(self.cuts + other.cuts)))

    def __len__(self):
        return len(self.cuts)

    def __iter__(self):
        yield from self.cuts

    def __getitem__(self, variable):
        return Cuts(tuple(cut for cut in self.cuts if cut.variable == variable))

    def __str__(self):
        return str([f"{c}" for c in self.cuts])
