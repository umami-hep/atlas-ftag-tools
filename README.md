# FTAG TOOLS

This is a collection of Python tools for working with FTAG.

### Cuts

The `Cuts` class provides an interface for applying selections to arrays loaded from structured hdf5 files. For example

```python
import h5py
from ftag import Cuts

with h5py.File("file.h5", "r") as f:
    array = f["jet"]

# define some cuts
kinematic_cuts = Cuts.from_list(["pt > 20e3", "abs_eta < 2.5"])
flavour_cuts = Cuts.from_list(["HadronConeExclTruthLabelID == 5"])

# we can combine cuts
combined_cuts = kinematic_cuts + flavour_cuts

# apply all cuts
idx, selected = combined_cuts(array)

# selected indices are returned so that the cuts can be applied elsewhere
selected = another_array[idx]

# apply cuts for a specific varible
idx, selected = kinematic_cuts["pt"](array)

# apply all cuts
bjets = cuts.apply(array)
```

### Flavours

A list of flavours is provided. For example

```python
from ftag import Flavours

print(Flavours.bjets)

# apply HadronConeExclTruthLabelID == 5 selection
bjets = Flavours.bjets.cuts(array).values
```
