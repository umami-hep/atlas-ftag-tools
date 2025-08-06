from __future__ import annotations

from ftag.labels import LabelContainer

Flavours = LabelContainer.from_yaml(
    exclude_categories=[
        "single-btag-extended",
        "single-btag-extended-ghost",
    ]
)
Extended_Flavours = LabelContainer.from_yaml(
    exclude_categories=[
        "single-btag",
        "single-btag-ghost",
    ]
)
