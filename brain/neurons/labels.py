"""
Access to FlyWire community cell labels.

PROVENANCE: category A -- real data.

`labels.csv.gz` from the FlyWire v783 release contains free-text annotations
contributed by named proofreaders and their labs, with attribution (user_name,
user_affiliation, date_created). These carry functional identifications that the
structured `consolidated_cell_types` table does not, for example
"Sugar Gustatory Receptor Neuron" or "Proboscis motor neuron".

They are human annotations, so they are neither exhaustive nor guaranteed
consistent. Every lookup here reports how many neurons it matched and who
labelled them, so a caller can always see what the identification rests on.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import config


class LabelIndex:
    """Searchable index over real FlyWire community labels."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def find(self, pattern: str, case: bool = False) -> pd.DataFrame:
        """
        All labelled neurons whose label matches `pattern` (regex).

        Returns one row per (root_id, label) with attribution.
        """
        m = self.df["label"].str.contains(pattern, case=case, regex=True, na=False)
        return self.df.loc[m]

    def root_ids(self, pattern: str, case: bool = False) -> np.ndarray:
        return np.unique(self.find(pattern, case)["root_id"].to_numpy())

    def describe(self, pattern: str, case: bool = False) -> dict:
        """Match summary, including who contributed the labels."""
        hit = self.find(pattern, case)
        return {
            "pattern": pattern,
            "n_neurons": int(hit["root_id"].nunique()),
            "n_label_rows": int(len(hit)),
            "distinct_labels": sorted(set(hit["label"].str.strip()))[:20],
            "contributors": sorted(set(
                "%s (%s)" % (a, b) for a, b in
                zip(hit["user_name"].astype(str), hit["user_affiliation"].astype(str))
            ))[:10],
            "source": "FlyWire v783 labels.csv.gz",
        }


@lru_cache(maxsize=1)
def load_labels() -> LabelIndex:
    df = pd.read_csv(
        config.SRC["labels"],
        usecols=["root_id", "label", "user_name", "user_affiliation", "date_created"],
        dtype={"root_id": np.int64, "label": "object",
               "user_name": "object", "user_affiliation": "object"},
    )
    df["label"] = df["label"].astype(str).str.strip()
    return LabelIndex(df)


# --- Curated functional groups, each defined by a real-label regex ----------
# The patterns are ours; the labels they match are real FlyWire annotations.
FUNCTIONAL_GROUPS = {
    # Sugar GRNs, labelled by Philip Shiu (Kristin Scott Lab), citing
    # Engert et al. 2022 (doi:10.7554/eLife.78110) and Shiu, Sterne et al. 2022.
    "sugar_grn":       r"(?i)sugar[^|]*gustatory receptor neuron",
    # Primary bitter GRNs only; "second-order bitter neuron" is excluded.
    "bitter_grn":      r"(?i)(?<!second-order )bitter[^|]*gustatory receptor neuron",
    # Proboscis motor neurons, labelled by Claire McKellar.
    "proboscis_motor": r"(?i)proboscis (?:motor|premotor) neuron",
}


def functional_group(name: str) -> np.ndarray:
    """Real FlyWire root IDs for a curated functional group."""
    if name not in FUNCTIONAL_GROUPS:
        raise KeyError("unknown functional group %r; known: %s"
                       % (name, sorted(FUNCTIONAL_GROUPS)))
    return load_labels().root_ids(FUNCTIONAL_GROUPS[name])
