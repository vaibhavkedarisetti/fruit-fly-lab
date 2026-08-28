"""
Retinotopic map of the FlyWire optic lobe, derived from real column assignments.

PROVENANCE
----------
A. REAL DATA
   - `column_assignment.csv.gz` (FlyWire v783): assigns 45,528 columnar optic-lobe
     neurons of 31 types to 785 (left) / 796 (right) medulla columns, each with
     hex-lattice coordinates (p, q). One column per ommatidium.
   - 3D neuron coordinates (`coordinates.csv.gz`) used to *orient* the lattice.
   - Real synaptic connectivity used to assign receptive-field centres to
     wide-field visual projection neurons (LC4, LPLC2, ...) that have no column
     assignment of their own.

   The lattice orientation is measured, not assumed. Regressing the hex axes on
   real FAFB coordinates over the 1,581 Mi1 neurons (one per column) gives:
       u = p + q/2      -> dorsoventral axis, R^2 = 0.98   (elevation)
       v = (sqrt3/2)*q  -> anteroposterior axis, partial r = +0.95 (azimuth)
   `tests/test_retinotopy.py` re-derives and asserts these.

C. OUR APPROXIMATION
   - The hex lattice is mapped linearly onto the published field of view of one
     Drosophila eye. The real interommatidial angle varies across the eye
     (~4.5 deg frontally to ~8 deg laterally); we use a uniform mapping.
     Sources for the FOV: Heisenberg & Wolf 1984; Borst 2009, Curr Biol 19:R995.
   - A visual projection neuron's receptive-field centre is taken as the
     synapse-count-weighted mean visual direction of its column-assigned
     presynaptic partners. Its RF radius is the weighted standard deviation.
     This is a reasonable proxy but is NOT a measured receptive field.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import config

# --- C: published field of view of one Drosophila eye (degrees) -------------
FOV_AZIMUTH_DEG = 175.0     # anterior-posterior extent of one eye
FOV_ELEVATION_DEG = 160.0   # dorsoventral extent of one eye
# Eyes point laterally; centre of each eye's FOV, in head-centred azimuth.
EYE_CENTRE_AZIMUTH_DEG = {"left": -55.0, "right": +55.0}

# Types used to orient the lattice (exactly one per column).
ORIENTING_TYPE = "Mi1"


class Retinotopy:
    """Maps FlyWire optic-lobe columns and visual projection neurons to visual space."""

    def __init__(self, connectome):
        self.c = connectome
        self.columns = self._build_columns()
        self._cache: dict = {}

    # ------------------------------------------------------------------ build
    def _build_columns(self) -> pd.DataFrame:
        ca = pd.read_csv(
            config.SRC["column_assignment"],
            dtype={"root_id": np.int64, "column_id": np.int32,
                   "p": np.int32, "q": np.int32},
        )
        ca = ca[["root_id", "hemisphere", "type", "column_id", "p", "q"]].copy()

        # Hex axial -> orthogonal lattice cartesian (verified against anatomy).
        ca["u"] = ca["p"] + ca["q"] / 2.0                 # elevation axis
        ca["v"] = (np.sqrt(3.0) / 2.0) * ca["q"]          # azimuth axis

        out = []
        for hemi, s in ca.groupby("hemisphere"):
            s = s.copy()
            # Linear map of the occupied lattice onto the published FOV.
            u0, u1 = s["u"].min(), s["u"].max()
            v0, v1 = s["v"].min(), s["v"].max()
            # u increases dorsally (regression coefficient on FAFB y is negative,
            # and FAFB y increases ventrally), so elevation increases with u.
            s["elevation_deg"] = (s["u"] - u0) / (u1 - u0) * FOV_ELEVATION_DEG \
                - FOV_ELEVATION_DEG / 2.0
            rel_az = (s["v"] - v0) / (v1 - v0) * FOV_AZIMUTH_DEG - FOV_AZIMUTH_DEG / 2.0
            # v increases posteriorly; for the left eye that is decreasing azimuth.
            sign = +1.0 if hemi == "right" else -1.0
            s["azimuth_deg"] = EYE_CENTRE_AZIMUTH_DEG[hemi] + sign * rel_az
            out.append(s)
        return pd.concat(out, ignore_index=True)

    # ------------------------------------------------------------- public API
    @property
    def n_columns(self) -> dict:
        return self.columns.groupby("hemisphere")["column_id"].nunique().to_dict()

    def column_directions(self) -> pd.DataFrame:
        """One row per (hemisphere, column) with its visual direction."""
        return (self.columns
                .groupby(["hemisphere", "column_id"])[["azimuth_deg", "elevation_deg"]]
                .mean().reset_index())

    def receptive_fields(self, cell_type: str) -> pd.DataFrame:
        """
        Receptive-field centre and radius for every neuron of `cell_type`,
        derived from the real column assignments of its presynaptic partners.

        Returns columns: root_id, idx, side, azimuth_deg, elevation_deg,
                         rf_radius_deg, n_input_columns, input_synapses
        """
        if cell_type in self._cache:
            return self._cache[cell_type]

        c = self.c
        targets = c.by_cell_type(cell_type)
        if targets.empty:
            raise KeyError("no neurons of type %r in %s" % (cell_type, c.dataset))

        # Direction of every column-assigned neuron, by simulation index.
        col = self.columns.copy()
        col["idx"] = col["root_id"].map(
            dict(zip(c.neurons["root_id"], c.neurons["idx"]))
        )
        col = col.dropna(subset=["idx"])
        idx2az = dict(zip(col["idx"].astype(int), col["azimuth_deg"]))
        idx2el = dict(zip(col["idx"].astype(int), col["elevation_deg"]))
        known = np.zeros(c.n, dtype=bool)
        az = np.zeros(c.n)
        el = np.zeros(c.n)
        ii = np.fromiter(idx2az.keys(), dtype=np.int64)
        known[ii] = True
        az[ii] = np.fromiter(idx2az.values(), dtype=np.float64)
        el[ii] = np.fromiter(idx2el.values(), dtype=np.float64)

        rows = []
        w = c.w.tocsc()
        for _, t in targets.iterrows():
            j = int(t["idx"])
            colslice = w[:, j].tocoo()
            pres = colslice.row
            syn = np.abs(colslice.data).astype(np.float64)
            m = known[pres]
            if not m.any():
                rows.append((int(t["root_id"]), j, t["side"], np.nan, np.nan,
                             np.nan, 0, 0))
                continue
            p, s = pres[m], syn[m]
            wsum = s.sum()
            a = float((az[p] * s).sum() / wsum)
            e = float((el[p] * s).sum() / wsum)
            var = float((s * ((az[p] - a) ** 2 + (el[p] - e) ** 2)).sum() / wsum)
            rows.append((int(t["root_id"]), j, t["side"], a, e,
                         float(np.sqrt(var)), int(m.sum()), int(wsum)))

        df = pd.DataFrame(rows, columns=[
            "root_id", "idx", "side", "azimuth_deg", "elevation_deg",
            "rf_radius_deg", "n_input_columns", "input_synapses"])
        self._cache[cell_type] = df
        return df


@lru_cache(maxsize=1)
def load_retinotopy(connectome=None) -> Retinotopy:
    if connectome is None:
        from brain.neurons.registry import load_connectome
        connectome = load_connectome()
    return Retinotopy(connectome)
