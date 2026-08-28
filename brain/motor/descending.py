"""
Motor output: reading behavioural commands off real descending neurons (DNs).

IMPORTANT SCOPE LIMIT
---------------------
FlyWire FAFB v783 is a BRAIN connectome. The motor neurons that move legs and
wings are in the ventral nerve cord, which is NOT part of this dataset. The 110
neurons FlyWire labels `motor` innervate head structures (proboscis, antennae),
not the flight or leg muscles.

Therefore the last neural stage this project can simulate is the descending
neuron population (1,305 DNs in v783). That is the genuine output of the brain:
DNs are the only pathway from brain to ventral nerve cord. Everything past the
DNs -- muscles, legs, wings, body dynamics -- is a body model, and is labelled
as such in the UI.

PROVENANCE
----------
A. REAL DATA   : DN identities, cell types, sides, and all connectivity
                 driving them (FlyWire v783).
B. PUBLISHED   : the DN -> behaviour associations in DN_COMMANDS below. Each
                 entry carries its citation. These come from optogenetic
                 activation and silencing experiments, not from the connectome.
C. APPROX      : the mapping from DN firing rate to a 0-1 "command strength"
                 is a saturating function; thresholds are our choices.
D. ENGINEERING : the bookkeeping in this file.

Nothing here is a behavioural rule keyed to a stimulus. These functions never
see the stimulus; they see only DN spike counts produced by the simulation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DNCommand:
    """A published association between a descending neuron type and behaviour."""
    cell_type: str
    channel: str          # motor channel this DN contributes to
    behaviour: str        # plain-language description
    laterality: str       # 'ipsilateral', 'bilateral', or 'none'
    citation: str
    doi: str


# --- B: published descending-neuron -> behaviour associations ---------------
# Only DN types with direct experimental evidence are listed. Any DN not in
# this table contributes to no behavioural channel and is reported by the UI as
# "Not currently modeled".
DN_COMMANDS = (
    DNCommand(
        "DNp01", "escape_takeoff",
        "Giant Fibre. Drives short-mode escape: mesothoracic leg extension and "
        "takeoff within ~5 ms, without preparatory wing raising.",
        "bilateral",
        "von Reyn et al. 2014, Nat Neurosci 17:962-965; "
        "Card & Dickinson 2008, J Exp Biol 211:341-353",
        "10.1038/nn.3741",
    ),
    DNCommand(
        "DNp02", "escape_long_mode",
        "Contributes to long-mode (GF-independent) escape with preparatory "
        "wing raising and a directed jump.",
        "ipsilateral",
        "Namiki et al. 2018, eLife 7:e34272; Cheong et al. 2024, eLife 13:RP96323",
        "10.7554/eLife.34272",
    ),
    DNCommand(
        "DNp04", "escape_long_mode",
        "Looming-responsive descending neuron in the escape network.",
        "ipsilateral",
        "Namiki et al. 2018, eLife 7:e34272",
        "10.7554/eLife.34272",
    ),
    DNCommand(
        "DNp11", "escape_long_mode",
        "Escape-network descending neuron projecting to leg neuropils.",
        "ipsilateral",
        "Namiki et al. 2018, eLife 7:e34272",
        "10.7554/eLife.34272",
    ),
    DNCommand(
        "DNp09", "stop_freeze",
        "Drives stopping / freezing; suppresses walking.",
        "bilateral",
        "Zacarias et al. 2018, Nat Commun 9:3697",
        "10.1038/s41467-018-05875-1",
    ),
    DNCommand(
        "DNa01", "turn",
        "Steering: activity biases the fly's turning.",
        "ipsilateral",
        "Rayshubskiy et al. 2024, Nature 631:135-143",
        "10.1038/s41586-024-07523-9",
    ),
    DNCommand(
        "DNa02", "turn",
        "Steering: unilateral activity drives ipsilateral turning.",
        "ipsilateral",
        "Rayshubskiy et al. 2024, Nature 631:135-143",
        "10.1038/s41586-024-07523-9",
    ),
    DNCommand(
        "MDN", "backward_walk",
        "Moonwalker descending neuron: drives backward walking.",
        "bilateral",
        "Bidaye et al. 2014, Science 344:97-101",
        "10.1126/science.1249964",
    ),
)

CHANNELS = tuple(sorted({d.channel for d in DN_COMMANDS}))

# C: rate at which a channel is considered fully driven (Hz, per DN).
CHANNEL_HALF_MAX_HZ = 60.0


class DescendingReadout:
    """Turns DN spiking in the simulation into motor channel activations."""

    def __init__(self, connectome):
        self.c = connectome
        self.commands = []
        self.missing = []
        for cmd in DN_COMMANDS:
            cells = connectome.by_cell_type(cmd.cell_type)
            if cells.empty:
                self.missing.append(cmd.cell_type)
                continue
            self.commands.append((cmd, cells))

        # All descending neurons, for the "brain output" activity display.
        self.all_dn = connectome.neurons[
            connectome.neurons["super_class"].astype(str) == "descending"]
        self.dn_idx = self.all_dn["idx"].to_numpy(dtype=np.int64)

        # Index arrays per (cell_type, side)
        self.tracked = {}
        for cmd, cells in self.commands:
            for side in ("left", "right"):
                s = cells[cells["side"].astype(str) == side]
                self.tracked[(cmd.cell_type, side)] = s["idx"].to_numpy(dtype=np.int64)

        # Proboscis motor neurons are a genuine motor output that IS present in
        # the brain dataset (they innervate head muscles, not the VNC).
        # Labelled in FlyWire v783 by Claire McKellar.
        self.proboscis_idx = self._label_group_indices("proboscis_motor")

    def _label_group_indices(self, group: str) -> np.ndarray:
        try:
            from brain.neurons.labels import functional_group
            rids = functional_group(group)
        except Exception:
            return np.empty(0, dtype=np.int64)
        return np.array([self.c.idx(r) for r in rids
                         if int(r) in self.c._id2idx], dtype=np.int64)

    def proboscis_drive(self, spike_counts: np.ndarray, window_ms: float) -> float:
        """
        Saturating activation (0-1) of the real proboscis motor neurons.

        This is the one place the model reaches an actual motor neuron rather
        than stopping at a descending neuron.
        """
        if self.proboscis_idx.size == 0 or window_ms <= 0:
            return 0.0
        hz = float(spike_counts[self.proboscis_idx].sum()
                   / self.proboscis_idx.size / (window_ms * 1e-3))
        return hz / (hz + CHANNEL_HALF_MAX_HZ)

    # ------------------------------------------------------------------ read
    def rates(self, spike_counts: np.ndarray, window_ms: float) -> dict:
        """Firing rate (Hz) of every tracked DN type, per side."""
        if window_ms <= 0:
            window_ms = 1.0
        out = {}
        for (ctype, side), idx in self.tracked.items():
            if idx.size == 0:
                continue
            out["%s_%s" % (ctype, side)] = float(
                spike_counts[idx].sum() / idx.size / (window_ms * 1e-3))
        return out

    def channels(self, spike_counts: np.ndarray, window_ms: float) -> dict:
        """
        Activation (0-1) of each published motor channel, plus turn bias.

        Channel activation is a saturating function of the mean firing rate of
        the DNs assigned to that channel. This is the only step that is not
        connectome-derived, and it introduces no stimulus dependence.
        """
        if window_ms <= 0:
            window_ms = 1.0
        acc = {ch: [] for ch in CHANNELS}
        left_turn, right_turn = [], []

        for cmd, cells in self.commands:
            for side in ("left", "right"):
                idx = self.tracked[(cmd.cell_type, side)]
                if idx.size == 0:
                    continue
                hz = float(spike_counts[idx].sum() / idx.size / (window_ms * 1e-3))
                act = hz / (hz + CHANNEL_HALF_MAX_HZ)
                acc[cmd.channel].append(act)
                if cmd.channel == "turn":
                    (left_turn if side == "left" else right_turn).append(act)

        res = {ch: (float(np.max(v)) if v else 0.0) for ch, v in acc.items()}
        # Ipsilateral steering convention: net bias toward the more active side.
        l = float(np.mean(left_turn)) if left_turn else 0.0
        r = float(np.mean(right_turn)) if right_turn else 0.0
        res["turn_bias"] = r - l          # >0 : turn right
        return res

    def escape_laterality(self, spike_counts: np.ndarray) -> float:
        """
        Left/right imbalance across escape DNs, in [-1, 1] (>0 = right side more
        active). Used by the body model to direct the escape jump.
        """
        tot = {"left": 0.0, "right": 0.0}
        for cmd, _ in self.commands:
            if not cmd.channel.startswith("escape"):
                continue
            for side in ("left", "right"):
                idx = self.tracked[(cmd.cell_type, side)]
                if idx.size:
                    tot[side] += float(spike_counts[idx].sum()) / idx.size
        s = tot["left"] + tot["right"]
        return 0.0 if s == 0 else (tot["right"] - tot["left"]) / s

    # ----------------------------------------------------------- provenance
    @property
    def provenance(self) -> dict:
        return {
            "n_descending_neurons_in_dataset": int(len(self.all_dn)),
            "n_dn_types_in_dataset": int(self.all_dn["primary_type"].nunique()),
            "modelled_dn_types": [c.cell_type for c, _ in self.commands],
            "not_modelled_note": (
                "The other DN types in FlyWire v783 have no established "
                "behavioural assignment used here and are reported as "
                "'Not currently modeled'."),
            "vnc_limitation": (
                "Leg and wing motor neurons are in the ventral nerve cord and "
                "are absent from FlyWire FAFB v783. Simulation ends at the "
                "descending neurons."),
            "commands": [
                {"cell_type": c.cell_type, "channel": c.channel,
                 "behaviour": c.behaviour, "citation": c.citation, "doi": c.doi}
                for c, _ in self.commands
            ],
        }
