"""
Motor output: reading behavioural commands off real descending neurons (DNs).

IMPORTANT SCOPE LIMIT
---------------------
FlyWire FAFB v783 is a BRAIN connectome. The motor neurons that move legs and
wings are in the ventral nerve cord, which is NOT part of this dataset.

Therefore the last neural stage this project can simulate is the descending
neuron population.

Everything past the DNs -- VNC, muscles, legs, wings and body dynamics -- is
an explicitly labelled engineering body model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================================
# PUBLISHED DN -> BEHAVIOUR ASSOCIATIONS
# ============================================================================

@dataclass(frozen=True)
class DNCommand:
    """A published association between a descending neuron type and behaviour."""

    cell_type: str
    channel: str
    behaviour: str
    laterality: str
    citation: str
    doi: str


DN_COMMANDS = (
    DNCommand(
        "DNp01",
        "escape_takeoff",
        "Giant Fibre. Drives short-mode escape: mesothoracic leg "
        "extension and takeoff within ~5 ms, without preparatory "
        "wing raising.",
        "bilateral",
        "von Reyn et al. 2014, Nat Neurosci 17:962-965; "
        "Card & Dickinson 2008, J Exp Biol 211:341-353",
        "10.1038/nn.3741",
    ),

    DNCommand(
        "DNp02",
        "escape_long_mode",
        "Contributes to long-mode GF-independent escape with "
        "preparatory wing raising and a directed jump.",
        "ipsilateral",
        "Namiki et al. 2018, eLife 7:e34272; "
        "Cheong et al. 2024, eLife 13:RP96323",
        "10.7554/eLife.34272",
    ),

    DNCommand(
        "DNp04",
        "escape_long_mode",
        "Looming-responsive descending neuron in the escape network.",
        "ipsilateral",
        "Namiki et al. 2018, eLife 7:e34272",
        "10.7554/eLife.34272",
    ),

    DNCommand(
        "DNp11",
        "escape_long_mode",
        "Escape-network descending neuron projecting to leg neuropils.",
        "ipsilateral",
        "Namiki et al. 2018, eLife 7:e34272",
        "10.7554/eLife.34272",
    ),

    DNCommand(
        "DNp09",
        "stop_freeze",
        "Drives stopping / freezing; suppresses walking.",
        "bilateral",
        "Zacarias et al. 2018, Nat Commun 9:3697",
        "10.1038/s41467-018-05875-1",
    ),

    DNCommand(
        "DNa01",
        "turn",
        "Steering: activity biases the fly's turning.",
        "ipsilateral",
        "Rayshubskiy et al. 2024, Nature 631:135-143",
        "10.1038/s41586-024-07523-9",
    ),

    DNCommand(
        "DNa02",
        "turn",
        "Steering: unilateral activity drives ipsilateral turning.",
        "ipsilateral",
        "Rayshubskiy et al. 2024, Nature 631:135-143",
        "10.1038/s41586-024-07523-9",
    ),

    DNCommand(
        "MDN",
        "backward_walk",
        "Moonwalker descending neuron: drives backward walking.",
        "bilateral",
        "Bidaye et al. 2014, Science 344:97-101",
        "10.1126/science.1249964",
    ),
)


CHANNELS = tuple(
    sorted({d.channel for d in DN_COMMANDS})
)


# ============================================================================
# ENGINEERING RATE CONVERSION
# ============================================================================

CHANNEL_HALF_MAX_HZ = 60.0

# ---------------------------------------------------------------------------
# Engineering motor gains
# ---------------------------------------------------------------------------
#
# FAFB v783 does not contain the VNC leg motor circuitry required to simulate
# biological forward walking directly.
#
# Therefore the four outputs below are ENGINEERING interfaces between real
# descending-neuron activity and the downstream FlyBody/VNC model.
#
# IMPORTANT:
#   forward is NOT claimed to be a biologically validated forward-walking DN.
#   It is derived from bilateral DNa01/DNa02 steering-network activity.
#
# The important architectural change is that the body receives four separate
# command channels rather than one locomotion value plus a signed turn value.
# ---------------------------------------------------------------------------

FORWARD_GAIN = 4.0
BACKWARD_GAIN = 1.0
TURN_GAIN = 1.0


class DescendingReadout:
    """
    Turns DN spiking in the simulation into motor-channel activations.

    All neural activity entering this class comes from the real FlyWire
    connectome simulation.

    No stimulus information enters this class.
    """

    def __init__(self, connectome):

        self.c = connectome

        self.commands = []
        self.missing = []

        for cmd in DN_COMMANDS:

            cells = connectome.by_cell_type(
                cmd.cell_type
            )

            if cells.empty:

                self.missing.append(
                    cmd.cell_type
                )

                continue

            self.commands.append(
                (cmd, cells)
            )

        # All descending neurons.
        self.all_dn = connectome.neurons[
            connectome.neurons["super_class"]
            .astype(str)
            == "descending"
        ]

        self.dn_idx = (
            self.all_dn["idx"]
            .to_numpy(dtype=np.int64)
        )

        # Index arrays by cell type and side.
        self.tracked = {}

        for cmd, cells in self.commands:

            for side in ("left", "right"):

                s = cells[
                    cells["side"].astype(str)
                    == side
                ]

                self.tracked[
                    (cmd.cell_type, side)
                ] = s["idx"].to_numpy(
                    dtype=np.int64
                )

        # Real proboscis motor neurons present in the brain dataset.
        self.proboscis_idx = (
            self._label_group_indices(
                "proboscis_motor"
            )
        )

    def _label_group_indices(
        self,
        group: str,
    ) -> np.ndarray:

        try:

            from brain.neurons.labels import (
                functional_group,
            )

            rids = functional_group(
                group
            )

        except Exception:

            return np.empty(
                0,
                dtype=np.int64,
            )

        return np.array(
            [
                self.c.idx(r)
                for r in rids
                if int(r) in self.c._id2idx
            ],
            dtype=np.int64,
        )

    # ========================================================================
    # PROBOSCIS
    # ========================================================================

    def proboscis_drive(
        self,
        spike_counts: np.ndarray,
        window_ms: float,
    ) -> float:

        if (
            self.proboscis_idx.size == 0
            or window_ms <= 0
        ):
            return 0.0

        hz = float(
            spike_counts[
                self.proboscis_idx
            ].sum()
            / self.proboscis_idx.size
            / (window_ms * 1e-3)
        )

        return (
            hz
            / (hz + CHANNEL_HALF_MAX_HZ)
        )

    # ========================================================================
    # RATES
    # ========================================================================

    def rates(
        self,
        spike_counts: np.ndarray,
        window_ms: float,
    ) -> dict:

        if window_ms <= 0:
            window_ms = 1.0

        out = {}

        for (
            ctype,
            side,
        ), idx in self.tracked.items():

            if idx.size == 0:
                continue

            out[
                "%s_%s"
                % (ctype, side)
            ] = float(
                spike_counts[idx].sum()
                / idx.size
                / (window_ms * 1e-3)
            )

        return out

    # ========================================================================
    # MOTOR CHANNELS
    # ========================================================================

    def channels(
        self,
        spike_counts: np.ndarray,
        window_ms: float,
    ) -> dict:
        """
        Convert real DN spike counts into independent motor commands.

        Four ground-movement commands are exposed separately:

            forward
            backward
            left
            right

        They are intentionally independent at the body interface.

        Biological scope:
        -----------------
        DNa01/DNa02 have published steering associations and are therefore
        used for left/right steering. Because FAFB v783 does not contain the
        VNC leg motor circuitry, there is no direct biological forward-walking
        output available here.

        For that reason, `forward` is an explicitly labelled engineering
        locomotion command derived from bilateral DNa01/DNa02 activity.
        It must NOT be interpreted as a claim that DNa01/DNa02 are
        forward-walking neurons.

        MDN remains a separate backward-walking command.
        """

        if window_ms <= 0:
            window_ms = 1.0

        acc = {
            ch: []
            for ch in CHANNELS
        }

        left_turn = []
        right_turn = []
        backward_values = []

        for cmd, cells in self.commands:

            for side in ("left", "right"):

                idx = self.tracked[
                    (cmd.cell_type, side)
                ]

                if idx.size == 0:
                    continue

                hz = float(
                    spike_counts[idx].sum()
                    / idx.size
                    / (window_ms * 1e-3)
                )

                act = (
                    hz
                    / (
                        hz
                        + CHANNEL_HALF_MAX_HZ
                    )
                )

                acc[
                    cmd.channel
                ].append(act)

                if cmd.channel == "turn":
                    if side == "left":
                        left_turn.append(act)
                    else:
                        right_turn.append(act)

                elif cmd.channel == "backward_walk":
                    backward_values.append(act)

        # Published behavioural channels remain available.
        res = {
            ch: (
                float(np.max(values))
                if values
                else 0.0
            )
            for ch, values in acc.items()
        }

        # -------------------------------------------------------------------
        # INDEPENDENT LEFT / RIGHT STEERING COMMANDS
        # -------------------------------------------------------------------

        left = (
            float(np.mean(left_turn))
            if left_turn
            else 0.0
        )

        right = (
            float(np.mean(right_turn))
            if right_turn
            else 0.0
        )

        # Left and right are independent positive command strengths.
        res["left"] = float(
            np.clip(
                left * TURN_GAIN,
                0.0,
                1.0,
            )
        )

        res["right"] = float(
            np.clip(
                right * TURN_GAIN,
                0.0,
                1.0,
            )
        )

        # Keep the legacy signed bias for diagnostics/backward compatibility.
        res["turn_bias"] = (
            res["right"] - res["left"]
        )

        # -------------------------------------------------------------------
        # INDEPENDENT FORWARD COMMAND
        # -------------------------------------------------------------------
        #
        # Bilateral DNa01/DNa02 activity is used as the engineering source.
        # This preserves the successful Phase 1 locomotion mechanism while
        # separating its output from the left/right body commands.
        #
        # This is an engineering interface, not a biological forward-DN claim.
        # -------------------------------------------------------------------

        turn_activity = max(
            left,
            right,
        )

        res["forward"] = float(
            np.clip(
                turn_activity * FORWARD_GAIN,
                0.0,
                1.0,
            )
        )

        # -------------------------------------------------------------------
        # INDEPENDENT BACKWARD COMMAND
        # -------------------------------------------------------------------

        backward_activity = (
            float(np.max(backward_values))
            if backward_values
            else 0.0
        )

        res["backward"] = float(
            np.clip(
                backward_activity * BACKWARD_GAIN,
                0.0,
                1.0,
            )
        )

        # -------------------------------------------------------------------
        # Keep the old locomotion channel as a compatibility diagnostic.
        # New body code does not use it when `forward` is present.
        # -------------------------------------------------------------------

        res["locomotion_drive"] = res["forward"]

        return res

    # ========================================================================
    # ESCAPE LATERALITY
    # ========================================================================

    def escape_laterality(
        self,
        spike_counts: np.ndarray,
    ) -> float:

        tot = {
            "left": 0.0,
            "right": 0.0,
        }

        for cmd, _ in self.commands:

            if not cmd.channel.startswith(
                "escape"
            ):
                continue

            for side in ("left", "right"):

                idx = self.tracked[
                    (cmd.cell_type, side)
                ]

                if idx.size:

                    tot[side] += (
                        float(
                            spike_counts[idx].sum()
                        )
                        / idx.size
                    )

        total = (
            tot["left"]
            + tot["right"]
        )

        if total == 0.0:
            return 0.0

        return (
            tot["right"]
            - tot["left"]
        ) / total

    # ========================================================================
    # PROVENANCE
    # ========================================================================

    @property
    def provenance(self) -> dict:

        return {
            "n_descending_neurons_in_dataset": int(
                len(self.all_dn)
            ),

            "n_dn_types_in_dataset": int(
                self.all_dn[
                    "primary_type"
                ].nunique()
            ),

            "modelled_dn_types": [
                c.cell_type
                for c, _
                in self.commands
            ],

            "not_modelled_note": (
                "Other DN types in FlyWire v783 have "
                "no established behavioural assignment "
                "used here."
            ),

            "vnc_limitation": (
                "Leg and wing motor neurons are in the "
                "ventral nerve cord and are absent from "
                "FlyWire FAFB v783."
            ),

            "motor_interface": {
                "type": "four-channel engineering interface",
                "channels": ["forward", "backward", "left", "right"],
                "forward_source": "bilateral DNa01/DNa02 steering activity",
                "forward_gain": float(FORWARD_GAIN),
                "backward_source": "MDN",
                "backward_gain": float(BACKWARD_GAIN),
                "turn_source": "DNa01/DNa02 side-specific activity",
                "turn_gain": float(TURN_GAIN),
                "biological_claim": False,
                "purpose": (
                    "provide separate ground-movement commands "
                    "without supplying stimulus information"
                ),
            },

            "commands": [
                {
                    "cell_type": c.cell_type,
                    "channel": c.channel,
                    "behaviour": c.behaviour,
                    "citation": c.citation,
                    "doi": c.doi,
                }
                for c, _
                in self.commands
            ],
        }