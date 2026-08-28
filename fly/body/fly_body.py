"""
The digital fly's body.

SCOPE AND HONESTY
-----------------
This file is NOT part of the connectome simulation. It is a body model, and it
sits downstream of everything neural. It is category C/D: an engineering
approximation whose only inputs are motor-channel activations computed from real
descending-neuron spiking.

It contains no stimulus logic. It never sees the rock, the food or the wind. It
cannot: its `update()` signature takes motor channels and nothing else. If the
neural simulation produces no descending activity, the body does nothing.

Where published kinematics exist, they are used and cited:
  - Giant-Fibre ("short-mode") escape: takeoff ~5 ms after GF spiking, with no
    preparatory wing raising, and a shorter, less directed jump.
  - Non-GF ("long-mode") escape: ~200 ms of preparatory wing raising, producing
    a slower but directed takeoff away from the threat.
    Card & Dickinson 2008, J Exp Biol 211:341-353, doi:10.1242/jeb.012682
    von Reyn et al. 2014, Nat Neurosci 17:962-965, doi:10.1038/nn.3741
  - Walking speed range and turning are set to ordinary reported values.
    Strauss & Heisenberg 1990, J Comp Physiol A 167:403
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict


# --- C: thresholds at which a motor channel engages a behaviour -------------
ESCAPE_THRESHOLD = 0.35        # channel activation triggering takeoff
LONG_MODE_THRESHOLD = 0.30
FREEZE_THRESHOLD = 0.30
BACKWARD_THRESHOLD = 0.30
PROBOSCIS_THRESHOLD = 0.15

# --- B: published escape kinematics ----------------------------------------
GF_TAKEOFF_LATENCY_MS = 5.0        # short mode: near-immediate
LONG_MODE_PREP_MS = 200.0          # long mode: preparatory wing raising
JUMP_SPEED_MM_S = 750.0

# --- C: ordinary locomotor values ------------------------------------------
MAX_WALK_SPEED_MM_S = 25.0
MAX_TURN_RATE_DEG_S = 400.0
BACKWARD_SPEED_MM_S = 8.0


@dataclass
class BodyState:
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    heading_deg: float = 0.0        # 0 = +x, positive = counter-clockwise
    speed_mm_s: float = 0.0
    turn_rate_deg_s: float = 0.0
    wing_angle_deg: float = 0.0     # 0 = folded, 90 = fully raised
    proboscis_extension: float = 0.0  # 0 = retracted, 1 = fully extended
    leg_extension: float = 0.0      # mesothoracic leg extension during takeoff
    airborne: bool = False
    vz_mm_s: float = 0.0
    behaviour: str = "resting"
    escape_mode: str = ""           # 'short' (GF) or 'long' (non-GF)


class FlyBody:
    """Kinematic body driven only by descending-neuron motor channels."""

    GRAVITY_MM_S2 = 9810.0

    def __init__(self):
        self.state = BodyState()
        self._escape_armed_at = None      # sim time when escape command crossed
        self._escape_mode = None
        self._escape_complete = False     # latch: one threat -> one takeoff
        self._t_ms = 0.0
        self.events = []                  # timestamped behavioural events

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        self.state = BodyState()
        self._escape_armed_at = None
        self._escape_mode = None
        self._escape_complete = False
        self._t_ms = 0.0
        self.events = []

    # ----------------------------------------------------------------- update
    def update(self, dt_ms: float, channels: dict, t_ms: float,
               escape_laterality: float = 0.0,
               proboscis_drive: float = 0.0) -> BodyState:
        """
        Advance the body by dt_ms.

        `channels` comes from DescendingReadout.channels(); it is derived purely
        from real descending-neuron spike counts. `proboscis_drive` comes from
        real proboscis motor neurons, which ARE in the FlyWire brain dataset.
        """
        s = self.state
        self._t_ms = t_ms
        dt_s = dt_ms / 1000.0

        takeoff = channels.get("escape_takeoff", 0.0)     # DNp01, Giant Fibre
        long_mode = channels.get("escape_long_mode", 0.0)  # DNp02/04/11
        freeze = channels.get("stop_freeze", 0.0)          # DNp09
        backward = channels.get("backward_walk", 0.0)      # MDN
        turn_bias = channels.get("turn_bias", 0.0)         # DNa01/DNa02

        # A completed escape cannot re-arm until the command has fallen away
        # again, so one threat produces one takeoff rather than a loop.
        if self._escape_complete and takeoff < ESCAPE_THRESHOLD \
                and long_mode < LONG_MODE_THRESHOLD:
            self._escape_complete = False

        # --- escape: arm on the first suprathreshold command ---------------
        if (self._escape_armed_at is None and not s.airborne
                and not self._escape_complete):
            if takeoff >= ESCAPE_THRESHOLD:
                self._escape_armed_at = t_ms
                self._escape_mode = "short"
                self._log(t_ms, "GF (DNp01) escape command", takeoff)
            elif long_mode >= LONG_MODE_THRESHOLD:
                self._escape_armed_at = t_ms
                self._escape_mode = "long"
                self._log(t_ms, "long-mode escape command (DNp02/04/11)", long_mode)

        if self._escape_armed_at is not None and not s.airborne:
            elapsed = t_ms - self._escape_armed_at
            s.escape_mode = self._escape_mode
            if self._escape_mode == "short":
                # B: no preparatory wing raising; leg extension then takeoff.
                s.behaviour = "escape (short mode, GF-driven)"
                s.leg_extension = min(1.0, elapsed / GF_TAKEOFF_LATENCY_MS)
                if elapsed >= GF_TAKEOFF_LATENCY_MS:
                    self._takeoff(escape_laterality, directed=False, t_ms=t_ms)
            else:
                # B: ~200 ms of wing raising, then a directed jump.
                s.behaviour = "escape (long mode, preparing)"
                s.wing_angle_deg = 90.0 * min(1.0, elapsed / LONG_MODE_PREP_MS)
                s.leg_extension = min(1.0, elapsed / LONG_MODE_PREP_MS)
                if elapsed >= LONG_MODE_PREP_MS:
                    self._takeoff(escape_laterality, directed=True, t_ms=t_ms)

        # --- ballistic flight ----------------------------------------------
        if s.airborne:
            s.vz_mm_s -= self.GRAVITY_MM_S2 * dt_s
            s.z_mm += s.vz_mm_s * dt_s
            if s.z_mm <= 0.0:
                s.z_mm, s.vz_mm_s, s.airborne = 0.0, 0.0, False
                s.speed_mm_s = 0.0
                s.behaviour = "landed"
                s.leg_extension = 0.0
                self._escape_armed_at = None
                self._escape_mode = None
                self._escape_complete = True
                self._log(t_ms, "landed", 0.0)
            self._translate(dt_s)
            return s

        # --- ground behaviours (only if not escaping) ----------------------
        if self._escape_armed_at is None:
            if freeze >= FREEZE_THRESHOLD:
                s.speed_mm_s = 0.0
                s.turn_rate_deg_s = 0.0
                s.behaviour = "freezing (DNp09)"
            elif backward >= BACKWARD_THRESHOLD:
                s.speed_mm_s = -BACKWARD_SPEED_MM_S * backward
                s.behaviour = "walking backward (MDN)"
            elif abs(turn_bias) > 0.02:
                s.turn_rate_deg_s = -MAX_TURN_RATE_DEG_S * turn_bias
                s.speed_mm_s = MAX_WALK_SPEED_MM_S * min(1.0, abs(turn_bias) * 2)
                s.behaviour = "turning %s" % ("right" if turn_bias > 0 else "left")
            else:
                s.speed_mm_s *= 0.9
                s.turn_rate_deg_s = 0.0
                if abs(s.speed_mm_s) < 0.5:
                    s.speed_mm_s = 0.0
                    s.behaviour = "resting"

            # proboscis extension from real brain motor neurons
            target = 1.0 if proboscis_drive >= PROBOSCIS_THRESHOLD else 0.0
            s.proboscis_extension += (target - s.proboscis_extension) * min(
                1.0, dt_ms / 40.0)
            if s.proboscis_extension > 0.5 and s.behaviour == "resting":
                s.behaviour = "proboscis extension (feeding)"

            s.wing_angle_deg *= 0.92

        s.heading_deg = (s.heading_deg + s.turn_rate_deg_s * dt_s) % 360.0
        self._translate(dt_s)
        return s

    # ---------------------------------------------------------------- helpers
    def _takeoff(self, laterality: float, directed: bool, t_ms: float) -> None:
        s = self.state
        s.airborne = True
        s.vz_mm_s = JUMP_SPEED_MM_S * 0.6
        s.speed_mm_s = JUMP_SPEED_MM_S
        s.wing_angle_deg = 90.0
        if directed:
            # Long mode is directed away from the more active escape side.
            s.heading_deg = (s.heading_deg - 90.0 * laterality) % 360.0
        s.behaviour = "airborne (%s-mode takeoff)" % self._escape_mode
        self._log(t_ms, "takeoff (%s mode)" % self._escape_mode, 1.0)

    def _translate(self, dt_s: float) -> None:
        s = self.state
        rad = math.radians(s.heading_deg)
        s.x_mm += math.cos(rad) * s.speed_mm_s * dt_s
        s.y_mm += math.sin(rad) * s.speed_mm_s * dt_s

    def _log(self, t_ms: float, what: str, strength: float) -> None:
        self.events.append({"t_ms": round(t_ms, 1), "event": what,
                            "strength": round(float(strength), 3)})

    def as_dict(self) -> dict:
        d = asdict(self.state)
        d["events"] = self.events[-12:]
        return d

    @property
    def provenance(self) -> dict:
        return {
            "role": "body model, downstream of all neural simulation",
            "inputs": ("descending-neuron motor channels and proboscis motor "
                       "neuron drive only; the body never sees the stimulus"),
            "published_kinematics": {
                "short_mode_takeoff_latency_ms": GF_TAKEOFF_LATENCY_MS,
                "long_mode_preparation_ms": LONG_MODE_PREP_MS,
                "citation": ("Card & Dickinson 2008, J Exp Biol 211:341; "
                             "von Reyn et al. 2014, Nat Neurosci 17:962"),
            },
            "caveat": ("Body dynamics are a kinematic approximation, not a "
                       "biomechanical model. Leg and wing motor neurons are in "
                       "the ventral nerve cord and absent from FlyWire FAFB."),
        }
