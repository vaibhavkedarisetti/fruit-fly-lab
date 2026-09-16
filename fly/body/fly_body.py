"""
The digital fly's body.

SCOPE AND HONESTY
-----------------
This is NOT part of the FlyWire connectome simulation.

It is a downstream kinematic body model. Its only neural inputs are
motor-channel activations produced from descending-neuron activity.

The body does NOT see:
    - the exit
    - odor
    - food
    - rock
    - wind
    - distance to the target
    - any environment state

Environment -> sensory encoding -> neural simulation
           -> descending motor readout -> this body
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict


# ============================================================================
# MOTOR THRESHOLDS
# ============================================================================

ESCAPE_THRESHOLD = 0.35
LONG_MODE_THRESHOLD = 0.30
FREEZE_THRESHOLD = 0.30
BACKWARD_THRESHOLD = 0.30
PROBOSCIS_THRESHOLD = 0.15


# ============================================================================
# ESCAPE KINEMATICS
# ============================================================================

GF_TAKEOFF_LATENCY_MS = 5.0
LONG_MODE_PREP_MS = 200.0

JUMP_SPEED_MM_S = 750.0


# ============================================================================
# GROUND LOCOMOTION
# ============================================================================

MAX_WALK_SPEED_MM_S = 25.0
MAX_TURN_RATE_DEG_S = 400.0
BACKWARD_SPEED_MM_S = 8.0


# Engineering locomotion interface.
#
# IMPORTANT:
# This is NOT a claim that a particular DN is a validated forward-walking
# command. It is an interface between the descending-neuron readout and the
# kinematic body model.
MIN_LOCOMOTION_DRIVE = 0.12

LOCOMOTION_RESPONSE_EXPONENT = 0.65
LOCOMOTION_SMOOTHING = 0.35
LOCOMOTION_PERSISTENCE_MS = 250.0


# ============================================================================
# NUMERICAL INTEGRATION
# ============================================================================

# Maximum internal timestep used for body physics.
#
# This prevents a large external neural timestep from causing a takeoff and
# landing to happen inside one numerical update.
MAX_INTERNAL_STEP_MS = 5.0


# ============================================================================
# STATE
# ============================================================================

@dataclass
class BodyState:
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0

    # 0 degrees = +x.
    # Positive = counter-clockwise.
    heading_deg: float = 0.0

    speed_mm_s: float = 0.0
    turn_rate_deg_s: float = 0.0

    wing_angle_deg: float = 0.0
    proboscis_extension: float = 0.0
    leg_extension: float = 0.0

    airborne: bool = False
    vz_mm_s: float = 0.0

    behaviour: str = "resting"
    escape_mode: str = ""


# ============================================================================
# BODY
# ============================================================================

class FlyBody:
    """
    Kinematic fly body driven by descending-neuron motor channels.

    The body never receives environmental information.
    """

    GRAVITY_MM_S2 = 9810.0

    def __init__(self):
        self.state = BodyState()

        self._escape_armed_at: float | None = None
        self._escape_mode: str | None = None
        self._escape_complete = False

        # FIX:
        # Keep escape preparation as an accumulated duration rather than
        # deriving it from absolute t_ms.
        #
        # This keeps the state machine consistent when the caller uses
        # coarse external timesteps such as 16.67 ms, 50 ms, or 100 ms.
        self._escape_elapsed_ms = 0.0

        self._t_ms = 0.0

        self._locomotion_drive = 0.0
        self._locomotion_last_active_ms = -1e9

        self.events: list[dict] = []

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(self) -> None:
        self.state = BodyState()

        self._escape_armed_at = None
        self._escape_mode = None
        self._escape_complete = False

        # FIX:
        # Reset accumulated escape preparation time.
        self._escape_elapsed_ms = 0.0

        self._t_ms = 0.0

        self._locomotion_drive = 0.0
        self._locomotion_last_active_ms = -1e9

        self.events = []

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        dt_ms: float,
        channels: dict,
        t_ms: float,
        escape_laterality: float = 0.0,
        proboscis_drive: float = 0.0,
    ) -> BodyState:

        s = self.state

        dt_ms = max(0.0, float(dt_ms))
        t_ms = float(t_ms)

        self._t_ms = t_ms

        # --------------------------------------------------------------------
        # Read motor channels
        # --------------------------------------------------------------------

        takeoff = float(
            channels.get("escape_takeoff", 0.0)
        )

        long_mode = float(
            channels.get("escape_long_mode", 0.0)
        )

        freeze = float(
            channels.get("stop_freeze", 0.0)
        )

        # --------------------------------------------------------------------
        # Ground motor commands
        # --------------------------------------------------------------------
        # Preferred interface: four independent channels.
        #
        # Legacy channels remain supported so existing tests/callers continue
        # to work.
        has_four_channel_commands = any(
            key in channels
            for key in ("forward", "backward", "left", "right")
        )

        if has_four_channel_commands:
            forward_drive = float(channels.get("forward", 0.0))
            backward = float(channels.get("backward", 0.0))
            turn_left = float(channels.get("left", 0.0))
            turn_right = float(channels.get("right", 0.0))
        else:
            forward_drive = None
            backward = float(channels.get("backward_walk", 0.0))
            turn_bias = float(channels.get("turn_bias", 0.0))
            turn_left = max(0.0, -turn_bias)
            turn_right = max(0.0, turn_bias)

        locomotion_drive = channels.get(
            "locomotion_drive",
            None,
        )

        if forward_drive is not None:
            locomotion_drive = forward_drive

        # --------------------------------------------------------------------
        # Locomotion drive
        # --------------------------------------------------------------------

        if locomotion_drive is None:

            # Backwards-compatible fallback.
            #
            # If the readout does not provide a dedicated locomotion channel,
            # generic turn activity can still make the fly move.

            turn_drive = abs(
                float(channels.get("turn", 0.0))
            )

            locomotion_drive = max(
                turn_drive,
                abs(turn_bias),
            )

        locomotion_drive = max(
            0.0,
            min(1.0, float(locomotion_drive)),
        )

        if locomotion_drive >= MIN_LOCOMOTION_DRIVE:
            self._locomotion_last_active_ms = t_ms

        smoothing_alpha = min(
            1.0,
            dt_ms / 1000.0 / LOCOMOTION_SMOOTHING,
        )

        self._locomotion_drive += (
            locomotion_drive
            - self._locomotion_drive
        ) * smoothing_alpha

        if (
            self._locomotion_drive < MIN_LOCOMOTION_DRIVE
            and (
                t_ms
                - self._locomotion_last_active_ms
                <= LOCOMOTION_PERSISTENCE_MS
            )
        ):
            effective_locomotion = max(
                self._locomotion_drive,
                MIN_LOCOMOTION_DRIVE,
            )
        else:
            effective_locomotion = self._locomotion_drive

        # --------------------------------------------------------------------
        # Escape re-arming
        # --------------------------------------------------------------------

        if (
            self._escape_complete
            and takeoff < ESCAPE_THRESHOLD
            and long_mode < LONG_MODE_THRESHOLD
        ):
            self._escape_complete = False

        # --------------------------------------------------------------------
        # Arm escape
        # --------------------------------------------------------------------

        newly_armed = False

        if (
            self._escape_armed_at is None
            and not s.airborne
            and not self._escape_complete
        ):

            if takeoff >= ESCAPE_THRESHOLD:

                self._escape_armed_at = t_ms
                self._escape_mode = "short"

                # FIX:
                # Start a fresh preparation timer.
                self._escape_elapsed_ms = 0.0
                newly_armed = True

                self._log(
                    t_ms,
                    "GF (DNp01) escape command",
                    takeoff,
                )

            elif long_mode >= LONG_MODE_THRESHOLD:

                self._escape_armed_at = t_ms
                self._escape_mode = "long"

                # FIX:
                # Start a fresh preparation timer.
                self._escape_elapsed_ms = 0.0
                newly_armed = True

                self._log(
                    t_ms,
                    "long-mode escape command (DNp02/04/11)",
                    long_mode,
                )

        # --------------------------------------------------------------------
        # Escape preparation
        # --------------------------------------------------------------------
        #
        # FIXED TIMING MODEL:
        #
        # Escape preparation is now accumulated from dt_ms.
        #
        # The update that FIRST arms the escape does not count as preparation
        # time because the neural command only became active at the end of
        # that sampling interval.
        #
        # Every subsequent update adds its dt_ms to the preparation timer.
        #
        # This makes the state machine independent of how the caller chooses
        # its external timestep.
        # --------------------------------------------------------------------

        pending_takeoff = False
        flight_ms = 0.0
        directed_takeoff = False

        if (
            self._escape_armed_at is not None
            and not s.airborne
        ):

            if newly_armed:
                elapsed = self._escape_elapsed_ms
            else:
                elapsed = self._escape_elapsed_ms + dt_ms

            # Store the accumulated elapsed time.
            self._escape_elapsed_ms = elapsed

            s.escape_mode = self._escape_mode or ""

            if self._escape_mode == "short":

                s.behaviour = (
                    "escape (short mode, GF-driven)"
                )

                s.leg_extension = min(
                    1.0,
                    elapsed / GF_TAKEOFF_LATENCY_MS,
                )

                if elapsed >= GF_TAKEOFF_LATENCY_MS:

                    pending_takeoff = True
                    directed_takeoff = False

                    overshoot_ms = (
                        elapsed
                        - GF_TAKEOFF_LATENCY_MS
                    )

                    flight_ms = max(
                        0.0,
                        min(dt_ms, overshoot_ms),
                    )

            elif self._escape_mode == "long":

                s.behaviour = (
                    "escape (long mode, preparing)"
                )

                s.wing_angle_deg = (
                    90.0
                    * min(
                        1.0,
                        elapsed / LONG_MODE_PREP_MS,
                    )
                )

                s.leg_extension = (
                    min(
                        1.0,
                        elapsed / LONG_MODE_PREP_MS,
                    )
                )

                if elapsed >= LONG_MODE_PREP_MS:

                    pending_takeoff = True
                    directed_takeoff = True

                    overshoot_ms = (
                        elapsed
                        - LONG_MODE_PREP_MS
                    )

                    flight_ms = max(
                        0.0,
                        min(dt_ms, overshoot_ms),
                    )

        # --------------------------------------------------------------------
        # Advance physics: ground portion first (pre-takeoff, if any),
        # then the takeoff transition itself, then the airborne portion
        # (either the tail of this tick, or the whole tick if the fly was
        # already airborne coming in).
        # --------------------------------------------------------------------

        if pending_takeoff:
            ground_ms = max(0.0, dt_ms - flight_ms)
        elif not s.airborne:
            ground_ms = dt_ms
        else:
            ground_ms = 0.0

        remaining_ms = ground_ms

        while remaining_ms > 0.0:

            step_ms = min(
                remaining_ms,
                MAX_INTERNAL_STEP_MS,
            )

            self._advance_ground(
                step_ms,
                effective_locomotion,
                turn_left,
                turn_right,
                freeze,
                backward,
                proboscis_drive,
            )

            remaining_ms -= step_ms

        if pending_takeoff:

            self._takeoff(
                escape_laterality,
                directed=directed_takeoff,
                t_ms=t_ms,
            )

        if pending_takeoff:
            remaining_ms = flight_ms
        elif s.airborne:
            remaining_ms = dt_ms
        else:
            remaining_ms = 0.0

        while remaining_ms > 0.0:

            step_ms = min(
                remaining_ms,
                MAX_INTERNAL_STEP_MS,
            )

            step_s = step_ms / 1000.0

            self._advance_airborne(step_s, t_ms)

            remaining_ms -= step_ms

        return s

    # ========================================================================
    # GROUND
    # ========================================================================

    def _advance_ground(
        self,
        dt_ms: float,
        locomotion: float,
        turn_left: float,
        turn_right: float,
        freeze: float,
        backward: float,
        proboscis_drive: float,
    ) -> None:

        s = self.state

        dt_s = dt_ms / 1000.0

        # --------------------------------------------------------------------
        # Freeze
        # --------------------------------------------------------------------

        if freeze >= FREEZE_THRESHOLD:

            s.speed_mm_s = 0.0
            s.turn_rate_deg_s = 0.0

            s.behaviour = "freezing (DNp09)"

        # --------------------------------------------------------------------
        # Backward walking
        # --------------------------------------------------------------------

        elif backward >= BACKWARD_THRESHOLD:

            s.speed_mm_s = (
                -BACKWARD_SPEED_MM_S
                * backward
            )

            s.turn_rate_deg_s = (
                MAX_TURN_RATE_DEG_S
                * (turn_left - turn_right)
            )

            s.behaviour = "walking backward (MDN)"

        # --------------------------------------------------------------------
        # Normal walking
        # --------------------------------------------------------------------

        elif locomotion >= MIN_LOCOMOTION_DRIVE:

            normalized = (
                locomotion
                - MIN_LOCOMOTION_DRIVE
            ) / (
                1.0
                - MIN_LOCOMOTION_DRIVE
            )

            normalized = max(
                0.0,
                min(1.0, normalized),
            )

            drive = (
                normalized
                ** LOCOMOTION_RESPONSE_EXPONENT
            )

            s.speed_mm_s = (
                MAX_WALK_SPEED_MM_S
                * drive
            )

            s.turn_rate_deg_s = (
                MAX_TURN_RATE_DEG_S
                * (turn_left - turn_right)
            )

            if max(turn_left, turn_right) > 0.02:

                s.behaviour = (
                    "walking and turning %s"
                    % (
                        "left"
                        if turn_left > turn_right
                        else "right"
                    )
                )

            else:

                s.behaviour = "walking"

        # --------------------------------------------------------------------
        # Rest
        # --------------------------------------------------------------------

        else:

            s.speed_mm_s *= 0.90

            # Steering is an independent motor command and therefore remains
            # active even when there is no forward/backward locomotion drive.
            s.turn_rate_deg_s = (
                MAX_TURN_RATE_DEG_S
                * (turn_left - turn_right)
            )

            if max(turn_left, turn_right) > 0.02:
                s.behaviour = (
                    "turning left"
                    if turn_left > turn_right
                    else "turning right"
                )

            if abs(s.speed_mm_s) < 0.5:

                s.speed_mm_s = 0.0
                s.behaviour = "resting"

        # --------------------------------------------------------------------
        # Proboscis
        # --------------------------------------------------------------------

        target = (
            1.0
            if proboscis_drive >= PROBOSCIS_THRESHOLD
            else 0.0
        )

        s.proboscis_extension += (
            target
            - s.proboscis_extension
        ) * min(
            1.0,
            dt_ms / 40.0,
        )

        if (
            s.proboscis_extension > 0.5
            and s.behaviour == "resting"
        ):

            s.behaviour = (
                "proboscis extension (feeding)"
            )

        s.wing_angle_deg *= 0.92

        # --------------------------------------------------------------------
        # Ground translation
        # --------------------------------------------------------------------

        s.heading_deg = (
            s.heading_deg
            + s.turn_rate_deg_s * dt_s
        ) % 360.0

        self._translate(dt_s)

    # ========================================================================
    # AIRBORNE
    # ========================================================================

    def _advance_airborne(
        self,
        dt_s: float,
        t_ms: float,
    ) -> None:

        s = self.state

        s.vz_mm_s -= (
            self.GRAVITY_MM_S2
            * dt_s
        )

        s.z_mm += (
            s.vz_mm_s
            * dt_s
        )

        self._translate(dt_s)

        if s.z_mm <= 0.0:

            s.z_mm = 0.0
            s.vz_mm_s = 0.0
            s.airborne = False

            s.speed_mm_s = 0.0

            s.behaviour = "landed"

            s.leg_extension = 0.0

            self._escape_armed_at = None
            self._escape_mode = None
            self._escape_complete = True

            # FIX:
            # The completed escape no longer has preparation time attached.
            self._escape_elapsed_ms = 0.0

            self._log(
                t_ms,
                "landed",
                0.0,
            )

    # ========================================================================
    # TAKEOFF
    # ========================================================================

    def _takeoff(
        self,
        laterality: float,
        directed: bool,
        t_ms: float,
    ) -> None:

        s = self.state

        s.airborne = True

        s.z_mm = max(
            s.z_mm,
            0.001,
        )

        s.vz_mm_s = (
            JUMP_SPEED_MM_S
            * 0.6
        )

        s.speed_mm_s = JUMP_SPEED_MM_S

        s.wing_angle_deg = 90.0

        if directed:

            # Long-mode direction is biased according to escape laterality.
            s.heading_deg = (
                s.heading_deg
                - 90.0 * laterality
            ) % 360.0

        s.behaviour = (
            "airborne (%s-mode takeoff)"
            % self._escape_mode
        )

        self._log(
            t_ms,
            "takeoff (%s mode)"
            % self._escape_mode,
            1.0,
        )

    # ========================================================================
    # TRANSLATION
    # ========================================================================

    def _translate(
        self,
        dt_s: float,
    ) -> None:

        s = self.state

        rad = math.radians(
            s.heading_deg
        )

        s.x_mm += (
            math.cos(rad)
            * s.speed_mm_s
            * dt_s
        )

        s.y_mm += (
            math.sin(rad)
            * s.speed_mm_s
            * dt_s
        )

    # ========================================================================
    # EVENTS
    # ========================================================================

    def _log(
        self,
        t_ms: float,
        what: str,
        strength: float,
    ) -> None:

        self.events.append(
            {
                "t_ms": round(
                    float(t_ms),
                    1,
                ),
                "event": what,
                "strength": round(
                    float(strength),
                    3,
                ),
            }
        )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def as_dict(self) -> dict:

        d = asdict(self.state)

        d["events"] = self.events[-12:]

        return d

    # ========================================================================
    # PROVENANCE
    # ========================================================================

    @property
    def provenance(self) -> dict:

        return {
            "role": (
                "body model, downstream of all "
                "neural simulation"
            ),

            "inputs": (
                "descending-neuron motor readout providing four independent "
                "ground motor channels (forward/backward/left/right), "
                "escape channels, and proboscis motor neuron drive only; "
                "the body never sees the stimulus"
            ),

            "published_kinematics": {
                "short_mode_takeoff_latency_ms":
                    GF_TAKEOFF_LATENCY_MS,

                "long_mode_preparation_ms":
                    LONG_MODE_PREP_MS,

                "citation": (
                    "Card & Dickinson 2008, "
                    "J Exp Biol 211:341; "
                    "von Reyn et al. 2014, "
                    "Nat Neurosci 17:962"
                ),
            },

            "caveat": (
                "Body dynamics are a kinematic "
                "approximation, not a biomechanical "
                "model. Leg and wing motor neurons "
                "are in the ventral nerve cord and "
                "absent from FlyWire FAFB."
            ),

            "locomotion_caveat": (
                "locomotion_drive is an engineering "
                "interface. It must not be interpreted "
                "as a claim that a particular recorded "
                "descending neuron is the biological "
                "forward-walking command."
            ),

            "numerical_integration": {
                "max_internal_step_ms":
                    MAX_INTERNAL_STEP_MS
            },
        }   